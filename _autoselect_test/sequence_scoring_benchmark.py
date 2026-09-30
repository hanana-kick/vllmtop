from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "dc255159ae75b03b99200cd37a2d42ddc42b24d6"
WEIGHT = "model.safetensors-00001-of-00001.safetensors"
EXPECTED_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
CANDIDATE_COUNT = 9

CASES = [
    {
        "name": "temperature",
        "prompt": "The measured temperature is exactly 55 degrees Celsius.",
        "field": "temperature",
        "description": "Measured temperature in degrees Celsius.",
        "minimum": -20.0,
        "maximum": 80.0,
        "integer": False,
        "target": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000,
        "maximum": 50000,
        "integer": True,
        "target": 37750,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있어.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "integer": False,
        "target": None,
    },
]

def candidate_text(value):
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-sequence-scoring",
))
h = hashlib.sha256()
with (model_dir / WEIGHT).open("rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
assert h.hexdigest() == EXPECTED_SHA256

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.to("cpu")
model.eval()

results = []
for case in CASES:
    span = case["maximum"] - case["minimum"]
    candidates = []
    for i in range(CANDIDATE_COUNT):
        raw = case["minimum"] + span * i / (CANDIDATE_COUNT - 1)
        if case["integer"]:
            raw = int(round(raw))
        candidates.append(raw)

    options = ", ".join(candidate_text(v) for v in candidates)
    messages = [
        {
            "role": "system",
            "content": (
                "You estimate one numeric JSON field. Return only the numeric value, "
                "with no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}].\n"
                f"Choose the best estimate from these allowed numeric candidates:\n{options}\n"
                "Return exactly one candidate value."
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + "Answer:"
    prefix_ids = tokenizer.encode(rendered, add_special_tokens=False)

    rows = []
    suffixes = []
    for value in candidates:
        text = candidate_text(value)
        suffix_ids = tokenizer.encode(text, add_special_tokens=False)
        rows.append(prefix_ids + suffix_ids)
        suffixes.append((text, suffix_ids))

    max_len = max(map(len, rows))
    pad_id = tokenizer.pad_token_id
    batch_ids = []
    masks = []
    starts = []
    for row, (_text, suffix_ids) in zip(rows, suffixes, strict=True):
        pad = max_len - len(row)
        batch_ids.append([pad_id] * pad + row)
        masks.append([0] * pad + [1] * len(row))
        starts.append(pad + len(prefix_ids))

    input_ids = torch.tensor(batch_ids, dtype=torch.long)
    attention_mask = torch.tensor(masks, dtype=torch.long)

    started = time.perf_counter()
    with torch.inference_mode():
        base_output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    backbone_seconds = time.perf_counter() - started
    hidden = base_output.last_hidden_state

    sum_scores = []
    mean_scores = []
    per_token = []
    lm_head_seconds = 0.0
    for row_index, (_text, suffix_ids) in enumerate(suffixes):
        start = starts[row_index]
        prediction_positions = [
            start - 1 + j for j in range(len(suffix_ids))
        ]
        selected_hidden = hidden[row_index, prediction_positions, :]
        lm_started = time.perf_counter()
        logits = model.lm_head(selected_hidden).float()
        lm_head_seconds += time.perf_counter() - lm_started
        log_probs = torch.log_softmax(logits, dim=-1)
        token_ids = torch.tensor(suffix_ids, dtype=torch.long)
        token_log_probs = log_probs[
            torch.arange(len(suffix_ids)), token_ids
        ]
        scores = [float(v.item()) for v in token_log_probs]
        per_token.append(scores)
        total = sum(scores)
        sum_scores.append(total)
        mean_scores.append(total / len(scores))

    methods = {}
    for method_name, scores in (
        ("sum_logprob", sum_scores),
        ("mean_logprob", mean_scores),
    ):
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0)
        winner = int(torch.argmax(score_tensor).item())
        expected = sum(
            float(probs[i].item()) * float(candidates[i])
            for i in range(len(candidates))
        )
        item = {
            "winner_index": winner,
            "winner_value": candidates[winner],
            "expected_value": expected,
            "probabilities": [float(p.item()) for p in probs],
            "scores": scores,
        }
        if case["target"] is not None:
            item["winner_error"] = abs(float(candidates[winner]) - float(case["target"]))
            item["expected_error"] = abs(expected - float(case["target"]))
        methods[method_name] = item

    results.append({
        "case": case["name"],
        "target": case["target"],
        "candidates": candidates,
        "candidate_texts": [x[0] for x in suffixes],
        "candidate_token_lengths": [len(x[1]) for x in suffixes],
        "batch_size": len(candidates),
        "forward_calls": 1,
        "backbone_seconds": backbone_seconds,
        "lm_head_seconds": lm_head_seconds,
        "token_logprobs": per_token,
        "methods": methods,
    })

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "candidate_count": CANDIDATE_COUNT,
    "results": results,
}
Path("sequence-scoring-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
