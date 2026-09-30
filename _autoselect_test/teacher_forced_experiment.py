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

CASES = [
    {
        "name": "temperature",
        "prompt": "The measured temperature is exactly 55 degrees Celsius.",
        "field": "temperature",
        "description": "Measured temperature in degrees Celsius.",
        "minimum": -20.0,
        "maximum": 80.0,
        "integer": False,
        "expected": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "integer": True,
        "expected": 37750.0,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있다.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "integer": False,
        "expected": None,
    },
]


def format_number(value, integer):
    if integer:
        return str(int(round(value)))
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return format(value, ".12g")


def render_prompt(tokenizer, case):
    messages = [
        {
            "role": "system",
            "content": "Return only the numeric value requested by the user. Do not explain.",
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                "Choose the single best numeric value for this field. "
                "If an exact field value is explicitly stated in the context, use it exactly."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + "Answer:"


def lcp_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def score_candidates(model, tokenizer, prompt, candidates):
    base_ids = tokenizer.encode(prompt, add_special_tokens=False)
    rows = []
    suffix_lengths = []
    lcps = []
    candidate_texts = []
    for value in candidates:
        text = format_number(value, isinstance(value, int))
        ids = tokenizer.encode(prompt + text, add_special_tokens=False)
        lcp = lcp_len(base_ids, ids)
        if lcp == 0:
            raise RuntimeError("candidate destroyed entire prompt tokenization")
        suffix_len = len(ids) - lcp
        if suffix_len < 1:
            raise RuntimeError("empty candidate suffix")
        rows.append(ids)
        suffix_lengths.append(suffix_len)
        lcps.append(lcp)
        candidate_texts.append(text)

    max_len = max(len(ids) for ids in rows)
    max_suffix = max(suffix_lengths)
    keep = max_suffix + 1

    input_ids = []
    attention_mask = []
    pad_id = tokenizer.pad_token_id
    for ids in rows:
        pad = max_len - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

    started = time.perf_counter()
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=keep,
        )
    seconds = time.perf_counter() - started

    logits = out.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = []
    for row, ids in enumerate(rows):
        suffix_len = suffix_lengths[row]
        lcp = lcps[row]
        token_logps = []
        for j in range(suffix_len):
            relative_prediction_pos = keep - suffix_len + j - 1
            if relative_prediction_pos < 0:
                raise RuntimeError("not enough kept logits")
            target_token = ids[lcp + j]
            token_logps.append(float(log_probs[row, relative_prediction_pos, target_token].item()))
        scores.append({
            "candidate": candidates[row],
            "text": candidate_texts[row],
            "tokens": ids[lcp:],
            "token_logprobs": token_logps,
            "sum_logprob": sum(token_logps),
            "mean_logprob": sum(token_logps) / len(token_logps),
        })
    return scores, seconds, keep


model_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir="qwen3.5-0.8b-teacher-forced"))
h = hashlib.sha256()
with (model_dir / WEIGHT).open("rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
assert h.hexdigest() == EXPECTED_SHA256

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.to("cpu").eval()

results = []
for case in CASES:
    values = []
    for i in range(9):
        u = i / 8
        raw = case["minimum"] + u * (case["maximum"] - case["minimum"])
        value = int(round(raw)) if case["integer"] else raw
        values.append(value)

    prompt = render_prompt(tokenizer, case)
    scores, seconds, keep = score_candidates(model, tokenizer, prompt, values)
    best_sum = max(scores, key=lambda x: x["sum_logprob"])
    best_mean = max(scores, key=lambda x: x["mean_logprob"])

    entry = {
        "name": case["name"],
        "expected": case["expected"],
        "candidates": values,
        "batch_size": len(values),
        "logits_to_keep": keep,
        "forward_calls": 1,
        "forward_seconds": seconds,
        "best_by_sum": best_sum,
        "best_by_mean": best_mean,
        "scores": scores,
    }
    if case["expected"] is not None:
        entry["sum_abs_error"] = abs(float(best_sum["candidate"]) - case["expected"])
        entry["mean_abs_error"] = abs(float(best_mean["candidate"]) - case["expected"])
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "weights_sha256": h.hexdigest(),
    "strategy": "teacher-forced multi-token numeric candidate scoring",
    "results": results,
}
Path("teacher-forced-result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
