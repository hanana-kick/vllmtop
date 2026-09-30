from __future__ import annotations

import hashlib
import json
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
        "target": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "target": 37750.0,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있어.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "target": None,
    },
]

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-bucket-benchmark",
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

digit_ids = []
for digit in range(10):
    ids = tokenizer.encode(str(digit), add_special_tokens=False)
    assert len(ids) == 1
    digit_ids.append(ids[0])

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.to("cpu")
model.eval()

prompts = []
meta = []
for case in CASES:
    lo = case["minimum"]
    hi = case["maximum"]
    span = hi - lo
    edges = [lo + span * i / 10 for i in range(11)]
    lines = []
    for i in range(10):
        left = edges[i]
        right = edges[i + 1]
        bracket = "]" if i == 9 else ")"
        lines.append(f"{i} = [{left}, {right}{bracket}")
    messages = [
        {
            "role": "system",
            "content": (
                "You classify a numeric field into one of ten explicit numeric intervals. "
                "Output exactly one digit 0 through 9."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{lo}, {hi}]\n\n"
                "Choose the interval containing the best value for this field:\n"
                + "\n".join(lines)
                + "\n\nReturn only the interval digit."
            ),
        },
    ]
    prompts.append(tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + "Answer:")
    meta.append((case, edges))

encoded = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)
started = time.perf_counter()
with torch.inference_mode():
    output = model(**encoded, use_cache=False, logits_to_keep=1)
seconds = time.perf_counter() - started

logits = output.logits[:, -1, :].float()
ids = torch.tensor(digit_ids, dtype=torch.long)
results = []
for row, (case, edges) in enumerate(meta):
    probs = torch.softmax(logits[row, ids], dim=-1)
    winner = int(torch.argmax(probs).item())
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(10)]
    argmax_value = centers[winner]
    expected_value = sum(float(probs[i].item()) * centers[i] for i in range(10))
    normalized_expectation = sum(float(probs[i].item()) * ((i + 0.5) / 10) for i in range(10))
    top5 = torch.topk(probs, k=5)
    entry = {
        "case": case["name"],
        "winner_bucket": winner,
        "argmax_value": argmax_value,
        "expected_value": expected_value,
        "normalized_expectation": normalized_expectation,
        "probabilities": [float(probs[i].item()) for i in range(10)],
        "top5": [
            {"bucket": int(i.item()), "probability": float(p.item())}
            for p, i in zip(top5.values, top5.indices, strict=True)
        ],
    }
    if case["target"] is not None:
        entry["target"] = case["target"]
        entry["argmax_error"] = abs(argmax_value - case["target"])
        entry["expected_error"] = abs(expected_value - case["target"])
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "batch_size": len(prompts),
    "forward_calls": 1,
    "forward_seconds": seconds,
    "digit_token_ids": digit_ids,
    "results": results,
}
Path("bucket-regression-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
