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
THRESHOLDS = [(i + 0.5) / 8 for i in range(8)]

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

def isotonic_nonincreasing(values):
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) >= 2:
            a = blocks[-2][0] / blocks[-2][1]
            b = blocks[-1][0] / blocks[-1][1]
            if a >= b:
                break
            r = blocks.pop()
            l = blocks.pop()
            blocks.append([l[0] + r[0], l[1] + r[1]])
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return out

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-packed-probe",
))
h = hashlib.sha256()
with (model_dir / WEIGHT).open("rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
assert h.hexdigest() == EXPECTED_SHA256

tokenizer = AutoTokenizer.from_pretrained(model_dir)
false_ids = tokenizer.encode("false", add_special_tokens=False)
true_ids = tokenizer.encode("true", add_special_tokens=False)
assert len(false_ids) == 1 and len(true_ids) == 1
false_id, true_id = false_ids[0], true_ids[0]

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
    actual_thresholds = [case["minimum"] + u * span for u in THRESHOLDS]
    question_lines = "\n".join(
        f"Q{i+1}: Is the best value greater than or equal to {threshold}?"
        for i, threshold in enumerate(actual_thresholds)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You estimate one numeric field. Each question is independent. "
                "At every answer slot, the only valid answers are false or true."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}].\n\n"
                "There are eight independent threshold questions. Answer each in its matching slot.\n"
                f"{question_lines}"
            ),
        },
    ]
    base_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(base_text, add_special_tokens=False)
    answer_positions = []

    # Construct token IDs segment-by-segment so answer slot positions are stable.
    for i in range(8):
        slot_ids = tokenizer.encode(f"A{i+1}:", add_special_tokens=False)
        input_ids.extend(slot_ids)
        answer_positions.append(len(input_ids) - 1)
        if i != 7:
            input_ids.extend(tokenizer.encode("\n", add_special_tokens=False))

    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    positions = torch.tensor(answer_positions, dtype=torch.long)

    started = time.perf_counter()
    with torch.inference_mode():
        output = model(
            input_ids=input_tensor,
            use_cache=False,
            logits_to_keep=positions,
        )
    seconds = time.perf_counter() - started

    # Shape [1, 8, vocab]
    selected_logits = output.logits[0].float()
    probabilities = []
    for row in range(8):
        pair_logits = selected_logits[row, torch.tensor([false_id, true_id])]
        probs = torch.softmax(pair_logits, dim=-1)
        probabilities.append(float(probs[1].item()))

    monotonic = isotonic_nonincreasing(probabilities)
    normalized = sum(monotonic) / len(monotonic)
    value = case["minimum"] + normalized * span
    entry = {
        "case": case["name"],
        "sequence_length": len(input_ids),
        "answer_positions": answer_positions,
        "forward_calls": 1,
        "batch_size": 1,
        "logit_positions": len(answer_positions),
        "forward_seconds": seconds,
        "probabilities_true": probabilities,
        "probabilities_true_monotonic": monotonic,
        "normalized": normalized,
        "value": value,
    }
    if case["target"] is not None:
        entry["target"] = case["target"]
        entry["absolute_error"] = abs(value - case["target"])
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "false_token_id": false_id,
    "true_token_id": true_id,
    "results": results,
}
Path("packed-probe-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
