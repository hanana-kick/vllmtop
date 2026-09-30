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

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def logit(p):
    return math.log(p / (1.0 - p))

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-anchor-calibration",
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
    query_thresholds = [
        ("anchor_true", case["minimum"] - span),
        *[("threshold", case["minimum"] + u * span) for u in THRESHOLDS],
        ("anchor_false", case["maximum"] + span),
    ]

    prompts = []
    for kind, threshold in query_thresholds:
        messages = [
            {
                "role": "system",
                "content": "You estimate numeric values from context. Output only the requested boolean.",
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{case['prompt']}\n\n"
                    f"Field: {case['field']}\n"
                    f"Field description: {case['description']}\n"
                    f"Allowed range: [{case['minimum']}, {case['maximum']}]. "
                    "The best value must stay inside this range.\n"
                    f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
                    "Return exactly false or true."
                ),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + "Answer:")

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

    diffs = []
    for row in range(len(prompts)):
        diffs.append(
            float(logits[row, true_id].item() - logits[row, false_id].item())
        )

    d_true = diffs[0]
    d_false = diffs[-1]
    interior = diffs[1:-1]

    decoded = {}

    raw_probs = [sigmoid(d) for d in interior]
    raw_mono = isotonic_nonincreasing(raw_probs)
    raw_u = sum(raw_mono) / len(raw_mono)
    decoded["uncalibrated"] = {
        "normalized": raw_u,
        "value": case["minimum"] + raw_u * span,
        "probabilities": raw_probs,
    }

    denom = d_true - d_false
    if abs(denom) > 1e-8:
        linear_probs = [min(1.0, max(0.0, (d - d_false) / denom)) for d in interior]
        linear_mono = isotonic_nonincreasing(linear_probs)
        linear_u = sum(linear_mono) / len(linear_mono)
        decoded["anchor_linear"] = {
            "normalized": linear_u,
            "value": case["minimum"] + linear_u * span,
            "probabilities": linear_probs,
            "probabilities_monotonic": linear_mono,
        }

        for target in (0.9, 0.95, 0.99):
            hi = logit(target)
            lo = logit(1.0 - target)
            scale = (hi - lo) / (d_true - d_false)
            offset = hi - scale * d_true
            probs = [sigmoid(scale * d + offset) for d in interior]
            mono = isotonic_nonincreasing(probs)
            u = sum(mono) / len(mono)
            decoded[f"anchor_logistic_{target}"] = {
                "normalized": u,
                "value": case["minimum"] + u * span,
                "probabilities": probs,
                "probabilities_monotonic": mono,
                "scale": scale,
                "offset": offset,
            }

    for item in decoded.values():
        if case["target"] is not None:
            item["absolute_error"] = abs(item["value"] - case["target"])

    results.append({
        "case": case["name"],
        "target": case["target"],
        "batch_size": len(prompts),
        "forward_calls": 1,
        "forward_seconds": seconds,
        "anchor_true_threshold": query_thresholds[0][1],
        "anchor_false_threshold": query_thresholds[-1][1],
        "anchor_true_logit_diff": d_true,
        "anchor_false_logit_diff": d_false,
        "interior_logit_diffs": interior,
        "decoded": decoded,
    })

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "threshold_count": len(THRESHOLDS),
    "results": results,
}
Path("anchor-calibration-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
