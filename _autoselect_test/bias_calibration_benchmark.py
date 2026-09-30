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
        "prompt": "현재 체력은 10이고 적이 달려들고 있다.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "target": None,
    },
]

CALIBRATIONS = {
    "neutral": (
        "Calibration only: Ignore the context and field. There is intentionally no "
        "evidence favoring false or true. Treat both answers as equally plausible. "
        "Return exactly false or true."
    ),
    "coin": (
        "Calibration only: Before a fair coin is tossed, define false as heads and "
        "true as tails. Both outcomes are equally likely. Return exactly false or true."
    ),
    "unknown": (
        "Calibration only: There is no information about whether an unspecified "
        "statement is false or true. Return exactly false or true."
    ),
}

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
    return 1.0 / (1.0 + math.exp(-x))

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-bias-calibration",
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

all_results = []
for case in CASES:
    prompts = []
    meta = []

    for u in THRESHOLDS:
        threshold = case["minimum"] + u * (case["maximum"] - case["minimum"])
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
                    f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
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
        meta.append(("threshold", u, threshold))

    for name, question in CALIBRATIONS.items():
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
                    f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                    f"{question}"
                ),
            },
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + "Answer:")
        meta.append(("calibration", name, None))

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

    next_logits = output.logits[:, -1, :].float()
    threshold_log_odds = []
    calibration_bias = {}
    raw_probabilities = []
    threshold_rows = []

    for row, item in enumerate(meta):
        logit_false = float(next_logits[row, false_id].item())
        logit_true = float(next_logits[row, true_id].item())
        diff = logit_true - logit_false
        if item[0] == "threshold":
            raw_probabilities.append(sigmoid(diff))
            threshold_log_odds.append(diff)
            threshold_rows.append({
                "u": item[1],
                "threshold": item[2],
                "logit_diff": diff,
                "raw_probability_true": sigmoid(diff),
            })
        else:
            calibration_bias[item[1]] = diff

    methods = {"uncalibrated": 0.0}
    methods.update(calibration_bias)
    methods["mean_neutral_coin"] = (
        calibration_bias["neutral"] + calibration_bias["coin"]
    ) / 2.0
    methods["mean_all"] = sum(calibration_bias.values()) / len(calibration_bias)

    decoded = {}
    for method, bias in methods.items():
        probabilities = [sigmoid(diff - bias) for diff in threshold_log_odds]
        monotonic = isotonic_nonincreasing(probabilities)
        normalized = sum(monotonic) / len(monotonic)
        value = case["minimum"] + normalized * (case["maximum"] - case["minimum"])
        item = {
            "bias_logit": bias,
            "normalized": normalized,
            "value": value,
            "probabilities": probabilities,
            "probabilities_monotonic": monotonic,
        }
        if case["target"] is not None:
            item["absolute_error"] = abs(value - case["target"])
        decoded[method] = item

    all_results.append({
        "case": case["name"],
        "target": case["target"],
        "forward_calls": 1,
        "batch_size": len(prompts),
        "forward_seconds": seconds,
        "calibration_bias": calibration_bias,
        "thresholds": threshold_rows,
        "decoded": decoded,
    })

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "false_token_id": false_id,
    "true_token_id": true_id,
    "threshold_count": len(THRESHOLDS),
    "results": all_results,
}
Path("bias-calibration-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
