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

PAIRS = {
    "false_true": {
        "low": "false",
        "high": "true",
        "question": lambda t: (
            f"Is the best value for this field greater than or equal to {t}? "
            "Return exactly false or true."
        ),
    },
    "no_yes": {
        "low": "no",
        "high": "yes",
        "question": lambda t: (
            f"Is the best value for this field greater than or equal to {t}? "
            "Return exactly no or yes."
        ),
    },
    "below_above": {
        "low": "below",
        "high": "above",
        "question": lambda t: (
            f"Relative to threshold {t}, is the best value below the threshold or "
            "at/above the threshold? Return exactly below or above."
        ),
    },
    "lower_higher": {
        "low": "lower",
        "high": "higher",
        "question": lambda t: (
            f"Relative to threshold {t}, is the best value lower than the threshold or "
            "at least as high as the threshold? Return exactly lower or higher."
        ),
    },
    "low_high": {
        "low": "low",
        "high": "high",
        "question": lambda t: (
            f"Relative to threshold {t}, classify the best value as low if it is below "
            "the threshold, otherwise high. Return exactly low or high."
        ),
    },
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

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-token-pair-benchmark",
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

pair_tokens = {}
for name, pair in PAIRS.items():
    low_ids = tokenizer.encode(pair["low"], add_special_tokens=False)
    high_ids = tokenizer.encode(pair["high"], add_special_tokens=False)
    pair_tokens[name] = {
        "low": low_ids,
        "high": high_ids,
        "usable": len(low_ids) == 1 and len(high_ids) == 1 and low_ids[0] != high_ids[0],
    }

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.to("cpu")
model.eval()

results = []
for pair_name, pair in PAIRS.items():
    token_info = pair_tokens[pair_name]
    if not token_info["usable"]:
        continue
    low_id = token_info["low"][0]
    high_id = token_info["high"][0]

    for case in CASES:
        prompts = []
        threshold_values = []
        for u in THRESHOLDS:
            t = case["minimum"] + u * (case["maximum"] - case["minimum"])
            threshold_values.append(t)
            messages = [
                {
                    "role": "system",
                    "content": "You estimate numeric values from context. Output only the requested class.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{case['prompt']}\n\n"
                        f"Field: {case['field']}\n"
                        f"Field description: {case['description']}\n"
                        f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                        f"{pair['question'](t)}"
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

        probabilities = []
        for row in range(len(prompts)):
            pair_logits = logits[row, torch.tensor([low_id, high_id])]
            probs = torch.softmax(pair_logits, dim=-1)
            probabilities.append(float(probs[1].item()))

        monotonic = isotonic_nonincreasing(probabilities)
        normalized = sum(monotonic) / len(monotonic)
        value = case["minimum"] + normalized * (case["maximum"] - case["minimum"])
        entry = {
            "pair": pair_name,
            "low_text": pair["low"],
            "high_text": pair["high"],
            "low_token_id": low_id,
            "high_token_id": high_id,
            "case": case["name"],
            "batch_size": len(prompts),
            "forward_calls": 1,
            "forward_seconds": seconds,
            "probabilities_high": probabilities,
            "probabilities_high_monotonic": monotonic,
            "normalized": normalized,
            "value": value,
        }
        if case["target"] is not None:
            entry["target"] = case["target"]
            entry["absolute_error"] = abs(value - case["target"])
        results.append(entry)

payload = {
    "pair_tokens": pair_tokens,
    "threshold_count": len(THRESHOLDS),
    "results": results,
}
Path("token-pair-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
