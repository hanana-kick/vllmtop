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
        "expected": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "expected": 37750.0,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있다.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "expected": None,
    },
]


def isotonic(values):
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) >= 2:
            a = blocks[-2][0] / blocks[-2][1]
            b = blocks[-1][0] / blocks[-1][1]
            if a >= b:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return out


def render(tokenizer, case, threshold, swapped, variant):
    values = (True, False) if swapped else (False, True)
    mapping = f"A = {'true' if values[0] else 'false'}\nB = {'true' if values[1] else 'false'}"
    if variant == "current":
        instruction = (
            f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
        )
    elif variant == "exact-aware":
        instruction = (
            "First determine the field value from the context. "
            "If the context explicitly states the field value, use that exact value; "
            "only infer a value when it is not explicitly stated.\n"
            f"Question: Is that field value greater than or equal to {threshold}?\n"
        )
    else:
        raise ValueError(variant)
    messages = [
        {"role": "system", "content": "You are a deterministic classifier. Output only A or B."},
        {
            "role": "user",
            "content": (
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                f"{instruction}"
                f"{mapping}\nReturn exactly A or B."
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ) + "Answer:"


def forward(model, tokenizer, prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    t = time.perf_counter()
    with torch.inference_mode():
        out = model(**encoded, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float(), time.perf_counter() - t


model_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir="qwen3.5-0.8b-prompt-variant"))
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

id_a = tokenizer.encode("A", add_special_tokens=False)[0]
id_b = tokenizer.encode("B", add_special_tokens=False)[0]

results = []
for case in CASES:
    entry = {"name": case["name"], "expected": case["expected"], "variants": {}}
    norms = [(i + 0.5) / 4 for i in range(4)]
    thresholds = [case["minimum"] + u * (case["maximum"] - case["minimum"]) for u in norms]
    for variant in ("current", "exact-aware"):
        prompts = []
        for threshold in thresholds:
            prompts.append(render(tokenizer, case, threshold, False, variant))
            prompts.append(render(tokenizer, case, threshold, True, variant))
        logits, seconds = forward(model, tokenizer, prompts)
        probs = []
        for i in range(4):
            normal = logits[i * 2, torch.tensor([id_a, id_b])]
            swapped = logits[i * 2 + 1, torch.tensor([id_a, id_b])]
            d1 = float((normal[1] - normal[0]).item())
            d2 = float((swapped[0] - swapped[1]).item())
            d = 0.5 * (d1 + d2)
            probs.append(1.0 / (1.0 + math.exp(-d)))
        mono = isotonic(probs)
        u = sum(mono) / len(mono)
        value = case["minimum"] + u * (case["maximum"] - case["minimum"])
        payload = {
            "value": value,
            "normalized": u,
            "seconds": seconds,
            "probabilities": probs,
            "monotonic": mono,
        }
        if case["expected"] is not None:
            payload["absolute_error"] = abs(value - case["expected"])
        entry["variants"][variant] = payload
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "weights_sha256": h.hexdigest(),
    "results": results,
}
Path("prompt-variant-result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
