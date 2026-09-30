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


def isotonic_nonincreasing(values):
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) >= 2:
            left = blocks[-2][0] / blocks[-2][1]
            right = blocks[-1][0] / blocks[-1][1]
            if left >= right:
                break
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0] + b[0], a[1] + b[1]])
    out = []
    for total, count in blocks:
        out.extend([total / count] * count)
    return out


def render(tokenizer, case, threshold, mode):
    if mode == "direct":
        system = "You estimate numeric values from context. Output only the requested boolean."
        body = (
            f"Context:\n{case['prompt']}\n\n"
            f"Field: {case['field']}\n"
            f"Field description: {case['description']}\n"
            f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
            f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
            "Return exactly false or true."
        )
    elif mode == "ab_normal":
        system = "You are a deterministic classifier. Output only A or B."
        body = (
            f"Context:\n{case['prompt']}\n\n"
            f"Field: {case['field']}\n"
            f"Field description: {case['description']}\n"
            f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
            f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
            "A = false\nB = true\nReturn exactly A or B."
        )
    elif mode == "ab_swapped":
        system = "You are a deterministic classifier. Output only A or B."
        body = (
            f"Context:\n{case['prompt']}\n\n"
            f"Field: {case['field']}\n"
            f"Field description: {case['description']}\n"
            f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
            f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
            "A = true\nB = false\nReturn exactly A or B."
        )
    else:
        raise ValueError(mode)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text + "Answer:"


def token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"{text!r} is not one token: {ids}")
    return ids[0]


def forward(model, tokenizer, prompts):
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    started = time.perf_counter()
    with torch.inference_mode():
        out = model(**encoded, use_cache=False, logits_to_keep=1)
    return out.logits[:, -1, :].float(), time.perf_counter() - started


model_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir="qwen3.5-0.8b-label-swap"))
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

id_false = token_id(tokenizer, "false")
id_true = token_id(tokenizer, "true")
id_a = token_id(tokenizer, "A")
id_b = token_id(tokenizer, "B")

results = []
for case in CASES:
    # Baseline: 8 direct thresholds.
    direct_norms = [(i + 0.5) / 8 for i in range(8)]
    direct_thresholds = [
        case["minimum"] + u * (case["maximum"] - case["minimum"]) for u in direct_norms
    ]
    direct_prompts = [render(tokenizer, case, t, "direct") for t in direct_thresholds]
    direct_logits, direct_seconds = forward(model, tokenizer, direct_prompts)
    direct_probs = []
    for row in range(len(direct_prompts)):
        pair = direct_logits[row, torch.tensor([id_false, id_true])]
        direct_probs.append(float(torch.softmax(pair, dim=-1)[1].item()))
    direct_mono = isotonic_nonincreasing(direct_probs)
    direct_u = sum(direct_mono) / len(direct_mono)
    direct_value = case["minimum"] + direct_u * (case["maximum"] - case["minimum"])

    # Label-swap: 4 thresholds x 2 mappings = same batch size 8.
    swap_norms = [(i + 0.5) / 4 for i in range(4)]
    swap_thresholds = [
        case["minimum"] + u * (case["maximum"] - case["minimum"]) for u in swap_norms
    ]
    swap_prompts = []
    for t in swap_thresholds:
        swap_prompts.append(render(tokenizer, case, t, "ab_normal"))
        swap_prompts.append(render(tokenizer, case, t, "ab_swapped"))
    swap_logits, swap_seconds = forward(model, tokenizer, swap_prompts)
    swap_probs = []
    swap_details = []
    for i, t in enumerate(swap_thresholds):
        normal = swap_logits[i * 2, torch.tensor([id_a, id_b])]
        swapped = swap_logits[i * 2 + 1, torch.tensor([id_a, id_b])]
        d_normal = float((normal[1] - normal[0]).item())
        d_swapped = float((swapped[0] - swapped[1]).item())
        semantic_log_odds = 0.5 * (d_normal + d_swapped)
        p = 1.0 / (1.0 + math.exp(-semantic_log_odds))
        swap_probs.append(p)
        swap_details.append({
            "threshold": t,
            "normal_true_logodds": d_normal,
            "swapped_true_logodds": d_swapped,
            "calibrated_probability_ge": p,
        })
    swap_mono = isotonic_nonincreasing(swap_probs)
    swap_u = sum(swap_mono) / len(swap_mono)
    swap_value = case["minimum"] + swap_u * (case["maximum"] - case["minimum"])

    entry = {
        "name": case["name"],
        "expected": case["expected"],
        "baseline_direct8": {
            "value": direct_value,
            "normalized": direct_u,
            "forward_seconds": direct_seconds,
            "probabilities": direct_probs,
            "monotonic": direct_mono,
        },
        "label_swap4x2": {
            "value": swap_value,
            "normalized": swap_u,
            "forward_seconds": swap_seconds,
            "probabilities": swap_probs,
            "monotonic": swap_mono,
            "details": swap_details,
        },
    }
    if case["expected"] is not None:
        entry["baseline_direct8"]["absolute_error"] = abs(direct_value - case["expected"])
        entry["label_swap4x2"]["absolute_error"] = abs(swap_value - case["expected"])
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "weights_sha256": h.hexdigest(),
    "token_ids": {"false": id_false, "true": id_true, "A": id_a, "B": id_b},
    "results": results,
}
Path("label-swap-result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
