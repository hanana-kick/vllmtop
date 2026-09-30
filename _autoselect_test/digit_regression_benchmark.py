from __future__ import annotations

import hashlib
import json
import sys
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
        "minimum": -20.0,
        "maximum": 80.0,
        "target": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "target": 37750.0,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있다.",
        "minimum": 0.0,
        "maximum": 1.0,
        "target": None,
    },
]

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-digit-benchmark",
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
    if len(ids) != 1:
        raise RuntimeError(f"digit {digit} is not one token: {ids}")
    digit_ids.append(ids[0])

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.to("cpu")
model.eval()

def digit_position_name(width: int, position: int) -> str:
    if width == 2:
        return ("tens", "ones")[position]
    if width == 3:
        return ("hundreds", "tens", "ones")[position]
    raise ValueError(width)

rows = []
metadata = []
for case in CASES:
    for width in (2, 3):
        max_code = 10**width - 1
        for position in range(width):
            pos_name = digit_position_name(width, position)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You perform deterministic numeric normalization. "
                        "Return exactly one decimal digit 0 through 9."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{case['prompt']}\n\n"
                        f"Field range: [{case['minimum']}, {case['maximum']}].\n"
                        f"Linearly normalize the best field value into an integer from "
                        f"0 to {max_code} inclusive. Represent it as exactly {width} digits "
                        f"with leading zeros.\n"
                        f"Return only the {pos_name} digit of that {width}-digit normalized integer."
                    ),
                },
            ]
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rows.append(rendered)
            metadata.append((case, width, position))

encoded = tokenizer(
    rows,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)
started = time.perf_counter()
with torch.inference_mode():
    outputs = model(
        **encoded,
        use_cache=False,
        logits_to_keep=1,
    )
forward_seconds = time.perf_counter() - started

logits = outputs.logits[:, -1, :].float()
ids_tensor = torch.tensor(digit_ids, dtype=torch.long)

grouped = {}
for row_idx, (case, width, position) in enumerate(metadata):
    probs = torch.softmax(logits[row_idx, ids_tensor], dim=-1)
    winner = int(torch.argmax(probs).item())
    top3 = torch.topk(probs, k=3)
    key = (case["name"], width)
    grouped.setdefault(key, {
        "case": case,
        "width": width,
        "digits": [None] * width,
        "positions": [None] * width,
    })
    grouped[key]["digits"][position] = winner
    grouped[key]["positions"][position] = {
        "winner": winner,
        "winner_probability": float(probs[winner].item()),
        "top3": [
            {"digit": int(i.item()), "probability": float(p.item())}
            for p, i in zip(top3.values, top3.indices, strict=True)
        ],
    }

results = []
for (_case_name, width), group in grouped.items():
    case = group["case"]
    code = 0
    for digit in group["digits"]:
        code = code * 10 + int(digit)
    max_code = 10**width - 1
    normalized = code / max_code
    value = case["minimum"] + normalized * (case["maximum"] - case["minimum"])
    entry = {
        "case": case["name"],
        "digits": width,
        "digit_values": group["digits"],
        "normalized_code": code,
        "normalized": normalized,
        "value": value,
        "positions": group["positions"],
    }
    if case["target"] is not None:
        entry["target"] = case["target"]
        entry["absolute_error"] = abs(value - case["target"])
    results.append(entry)

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "digit_token_ids": digit_ids,
    "batch_size": len(rows),
    "forward_calls": 1,
    "forward_seconds": forward_seconds,
    "results": results,
}
Path("digit-regression-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
