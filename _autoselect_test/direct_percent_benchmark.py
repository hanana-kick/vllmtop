from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path("_autoselect_test/src").resolve()))
from autoselect.selector import QwenSingleForwardSelector

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
    local_dir="qwen3.5-0.8b-direct-percent",
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

candidate_texts = [str(i) for i in range(101)]
candidate_ids = []
missing = {}
for text in candidate_texts:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        missing[text] = ids
    else:
        candidate_ids.append(ids[0])

token_report = {
    "single_token_count": 101 - len(missing),
    "missing": missing,
    "unique_token_count": len(set(candidate_ids)),
}

prefills = {
    "none": "",
    "answer_colon": "Answer:",
    "value_colon": "Value:",
}

rows = []
row_meta = []
for case in CASES:
    for prefill_name, prefill in prefills.items():
        messages = [
            {
                "role": "system",
                "content": (
                    "You estimate a numeric field from context. "
                    "Return only one integer from 0 to 100 inclusive."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{case['prompt']}\n\n"
                    f"The field has allowed range [{case['minimum']}, {case['maximum']}].\n"
                    "Map the best value linearly into the normalized scale 0..100, "
                    "where 0 means the minimum and 100 means the maximum.\n"
                    "Return only the normalized integer."
                ),
            },
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ) + prefill
        rows.append(rendered)
        row_meta.append((case, prefill_name))

if missing:
    raise RuntimeError(f"not all 0..100 candidates are single tokens: {missing}")

encoded = tokenizer(
    rows,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)
with torch.inference_mode():
    started = time.perf_counter()
    outputs = model(
        **encoded,
        use_cache=False,
        logits_to_keep=1,
    )
    forward_seconds = time.perf_counter() - started

logits = outputs.logits[:, -1, :].float()
ids_tensor = torch.tensor(candidate_ids, dtype=torch.long)

results = []
for row_index, (case, prefill_name) in enumerate(row_meta):
    candidate_logits = logits[row_index, ids_tensor]
    probs = torch.softmax(candidate_logits, dim=-1)
    winner = int(torch.argmax(candidate_logits).item())
    pct = winner
    value = case["minimum"] + (pct / 100.0) * (case["maximum"] - case["minimum"])
    topk = torch.topk(probs, k=5)
    entry = {
        "case": case["name"],
        "prefill": prefill_name,
        "normalized_integer": pct,
        "value": value,
        "top5": [
            {
                "candidate": int(index.item()),
                "probability": float(prob.item()),
            }
            for prob, index in zip(topk.values, topk.indices, strict=True)
        ],
    }
    if case["target"] is not None:
        entry["target"] = case["target"]
        entry["absolute_error"] = abs(value - case["target"])
    results.append(entry)

# Baseline current autoselect, one request per case.
baseline = []
selector = QwenSingleForwardSelector(
    model_id=str(model_dir),
    dtype="bfloat16",
    numeric_thresholds=8,
)
for case in CASES:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "minimum": case["minimum"],
                "maximum": case["maximum"],
            }
        },
    }
    started = time.perf_counter()
    selected = selector.select(case["prompt"], schema)
    seconds = time.perf_counter() - started
    item = {
        "case": case["name"],
        "value": selected.value["value"],
        "batch_size": selected.batch_size,
        "forward_calls": selected.forward_calls,
        "seconds": seconds,
    }
    if case["target"] is not None:
        item["target"] = case["target"]
        item["absolute_error"] = abs(selected.value["value"] - case["target"])
    baseline.append(item)

payload = {
    "token_report": token_report,
    "direct_percent": {
        "batch_size": len(rows),
        "single_forward_seconds": forward_seconds,
        "results": results,
    },
    "threshold_baseline": baseline,
}
Path("direct-percent-benchmark.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
