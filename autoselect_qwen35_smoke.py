from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "dc255159ae75b03b99200cd37a2d42ddc42b24d6"
EXPECTED_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
WEIGHT_NAME = "model.safetensors-00001-of-00001.safetensors"

prompt = "현재 체력은 10이고 적이 달려들고 있다."
fields = [
    (
        "danger_level",
        [round(i / 10, 1) for i in range(11)],
        "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
    ),
    ("run", [False, True], "즉시 도망쳐야 하는지 여부."),
]

started = time.perf_counter()
model_dir = Path(
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir="qwen3.5-0.8b",
    )
)
download_done = time.perf_counter()

weight_path = model_dir / WEIGHT_NAME
h = hashlib.sha256()
with weight_path.open("rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
weight_sha256 = h.hexdigest()
weight_bytes = weight_path.stat().st_size
if weight_sha256 != EXPECTED_SHA256:
    raise RuntimeError(
        f"weight SHA256 mismatch: expected {EXPECTED_SHA256}, got {weight_sha256}"
    )
hash_done = time.perf_counter()

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
load_done = time.perf_counter()

labels = [chr(ord("A") + i) for i in range(26)]
rendered_prompts: list[str] = []
label_token_ids: list[list[int]] = []

for path, candidates, description in fields:
    options = "\n".join(
        f"{label} = {json.dumps(value, ensure_ascii=False)}"
        for label, value in zip(labels, candidates, strict=False)
    )
    messages = [
        {
            "role": "system",
            "content": "You are a deterministic classifier. Output exactly one provided label.",
        },
        {
            "role": "user",
            "content": (
                "Select the single best value for the requested JSON field from the finite options.\n"
                "Return exactly one option label and nothing else.\n\n"
                f"Input:\n{prompt}\n\n"
                f"Field: {path}\n"
                f"Field description: {description}\n"
                f"Options:\n{options}"
            ),
        },
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base_ids = tokenizer.encode(rendered, add_special_tokens=False)
    ids: list[int] = []
    for label in labels[: len(candidates)]:
        full_ids = tokenizer.encode(rendered + label, add_special_tokens=False)
        if full_ids[: len(base_ids)] == base_ids and len(full_ids) == len(base_ids) + 1:
            ids.append(full_ids[-1])
        else:
            direct = tokenizer.encode(label, add_special_tokens=False)
            if len(direct) != 1:
                raise RuntimeError(f"label {label!r} is not a one-token candidate")
            ids.append(direct[0])
    rendered_prompts.append(rendered)
    label_token_ids.append(ids)

encoded = tokenizer(
    rendered_prompts,
    return_tensors="pt",
    padding=True,
    add_special_tokens=False,
)
encoded = {key: value.to("cpu") for key, value in encoded.items()}

forward_calls = 0
forward_started = time.perf_counter()
with torch.inference_mode():
    forward_calls += 1
    outputs = model(
        **encoded,
        use_cache=False,
        logits_to_keep=1,
    )
forward_done = time.perf_counter()

if forward_calls != 1:
    raise RuntimeError(f"expected exactly 1 forward call, got {forward_calls}")

next_logits = outputs.logits[:, -1, :].float()
value = {}
details = []

for row, (path, candidates, _description) in enumerate(fields):
    ids = torch.tensor(label_token_ids[row], dtype=torch.long)
    candidate_logits = next_logits[row, ids]
    probs = torch.softmax(candidate_logits, dim=-1)
    winner = int(torch.argmax(candidate_logits).item())
    value[path] = candidates[winner]
    details.append(
        {
            "path": path,
            "selected_label": labels[winner],
            "selected_value": candidates[winner],
            "probability_among_candidates": float(probs[winner].item()),
            "candidate_probabilities": {
                labels[i]: float(probs[i].item()) for i in range(len(candidates))
            },
            "candidate_token_ids": label_token_ids[row],
        }
    )

result = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "model_class": model.__class__.__name__,
    "parameters": sum(p.numel() for p in model.parameters()),
    "dtype": str(next(model.parameters()).dtype),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "weights": {
        "path": str(weight_path),
        "bytes": weight_bytes,
        "sha256": weight_sha256,
        "expected_sha256": EXPECTED_SHA256,
        "verified": weight_sha256 == EXPECTED_SHA256,
    },
    "single_forward": {
        "forward_calls": forward_calls,
        "batch_size": int(encoded["input_ids"].shape[0]),
        "sequence_length": int(encoded["input_ids"].shape[1]),
        "use_cache": False,
        "logits_to_keep": 1,
        "generate_called": False,
    },
    "input": {
        "prompt": prompt,
        "schema": {
            "type": "object",
            "properties": {
                "danger_level": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "multipleOf": 0.1,
                },
                "run": {"type": "boolean"},
            },
        },
    },
    "value": value,
    "fields": details,
    "prompt_tails": [text[-120:] for text in rendered_prompts],
    "timing_seconds": {
        "download": download_done - started,
        "sha256": hash_done - download_done,
        "model_load": load_done - hash_done,
        "forward": forward_done - forward_started,
        "total": forward_done - started,
    },
    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "cpu_count": os.cpu_count(),
}

Path("qwen-smoke-result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, indent=2))
