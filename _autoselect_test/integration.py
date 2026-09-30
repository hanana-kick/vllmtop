from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path("_autoselect_test/src").resolve()))

from autoselect.selector import QwenSingleForwardSelector

MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "dc255159ae75b03b99200cd37a2d42ddc42b24d6"
WEIGHT = "model.safetensors-00001-of-00001.safetensors"
EXPECTED_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

model_dir = Path(snapshot_download(
    repo_id=MODEL_ID,
    revision=REVISION,
    local_dir="qwen3.5-0.8b-hybrid-test",
))

h = hashlib.sha256()
with (model_dir / WEIGHT).open("rb") as f:
    while chunk := f.read(8 * 1024 * 1024):
        h.update(chunk)
sha256 = h.hexdigest()
assert sha256 == EXPECTED_SHA256

selector = QwenSingleForwardSelector(
    model_id=str(model_dir),
    dtype="bfloat16",
    numeric_thresholds=4,
    literal_margin=0.5,
)

cases = [
    {
        "name": "temperature_with_distractor",
        "prompt": "The measured temperature is exactly 55 degrees Celsius. The humidity index is 40.",
        "schema": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "minimum": -20,
                    "maximum": 80,
                    "description": "Measured temperature in degrees Celsius.",
                }
            },
        },
        "expected": 55.0,
    },
    {
        "name": "score_with_previous_value",
        "prompt": "The previous score was 30000. The current score is exactly 37750.",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 50000,
                    "description": "Current score.",
                }
            },
        },
        "expected": 37750,
    },
    {
        "name": "danger_and_boolean",
        "prompt": "현재 체력은 10이고 적이 달려들고 있다.",
        "schema": {
            "type": "object",
            "properties": {
                "danger_level": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
                },
                "run": {
                    "type": "boolean",
                    "description": "즉시 도망쳐야 하는지 여부.",
                },
            },
        },
        "expected": None,
    },
]

results = []
for case in cases:
    started = time.perf_counter()
    result = selector.select(case["prompt"], case["schema"])
    elapsed = time.perf_counter() - started
    assert result.forward_calls == 1
    entry = {
        "name": case["name"],
        "value": result.value,
        "forward_calls": result.forward_calls,
        "batch_size": result.batch_size,
        "seconds": elapsed,
        "fields": [
            {
                "path": field.path,
                "value": field.value,
                "method": field.method,
                "confidence": field.confidence,
                "details": field.details,
            }
            for field in result.fields
        ],
    }
    if case["expected"] is not None:
        key = next(iter(result.value))
        entry["absolute_error"] = abs(float(result.value[key]) - float(case["expected"]))
    results.append(entry)

assert results[0]["value"]["temperature"] == 55.0
assert results[0]["fields"][0]["method"] == "literal-sequence"
assert results[1]["value"]["score"] == 37750
assert results[1]["fields"][0]["method"] == "literal-sequence"
assert results[2]["fields"][0]["method"] == "label-swap-thresholds"
assert results[2]["value"]["run"] is True

payload = {
    "model_id": MODEL_ID,
    "revision": REVISION,
    "weights_sha256": sha256,
    "cases": results,
}
Path("hybrid-result.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
