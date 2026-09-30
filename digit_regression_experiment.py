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

cases = [
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

model_dir = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir="qwen-digit-test"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024):
        h.update(chunk)
assert h.hexdigest()==EXPECTED_SHA256

tok=AutoTokenizer.from_pretrained(model_dir)
tok.padding_side="left"
if tok.pad_token_id is None:
    tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu").eval()

digits=tuple(str(i) for i in range(10))
digit_ids=[]
for d in digits:
    ids=tok.encode(d,add_special_tokens=False)
    if len(ids)!=1:
        raise RuntimeError((d,ids))
    digit_ids.append(ids[0])

variants = {
    "ordinal": (
        "Infer the scalar value of the field from the context. "
        "Then normalize it within the allowed range. "
        "Output exactly one digit from 0 to 9, where 0 means the minimum end "
        "of the range and 9 means the maximum end. Intermediate digits are "
        "evenly spaced positions. Output only the digit."
    ),
    "decile": (
        "Choose which decile of the allowed numeric range best contains the field value. "
        "Return exactly one digit 0-9. 0 is the lowest decile, 9 is the highest decile. "
        "Do not output words or explanations."
    ),
    "percent": (
        "Estimate the field as a percentage of the way from the minimum to the maximum. "
        "Encode that percentage as one digit 0-9: 0=0%, 1≈11%, 2≈22%, 3≈33%, "
        "4≈44%, 5≈56%, 6≈67%, 7≈78%, 8≈89%, 9=100%. Output only one digit."
    ),
}

rows=[]
meta=[]
for variant_name, instruction in variants.items():
    for case in cases:
        messages=[
            {"role":"system","content":"You are a precise numeric estimator. Return only the requested digit."},
            {"role":"user","content":(
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}]\n\n"
                f"{instruction}"
            )},
        ]
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        rows.append(rendered)
        meta.append((variant_name,case))

enc=tok(rows,return_tensors="pt",padding=True,add_special_tokens=False)
with torch.inference_mode():
    t0=time.perf_counter()
    out=model(**enc,use_cache=False,logits_to_keep=1)
    forward=time.perf_counter()-t0

last=out.logits[:,-1,:].float()
ids=torch.tensor(digit_ids,dtype=torch.long)
results=[]
for row,(variant_name,case) in enumerate(meta):
    logits=last[row,ids]
    probs=torch.softmax(logits,dim=-1)
    argmax=int(torch.argmax(probs))
    expected_digit=sum(i*float(probs[i]) for i in range(10))
    u_argmax=argmax/9
    u_expected=expected_digit/9
    lo,hi=case["minimum"],case["maximum"]
    value_argmax=lo+u_argmax*(hi-lo)
    value_expected=lo+u_expected*(hi-lo)
    item={
        "variant":variant_name,
        "case":case["name"],
        "argmax_digit":argmax,
        "expected_digit":expected_digit,
        "value_argmax":value_argmax,
        "value_expected":value_expected,
        "probabilities":{str(i):float(probs[i]) for i in range(10)},
    }
    if case["target"] is not None:
        item["argmax_abs_error"]=abs(value_argmax-case["target"])
        item["expected_abs_error"]=abs(value_expected-case["target"])
    results.append(item)

payload={
    "model_id":MODEL_ID,
    "revision":REVISION,
    "weight_sha256":h.hexdigest(),
    "batch_size":len(rows),
    "forward_calls":1,
    "forward_seconds":forward,
    "digit_token_ids":dict(zip(digits,digit_ids)),
    "results":results,
}
Path("digit-regression-result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
