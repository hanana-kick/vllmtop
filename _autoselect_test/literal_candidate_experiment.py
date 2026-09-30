from __future__ import annotations

import hashlib
import json
import re
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
        "name": "temperature_with_distractor",
        "prompt": "The measured temperature is exactly 55 degrees Celsius. The humidity index is 40.",
        "field": "temperature",
        "description": "Measured temperature in degrees Celsius.",
        "minimum": -20.0,
        "maximum": 80.0,
        "integer": False,
        "expected": 55.0,
    },
    {
        "name": "score_with_previous_value",
        "prompt": "The previous score was 30000. The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "integer": True,
        "expected": 37750.0,
    },
]


def extract_numbers(text):
    return [float(x) for x in re.findall(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)", text)]


def format_number(value, integer):
    if integer:
        return str(int(round(value)))
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return format(value, ".12g")


def prompt_text(tokenizer, case):
    messages = [
        {"role": "system", "content": "Return only the requested numeric value. Do not explain."},
        {"role": "user", "content": (
            f"Context:\n{case['prompt']}\n\n"
            f"Field: {case['field']}\nField description: {case['description']}\n"
            f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
            "Return the value of this field. If explicitly stated, use that exact value."
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "Answer:"


def lcp(a,b):
    i=0
    while i<min(len(a),len(b)) and a[i]==b[i]:
        i+=1
    return i


def score(model, tokenizer, prompt, candidates, integer):
    base=tokenizer.encode(prompt, add_special_tokens=False)
    rows=[]; lcps=[]; lens=[]; texts=[]
    for value in candidates:
        text=format_number(value, integer)
        ids=tokenizer.encode(prompt+text, add_special_tokens=False)
        p=lcp(base,ids)
        rows.append(ids); lcps.append(p); lens.append(len(ids)-p); texts.append(text)
    max_len=max(map(len,rows)); max_suffix=max(lens); keep=max_suffix+1
    inp=[]; mask=[]
    for ids in rows:
        pad=max_len-len(ids)
        inp.append([tokenizer.pad_token_id]*pad+ids)
        mask.append([0]*pad+[1]*len(ids))
    inp=torch.tensor(inp); mask=torch.tensor(mask)
    t=time.perf_counter()
    with torch.inference_mode():
        out=model(input_ids=inp,attention_mask=mask,use_cache=False,logits_to_keep=keep)
    seconds=time.perf_counter()-t
    lp=torch.log_softmax(out.logits.float(),dim=-1)
    scored=[]
    for r,ids in enumerate(rows):
        vals=[]
        for j in range(lens[r]):
            rel=keep-lens[r]+j-1
            vals.append(float(lp[r,rel,ids[lcps[r]+j]].item()))
        scored.append({"candidate":candidates[r],"text":texts[r],"mean_logprob":sum(vals)/len(vals),"sum_logprob":sum(vals),"token_logprobs":vals})
    return scored,seconds


model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen3.5-0.8b-literal-candidates"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
assert h.hexdigest()==EXPECTED_SHA256
tokenizer=AutoTokenizer.from_pretrained(model_dir); tokenizer.padding_side="left"
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu").eval()

results=[]
for case in CASES:
    literals=[x for x in extract_numbers(case["prompt"]) if case["minimum"] <= x <= case["maximum"]]
    anchors=[case["minimum"], (case["minimum"]+case["maximum"])/2, case["maximum"]]
    candidates=[]
    for x in literals+anchors:
        v=int(round(x)) if case["integer"] else x
        if v not in candidates: candidates.append(v)
    scores,seconds=score(model,tokenizer,prompt_text(tokenizer,case),candidates,case["integer"])
    ranked=sorted(scores,key=lambda x:x["mean_logprob"],reverse=True)
    results.append({
        "name":case["name"],
        "expected":case["expected"],
        "extracted_literals":literals,
        "candidates":candidates,
        "selected":ranked[0]["candidate"],
        "margin_to_second":ranked[0]["mean_logprob"]-ranked[1]["mean_logprob"],
        "forward_calls":1,
        "batch_size":len(candidates),
        "forward_seconds":seconds,
        "ranking":ranked,
    })

payload={"model_id":MODEL_ID,"revision":REVISION,"weights_sha256":h.hexdigest(),"results":results}
Path("literal-candidate-result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
