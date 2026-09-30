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

CASES = [
    {
        "name": "temperature",
        "prompt": "The measured temperature is exactly 55 degrees Celsius.",
        "field": "temperature",
        "description": "Measured temperature in degrees Celsius.",
        "minimum": -20.0,
        "maximum": 80.0,
        "thresholds": [-10, 0, 10, 20, 30, 40, 50, 60, 70],
        "target": 55.0,
    },
    {
        "name": "score",
        "prompt": "The current score is exactly 37750.",
        "field": "score",
        "description": "Current score.",
        "minimum": 1000.0,
        "maximum": 50000.0,
        "thresholds": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000],
        "target": 37750.0,
    },
    {
        "name": "danger",
        "prompt": "현재 체력은 10이고 적이 달려들고 있어.",
        "field": "danger_level",
        "description": "현재 상황의 위험도. 0은 안전, 1은 매우 위험.",
        "minimum": 0.0,
        "maximum": 1.0,
        "thresholds": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
        "target": None,
    },
]

def isotonic_nonincreasing(values):
    blocks=[]
    for value in values:
        blocks.append([float(value),1])
        while len(blocks)>=2:
            a=blocks[-2][0]/blocks[-2][1]
            b=blocks[-1][0]/blocks[-1][1]
            if a>=b: break
            r=blocks.pop(); l=blocks.pop()
            blocks.append([l[0]+r[0],l[1]+r[1]])
    out=[]
    for total,count in blocks:
        out.extend([total/count]*count)
    return out

def trapz_survival(positions, probs):
    xs=[0.0]+positions+[1.0]
    ys=[1.0]+probs+[0.0]
    area=0.0
    for i in range(len(xs)-1):
        area += (xs[i+1]-xs[i])*(ys[i]+ys[i+1])/2
    return area

model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen3.5-0.8b-nice-threshold"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
assert h.hexdigest()==EXPECTED_SHA256

tokenizer=AutoTokenizer.from_pretrained(model_dir)
tokenizer.padding_side="left"
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
false_id=tokenizer.encode("false",add_special_tokens=False)[0]
true_id=tokenizer.encode("true",add_special_tokens=False)[0]

model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu"); model.eval()

results=[]
for case in CASES:
    prompts=[]
    for threshold in case["thresholds"]:
        messages=[
            {"role":"system","content":"You estimate numeric values from context. Output only the requested boolean."},
            {"role":"user","content":(
                f"Context:\n{case['prompt']}\n\n"
                f"Field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
                "Return exactly false or true."
            )},
        ]
        prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+"Answer:")
    encoded=tokenizer(prompts,return_tensors="pt",padding=True,add_special_tokens=False)
    started=time.perf_counter()
    with torch.inference_mode():
        output=model(**encoded,use_cache=False,logits_to_keep=1)
    seconds=time.perf_counter()-started
    logits=output.logits[:,-1,:].float()
    probs=[]
    for row in range(len(prompts)):
        pair=logits[row,torch.tensor([false_id,true_id])]
        probs.append(float(torch.softmax(pair,dim=-1)[1].item()))
    mono=isotonic_nonincreasing(probs)
    span=case["maximum"]-case["minimum"]
    positions=[(t-case["minimum"])/span for t in case["thresholds"]]
    normalized=trapz_survival(positions,mono)
    value=case["minimum"]+normalized*span
    entry={
        "case":case["name"],"thresholds":case["thresholds"],"positions":positions,
        "probabilities":probs,"probabilities_monotonic":mono,
        "normalized":normalized,"value":value,"batch_size":len(prompts),
        "forward_calls":1,"forward_seconds":seconds,
    }
    if case["target"] is not None:
        entry["target"]=case["target"]; entry["absolute_error"]=abs(value-case["target"])
    results.append(entry)

payload={"results":results}
Path("nice-threshold-benchmark.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
