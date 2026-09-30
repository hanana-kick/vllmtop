from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3.5-0.8B"
REVISION="dc255159ae75b03b99200cd37a2d42ddc42b24d6"
WEIGHT="model.safetensors-00001-of-00001.safetensors"
SHA="04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

cases=[
 {"name":"temperature","prompt":"The measured temperature is exactly 55 degrees Celsius.","field":"temperature","desc":"Measured temperature in degrees Celsius.","lo":-20.0,"hi":80.0,"target":55.0},
 {"name":"score","prompt":"The current score is exactly 37750.","field":"score","desc":"Current score.","lo":1000.0,"hi":50000.0,"target":37750.0},
 {"name":"danger","prompt":"현재 체력은 10이고 적이 달려들고 있다.","field":"danger_level","desc":"현재 상황의 위험도. 0은 안전, 1은 매우 위험.","lo":0.0,"hi":1.0,"target":None},
]
calibrations={
 "balanced_explicit":"This calibration question is intentionally balanced: true and false are equally likely. Return exactly false or true.",
 "no_information":"There is no information that favors either answer. Treat false and true as equally plausible. Return exactly false or true.",
 "token_only":"Ignore meaning. Choose between the two allowed output tokens with no preference. Return exactly false or true.",
 "coin_flip":"A fair coin determines the answer, but its result is hidden. Return exactly false or true.",
}

model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen-calibration-test"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
assert h.hexdigest()==SHA

tok=AutoTokenizer.from_pretrained(model_dir)
tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
false_id=tok.encode("false",add_special_tokens=False)
true_id=tok.encode("true",add_special_tokens=False)
assert len(false_id)==len(true_id)==1
false_id,false_id_list=false_id[0],false_id
true_id=true_id[0]

model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu").eval()

def chat(user, system="You estimate numeric values from context. Output only the requested boolean."):
    return tok.apply_chat_template(
      [{"role":"system","content":system},{"role":"user","content":user}],
      tokenize=False,add_generation_prompt=True,enable_thinking=False
    )+"Answer:"

rows=[]
meta=[]
thresholds=[(i+0.5)/8 for i in range(8)]
for c in cases:
    for u in thresholds:
        t=c["lo"]+u*(c["hi"]-c["lo"])
        rows.append(chat(
          f"Context:\n{c['prompt']}\n\nField: {c['field']}\n"
          f"Field description: {c['desc']}\nAllowed range: [{c['lo']}, {c['hi']}]\n"
          f"Question: Is the best value for this field greater than or equal to {t}?\n"
          "Return exactly false or true."
        ))
        meta.append(("threshold",c["name"],u,t))
for name,text in calibrations.items():
    rows.append(chat(text,system="Output only one of the two requested boolean tokens."))
    meta.append(("calibration",name,None,None))

enc=tok(rows,return_tensors="pt",padding=True,add_special_tokens=False)
with torch.inference_mode():
    t0=time.perf_counter()
    out=model(**enc,use_cache=False,logits_to_keep=1)
    forward=time.perf_counter()-t0
logits=out.logits[:,-1,:].float()

raw={}
for i,m in enumerate(meta):
    lf=float(logits[i,false_id])
    lt=float(logits[i,true_id])
    raw[m]=(lf,lt,lt-lf)

cal_bias={name:raw[("calibration",name,None,None)][2] for name in calibrations}

def sigmoid(x):
    return 1/(1+math.exp(-x))

def isotonic_nonincreasing(values):
    blocks=[]
    for v in values:
        blocks.append([float(v),1])
        while len(blocks)>=2:
            a=blocks[-2][0]/blocks[-2][1]
            b=blocks[-1][0]/blocks[-1][1]
            if a>=b: break
            r=blocks.pop(); l=blocks.pop()
            blocks.append([l[0]+r[0],l[1]+r[1]])
    out=[]
    for total,n in blocks:
        out += [total/n]*n
    return out

results=[]
for c in cases:
    lod=[raw[("threshold",c["name"],u,c["lo"]+u*(c["hi"]-c["lo"]))][2] for u in thresholds]
    methods={"none":0.0,**cal_bias}
    for method,bias in methods.items():
        probs=[sigmoid(x-bias) for x in lod]
        mono=isotonic_nonincreasing(probs)
        uhat=sum(mono)/len(mono)
        value=c["lo"]+uhat*(c["hi"]-c["lo"])
        item={
          "case":c["name"],"method":method,"bias_log_odds":bias,
          "normalized_value":uhat,"value":value,
          "probabilities":probs,"monotonic":mono
        }
        if c["target"] is not None:
            item["absolute_error"]=abs(value-c["target"])
        results.append(item)

payload={
 "model_id":MODEL_ID,"revision":REVISION,"weight_sha256":h.hexdigest(),
 "batch_size":len(rows),"forward_calls":1,"forward_seconds":forward,
 "token_ids":{"false":false_id,"true":true_id},
 "calibration_bias_log_odds":cal_bias,
 "results":results,
}
Path("calibration-result.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
