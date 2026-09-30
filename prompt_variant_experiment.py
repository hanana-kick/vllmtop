from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3.5-0.8B"
REVISION="dc255159ae75b03b99200cd37a2d42ddc42b24d6"
SHA="04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
WEIGHT="model.safetensors-00001-of-00001.safetensors"
cases=[
 {"name":"temperature","prompt":"The measured temperature is exactly 55 degrees Celsius.","field":"temperature","desc":"Measured temperature in degrees Celsius.","lo":-20.0,"hi":80.0,"target":55.0},
 {"name":"score","prompt":"The current score is exactly 37750.","field":"score","desc":"Current score.","lo":1000.0,"hi":50000.0,"target":37750.0},
]
variants={
 "baseline":{
   "system":"You estimate numeric values from context. Output only the requested boolean.",
   "body":lambda c,t:(
      f"Context:\n{c['prompt']}\n\nField: {c['field']}\nField description: {c['desc']}\n"
      f"Allowed range: [{c['lo']}, {c['hi']}]\n"
      f"Question: Is the best value for this field greater than or equal to {t}?\n"
      "Return exactly false or true."
   )
 },
 "exact_priority":{
   "system":"You compare a numeric field to a threshold. If the context states an exact numeric value for the field, use that value exactly. Only infer a value when no exact value is stated. Output only false or true.",
   "body":lambda c,t:(
      f"Context:\n{c['prompt']}\n\nField: {c['field']}\nField description: {c['desc']}\n"
      f"Allowed range: [{c['lo']}, {c['hi']}]\nThreshold: {t}\n"
      "Silently resolve one numeric value for the field. If an exact value is stated, do not estimate or reinterpret it. "
      "Then answer whether value >= threshold. Return exactly false or true."
   )
 },
 "terse_math":{
   "system":"Numeric comparison classifier. Use literal numeric facts exactly when available. Output only false or true.",
   "body":lambda c,t:(
      f"Context: {c['prompt']}\nField: {c['field']} ({c['desc']})\nRange: [{c['lo']}, {c['hi']}]\n"
      f"Evaluate: {c['field']} >= {t}\nReturn false or true only."
   )
 },
}
thresholds=[(i+0.5)/4 for i in range(4)]

model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen-prompt-variant-test"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
assert h.hexdigest()==SHA

tok=AutoTokenizer.from_pretrained(model_dir); tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
fid=tok.encode("false",add_special_tokens=False); tid=tok.encode("true",add_special_tokens=False)
assert len(fid)==len(tid)==1
fid,tid=fid[0],tid[0]
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu").eval()

def render(system,user):
    return tok.apply_chat_template(
      [{"role":"system","content":system},{"role":"user","content":user}],
      tokenize=False,add_generation_prompt=True,enable_thinking=False
    )+"Answer:"

rows=[]; meta=[]
for vname,v in variants.items():
    for c in cases:
        for u in thresholds:
            t=c["lo"]+u*(c["hi"]-c["lo"])
            rows.append(render(v["system"],v["body"](c,t)))
            meta.append(("threshold",vname,c["name"],u,t))
    rows.append(render(
      v["system"],
      "There is no information that favors either boolean answer. Treat false and true as equally plausible. Return exactly false or true."
    ))
    meta.append(("calibration",vname,None,None,None))

enc=tok(rows,return_tensors="pt",padding=True,add_special_tokens=False)
am=enc["attention_mask"]
pos=am.long().cumsum(-1)-1
pos.masked_fill_(am==0,0)
with torch.inference_mode():
    t0=time.perf_counter()
    out=model(**enc,position_ids=pos,use_cache=False,logits_to_keep=1)
    elapsed=time.perf_counter()-t0
last=out.logits[:,-1,:].float()
lod=[float(last[i,tid]-last[i,fid]) for i in range(len(rows))]

cal={}
for i,m in enumerate(meta):
    if m[0]=="calibration": cal[m[1]]=lod[i]

def sig(x): return 1/(1+math.exp(-x))
def iso(vals):
    blocks=[]
    for v in vals:
        blocks.append([v,1])
        while len(blocks)>=2 and blocks[-2][0]/blocks[-2][1] < blocks[-1][0]/blocks[-1][1]:
            b=blocks.pop(); a=blocks.pop(); blocks.append([a[0]+b[0],a[1]+b[1]])
    out=[]
    for s,n in blocks: out += [s/n]*n
    return out

results=[]
for vname in variants:
    for c in cases:
        vals=[]
        raw=[]
        for i,m in enumerate(meta):
            if m[0]=="threshold" and m[1]==vname and m[2]==c["name"]:
                raw.append(lod[i]); vals.append(sig(lod[i]-cal[vname]))
        mono=iso(vals)
        uhat=sum(mono)/len(mono)
        value=c["lo"]+uhat*(c["hi"]-c["lo"])
        results.append({
          "variant":vname,"case":c["name"],"calibration_log_odds":cal[vname],
          "probabilities":vals,"monotonic":mono,"normalized":uhat,"value":value,
          "absolute_error":abs(value-c["target"]),
          "normalized_absolute_error":abs(value-c["target"])/(c["hi"]-c["lo"])
        })

payload={"batch_size":len(rows),"forward_calls":1,"forward_seconds":elapsed,"results":results}
Path("prompt-variant-result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
print(json.dumps(payload,indent=2))
