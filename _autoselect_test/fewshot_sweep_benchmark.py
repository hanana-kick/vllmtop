from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3.5-0.8B"
REVISION="dc255159ae75b03b99200cd37a2d42ddc42b24d6"
WEIGHT="model.safetensors-00001-of-00001.safetensors"
EXPECTED_SHA256="04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
THRESHOLDS=[(i+0.5)/8 for i in range(8)]
TARGETS=[10,25,50,75,90]

def isotonic_nonincreasing(values):
    blocks=[]
    for value in values:
        blocks.append([float(value),1])
        while len(blocks)>=2:
            a=blocks[-2][0]/blocks[-2][1]; b=blocks[-1][0]/blocks[-1][1]
            if a>=b: break
            r=blocks.pop(); l=blocks.pop(); blocks.append([l[0]+r[0],l[1]+r[1]])
    out=[]
    for total,count in blocks: out.extend([total/count]*count)
    return out

model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen3.5-0.8b-fewshot-sweep"))
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

variants={
 "baseline":"",
 "false_then_true":(
   "Comparison examples:\n"
   "Example 1: known value = 25, threshold = 50. Is value >= threshold? false\n"
   "Example 2: known value = 75, threshold = 50. Is value >= threshold? true\n"
   "Follow exactly the same comparison rule for the actual field below.\n\n"
 ),
}

results=[]
for variant,preamble in variants.items():
  for target in TARGETS:
    prompts=[]
    for u in THRESHOLDS:
      threshold=u*100
      messages=[
       {"role":"system","content":"You estimate numeric values from context and answer comparison questions. Output only false or true."},
       {"role":"user","content":(
         preamble+
         f"Context:\nThe measured value is exactly {target}.\n\n"
         "Field: value\nField description: measured numeric value.\n"
         "Allowed range: [0, 100]\n"
         f"Question: Is the best value for this field greater than or equal to {threshold}?\n"
         "Return exactly false or true."
       )},
      ]
      prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+"Answer:")
    enc=tokenizer(prompts,return_tensors="pt",padding=True,add_special_tokens=False)
    started=time.perf_counter()
    with torch.inference_mode(): out=model(**enc,use_cache=False,logits_to_keep=1)
    seconds=time.perf_counter()-started
    logits=out.logits[:,-1,:].float()
    probs=[]
    for row in range(8):
      pair=logits[row,torch.tensor([false_id,true_id])]
      probs.append(float(torch.softmax(pair,dim=-1)[1].item()))
    mono=isotonic_nonincreasing(probs)
    estimate=100*sum(mono)/8
    results.append({"variant":variant,"target":target,"estimate":estimate,"absolute_error":abs(estimate-target),"probabilities":probs,"monotonic":mono,"forward_seconds":seconds})

summary={}
for variant in variants:
  rows=[r for r in results if r["variant"]==variant]
  summary[variant]={
    "mae":sum(r["absolute_error"] for r in rows)/len(rows),
    "max_error":max(r["absolute_error"] for r in rows),
    "estimates":[{"target":r["target"],"estimate":r["estimate"],"error":r["absolute_error"]} for r in rows],
  }

payload={"summary":summary,"results":results}
Path("fewshot-sweep-benchmark.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
