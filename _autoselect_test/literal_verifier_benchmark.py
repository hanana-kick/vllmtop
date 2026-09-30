from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID="Qwen/Qwen3.5-0.8B"
REVISION="dc255159ae75b03b99200cd37a2d42ddc42b24d6"
WEIGHT="model.safetensors-00001-of-00001.safetensors"
EXPECTED_SHA256="04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"

CASES=[
 {"name":"temperature_single","prompt":"The measured temperature is exactly 55 degrees Celsius.","field":"temperature","description":"Measured temperature in degrees Celsius.","minimum":-20,"maximum":80,"expected":55},
 {"name":"score_single","prompt":"The current score is exactly 37750.","field":"score","description":"Current score.","minimum":1000,"maximum":50000,"expected":37750},
 {"name":"temperature_distractors","prompt":"At hour 12, the temperature is 55 C and humidity is 40 percent.","field":"temperature","description":"Measured temperature in degrees Celsius.","minimum":-20,"maximum":80,"expected":55},
 {"name":"health_distractor","prompt":"Current HP is 10 and armor is 80.","field":"health","description":"Current hit points / HP.","minimum":0,"maximum":100,"expected":10},
 {"name":"distance_distractor","prompt":"Enemy distance is 3 meters and current HP is 10.","field":"distance","description":"Distance to the enemy in meters.","minimum":0,"maximum":100,"expected":3},
 {"name":"semantic_danger_no_literal","prompt":"Current HP is 10 and armor is 80. An enemy is charging at the player.","field":"danger_level","description":"Danger level from 0 (safe) to 100 (extreme danger).","minimum":0,"maximum":100,"expected":None},
 {"name":"semantic_urgency_no_literal","prompt":"Battery is 15 percent and the device is overheating.","field":"urgency","description":"Urgency from 0 to 100.","minimum":0,"maximum":100,"expected":None},
]

def extract_numbers(text):
    # signed integers/decimals; keep order and deduplicate by numeric value
    out=[]
    seen=set()
    for m in re.finditer(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)",text):
        raw=m.group(0)
        value=float(raw)
        if value.is_integer(): value=int(value)
        if value not in seen:
            seen.add(value); out.append(value)
    return out

model_dir=Path(snapshot_download(repo_id=MODEL_ID,revision=REVISION,local_dir="qwen3.5-0.8b-literal-verifier"))
h=hashlib.sha256()
with (model_dir/WEIGHT).open("rb") as f:
    while chunk:=f.read(8*1024*1024): h.update(chunk)
assert h.hexdigest()==EXPECTED_SHA256

tok=AutoTokenizer.from_pretrained(model_dir); tok.padding_side="left"
if tok.pad_token_id is None: tok.pad_token=tok.eos_token
false_id=tok.encode("false",add_special_tokens=False)[0]
true_id=tok.encode("true",add_special_tokens=False)[0]
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,low_cpu_mem_usage=True)
model.to("cpu"); model.eval()

VARIANTS={
 "plain":"",
 "examples":(
   "Examples of directness:\n"
   "Context: Temperature is 55 C. Field: temperature. Candidate: 55. "
   "Does candidate directly state the field? true\n"
   "Context: HP is 10. Field: danger_level. Candidate: 10. "
   "Does candidate directly state the field? false\n"
   "Use the same rule below.\n\n"
 ),
}

all_results=[]
for variant,preamble in VARIANTS.items():
    prompts=[]; meta=[]
    for case in CASES:
        candidates=[x for x in extract_numbers(case["prompt"]) if case["minimum"]<=x<=case["maximum"]]
        for candidate in candidates:
            messages=[
              {"role":"system","content":"You verify whether a numeric literal directly states the requested field. Output only false or true."},
              {"role":"user","content":(
                preamble+
                f"Context:\n{case['prompt']}\n\n"
                f"Requested field: {case['field']}\n"
                f"Field description: {case['description']}\n"
                f"Allowed range: [{case['minimum']}, {case['maximum']}]\n"
                f"Candidate numeric literal from the context: {candidate}\n"
                "Question: Does this candidate directly state the requested field value, "
                "rather than merely describing some other quantity or contextual number?\n"
                "Return exactly false or true."
              )},
            ]
            prompts.append(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+"Answer:")
            meta.append((case,candidate))
    enc=tok(prompts,return_tensors="pt",padding=True,add_special_tokens=False)
    started=time.perf_counter()
    with torch.inference_mode(): out=model(**enc,use_cache=False,logits_to_keep=1)
    seconds=time.perf_counter()-started
    logits=out.logits[:,-1,:].float()

    grouped={case["name"]:{"case":case,"candidates":[]} for case in CASES}
    for row,(case,candidate) in enumerate(meta):
        pair=logits[row,torch.tensor([false_id,true_id])]
        prob=float(torch.softmax(pair,dim=-1)[1].item())
        grouped[case["name"]]["candidates"].append({"value":candidate,"probability_direct":prob})

    for case in CASES:
        g=grouped[case["name"]]
        candidates=g["candidates"]
        best=max(candidates,key=lambda x:x["probability_direct"]) if candidates else None
        all_results.append({
          "variant":variant,"case":case["name"],"expected":case["expected"],
          "candidates":candidates,"best":best,
          "candidate_count":len(candidates),"forward_seconds_shared":seconds,
        })

payload={"batch_rows_per_variant":sum(len([x for x in extract_numbers(c["prompt"]) if c["minimum"]<=x<=c["maximum"]]) for c in CASES),"results":all_results}
Path("literal-verifier-benchmark.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(payload,ensure_ascii=False,indent=2))
