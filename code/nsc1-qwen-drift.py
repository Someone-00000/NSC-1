"""NSC-1 mmlu drift probe: base Qwen3.5-0.8B vs grafted (LoRA adapter_final).
Input: kernel output of potatto007/nsc1-qwen-graft (adapter_final/, gate_final.pt).
Same harness on both sides (identical args), --limit 10 per mmlu subtask
(57 subtasks -> 570 samples per mode); reports the base-vs-graft delta.
Output: drift_results.json, lm_eval logs for both runs.
"""
import os, json, glob, subprocess, traceback

OUT = "/kaggle/working"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
res = {"model": MODEL_ID, "status": "started", "base": {}, "graft": {}, "delta": {}}

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:500], flush=True)
subprocess.run(["pip", "install", "-q", "-U", "transformers", "accelerate", "peft", "lm-eval"], check=False)

def find_dir(marker):
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    return os.path.dirname(hits[0]) if hits else None

adapter_dir = find_dir("adapter_final/adapter_config.json") or find_dir("adapter_config.json")
print("adapter_dir", adapter_dir, flush=True)
res["adapter_dir"] = adapter_dir

COMMON = ["--tasks", "mmlu", "--num_fewshot", "5", "--limit", "10",
          "--batch_size", "auto"]

def run_lm_eval(tag, model_args):
    outp = os.path.join(OUT, f"mmlu_{tag}")
    cmd = (["python", "-m", "lm_eval", "--model", "hf", "--model_args", model_args]
           + COMMON + ["--output_path", outp])
    print(tag, "start", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600 * 4)
    open(os.path.join(OUT, f"lm_eval_{tag}_log.txt"), "w").write(
        r.stdout[-20000:] + "\nERR\n" + r.stderr[-20000:])
    best = {}
    for p in sorted(glob.glob(os.path.join(outp, "**", "results*.json"), recursive=True)):
        d = json.load(open(p))
        rr = d.get("results", d)
        for k, v in rr.items():
            if isinstance(v, dict):
                acc = v.get("acc,none")
                if acc is not None and "aggregate" not in k:
                    best[k] = acc
    return best, r.returncode

try:
    base, rc_b = run_lm_eval("base", f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True,max_length=2048")
    res["base"] = base
    res["base_returncode"] = rc_b
    print("base done", rc_b, flush=True)

    if adapter_dir:
        graft, rc_g = run_lm_eval("graft", f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True,max_length=2048,peft={adapter_dir}")
        res["graft"] = graft
        res["graft_returncode"] = rc_g
        print("graft done", rc_g, flush=True)
        keys = sorted(set(base) | set(graft))
        deltas = {k: round(graft.get(k, 0) - base.get(k, 0), 4) for k in keys if k in base and k in graft}
        res["delta"] = deltas
        if deltas:
            res["mean_delta"] = round(sum(deltas.values()) / len(deltas), 4)
    res["status"] = "ok" if base else "no_base_results"
except Exception:
    res["status"] = "error"
    res["error"] = traceback.format_exc()[-4000:]

with open(os.path.join(OUT, "drift_results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("drift status", res["status"], "mean_delta", res.get("mean_delta"), flush=True)
