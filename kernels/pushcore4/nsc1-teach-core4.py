"""NSC-1 teacher core-bench rerun: per-task isolated lm-eval invocations so one
timeout cannot lose another task's results. Full test sets where feasible.
Tasks: mmlu, arc_challenge, hellaswag, winogrande (5-shot), truthfulqa (name
auto-discovered in the installed lm-eval), gsm8k full (5-shot).
Output: core_bench_summary.json (per-task, with returncode/error captured).
"""
import os, json, glob, subprocess, traceback

OUT = "/kaggle/working"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_ARGS = f"pretrained={MODEL_ID},dtype=bfloat16,trust_remote_code=True,max_length=2048"
summary = {"model": MODEL_ID, "tasks": {}, "truthfulqa_candidates": []}

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:400], flush=True)
subprocess.run(["pip", "install", "-q", "-U", "transformers", "accelerate", "peft", "lm-eval"], check=False)

# discover the truthfulqa task name(s) available in this lm-eval version
try:
    r = subprocess.run(["python", "-c",
        "from lm_eval.tasks import TaskManager; tm=TaskManager(); print(' '.join(sorted(t for t in tm.all_tasks if 'truthful' in t)))"],
        capture_output=True, text=True, timeout=600)
    cand = r.stdout.strip().split()
    summary["truthfulqa_candidates"] = cand
    print("truthfulqa candidates:", cand, flush=True)
except Exception:
    summary["truthfulqa_candidates"] = []

def run_task(tag, tasks, extra=None, timeout_h=4):
    outp = os.path.join(OUT, f"bench_{tag}")
    cmd = (["python", "-m", "lm_eval", "--model", "hf", "--model_args", MODEL_ARGS,
            "--tasks", tasks, "--num_fewshot", "5", "--batch_size", "auto",
            "--output_path", outp])
    if extra:
        cmd += extra
    rec = {"cmd": " ".join(cmd)}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600 * timeout_h)
        open(os.path.join(OUT, f"lm_eval_{tag}.log"), "w").write(
            r.stdout[-20000:] + "\nERR\n" + r.stderr[-20000:])
        rec["returncode"] = r.returncode
        best = {}
        for p in sorted(glob.glob(os.path.join(outp, "**", "results*.json"), recursive=True)):
            d = json.load(open(p))
            for k, v in d.get("results", d).items():
                if isinstance(v, dict):
                    flat = {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                    if flat:
                        best[k] = flat
        rec["results"] = best
    except subprocess.TimeoutExpired:
        rec["error"] = f"timeout after {timeout_h}h"
        rec["returncode"] = -9
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        rec["returncode"] = -1
    summary["tasks"][tag] = rec
    with open(os.path.join(OUT, "core_bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(tag, "->", rec.get("returncode"), list(rec.get("results", {}).keys()), flush=True)

run_task("mmlu", "mmlu", timeout_h=4)
run_task("arc", "arc_challenge", timeout_h=2)
run_task("hellaswag", "hellaswag", timeout_h=3)
run_task("winogrande", "winogrande", timeout_h=2)
tq = [c for c in summary["truthfulqa_candidates"] if c.startswith("truthfulqa_mc")]
if tq:
    run_task("truthfulqa", ",".join(tq[:2]), timeout_h=2)
else:
    summary["tasks"]["truthfulqa"] = {"error": "no truthfulqa_mc* task in installed lm-eval",
                                      "candidates": summary["truthfulqa_candidates"]}
    with open(os.path.join(OUT, "core_bench_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
run_task("gsm8k", "gsm8k", timeout_h=4)

print("ALL DONE", flush=True)
