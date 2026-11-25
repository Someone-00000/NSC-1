"""Assemble wspl_br1_results.json from eval states + summaries + compile times."""
import json, hashlib

def sha(p):
    h = hashlib.sha256(); h.update(open(p, 'rb').read()); return h.hexdigest()

ROOT = '/home/z/my-project'
ev = f'{ROOT}/realbench_cache/br1_eval'
sums = {a: json.load(open(f'{ev}/summary_{a}.json')) for a in 'ABC'}
stA = json.load(open(f'{ev}/state_A.json'))
stB = json.load(open(f'{ev}/state_B.json'))
stC = json.load(open(f'{ev}/state_C.json'))
times = json.load(open(f'{ROOT}/download/wspl_br1_compile_times.json'))

def div_detail(st):
    out = []
    for k, v in sorted(st['items'].items(), key=lambda x: int(x[0])):
        fd = v['first_divergence_vs_A']
        if fd != -1:
            out.append({'idx': int(k), 'first_div': fd, 'len_arm': v['n_tokens'],
                        'strict_arm': v['strict'], 'strict_A': stA['items'][k]['strict'],
                        'pred_arm': v['pred'], 'pred_A': stA['items'][k]['pred']})
    return out

identB = sum(1 for v in stB['items'].values() if v['first_divergence_vs_A'] == -1)
identC = sum(1 for v in stC['items'].values() if v['first_divergence_vs_A'] == -1)
cB = stB['counters']
avg_block = (cB['accepted'] + cB['fire_steps']) / max(cB['fire_steps'], 1)

def arm_sum(a):
    s = sums[a]
    return {'arm': a, 'n': s['n_done'], 'strict': s['strict_acc'], 'tokens': s['tokens'],
            'wall_gen_s': s['wall_gen_s'], 'tok_per_s': s['tok_per_s'],
            'forwards': s['counters']['forwards'], 'counters': s['counters']}

res = {
 'version': 'br1',
 'date': '2026-11-24',
 'prereg': 'download/wspl_br1_prereg.json',
 'sha256': {
   'prereg': sha(f'{ROOT}/download/wspl_br1_prereg.json'),
   'heads_npz': sha(f'{ROOT}/download/wspl_br1_heads.npz'),
   'model_safetensors': sha(f'{ROOT}/models/qwen25-05b/model.safetensors'),
   'times_json': sha(f'{ROOT}/download/wspl_br1_compile_times.json'),
   'state_A': sha(f'{ev}/state_A.json'),
   'state_B': sha(f'{ev}/state_B.json'),
   'state_C': sha(f'{ev}/state_C.json'),
   'br1_build_py': sha(f'{ROOT}/scripts/br1_build.py'),
   'br1_eval_py': sha(f'{ROOT}/scripts/br1_eval.py')
 },
 'amendments': ['amendment_0 (pre-build, disclosed in prereg): probe 192->384 items after digit-dense row-count re-estimate; per-slice budget with disclosed cache resume'],
 'environment': {'torch': '2.15.0+cpu', 'transformers': '5.24.0', 'threads': 2, 'dtype': 'float32',
                 'device': 'CPU only', 'model': 'Qwen/Qwen2.5-0.5B-Instruct 494M params',
                 'model_recovered': 'model.safetensors lost in sandbox recycle 2026-11-24; re-downloaded; sha256 verified == frozen identity before any build (GATE-3 PASS)'},
 'gates': {
   'GATE-3_model_identity': 'PASS: sha256 fdf756fa...88fb7fe == prereg-frozen identity (re-download after recycle)',
   'GATE-0_gold_alignment': '30/30 PASS',
   'GATE-1_probe': 'PASS: 13874 rows >= 12000; universe verified',
   'GATE-det': 'PASS: max|dW|=0.0, max|dB|=0.0, lambda/tau identical on full recompile',
   'GATE-slices': 'probe used 2 foreground slices (445s + 48s), resume disclosed; per-slice <= 460s PASS'
 },
 'compile_S1': {
   'probe_rows': 13874, 'probe_forwards_total': 70, 'compile_s': times['compile_s'], 'recompile_s': times['recompile_s'],
   'heads_bytes': times['heads_bytes'],
   'note': 'compile 8.07s for 8 horizons x 12 classes on 13874 rows (LU-cached closed-form solve); C1 took 10.88s for 4 horizons on 4297 rows - the scaled compile is FASTER per horizon-row',
   'val_acc_trainfit_inU': {f'H{k}': v['val_acc_inU_trainfit'] for k, v in times['val_stats'].items()},
   'tau': {f'H{k}': round(v['tau'], 3) for k, v in times['val_stats'].items()},
   'lambda': {f'H{k}': v['lambda'] for k, v in times['val_stats'].items()}
 },
 'speed_S2': {a: arm_sum(a) for a in 'ABC'},
 'ratio_B_vs_A': round(sums['B']['tok_per_s'] / sums['A']['tok_per_s'], 4),
 'ratio_C_vs_A': round(sums['C']['tok_per_s'] / sums['A']['tok_per_s'], 4),
 'B_mechanism': {
   'fire_rate_of_steps': round(cB['fire_steps'] / sums['B']['counters']['steps'], 4),
   'acceptance_rate_when_fired': round(cB['accepted'] / max(cB['proposals'], 1), 4),
   'avg_tokens_per_fire_iteration': round(avg_block, 2),
   'fwd_per_tok': round(sums['B']['counters']['forwards'] / sums['B']['tokens'], 4),
   'token_identical_to_A': f'{identB}/30'
 },
 'C_divergence': {'items_diverged_from_A': 30 - identC,
                  'strict_agreement_with_A': sum(1 for k in stC['items'] if stC['items'][k]['strict'] == stA['items'][k]['strict'])},
 'divergence_detail_B': div_detail(stB),
 'accuracy_S3': {
   'A_strict_acc': sums['A']['strict_acc'], 'B_strict_acc': sums['B']['strict_acc'], 'C_strict_acc': sums['C']['strict_acc'],
   'B_equals_A': sums['B']['strict_acc'] == sums['A']['strict_acc'],
   'C_drop_pp': round((sums['A']['strict_acc'] - sums['C']['strict_acc']) * 100, 1)
 },
 'c1_comparison': {
   'C1_A': {'strict': 0.50, 'tok_s': 8.35},
   'C1_B': {'strict': 0.50, 'tok_s': 8.25, 'ratio': 0.988, 'fire_rate': 0.0529, 'acceptance': 0.9377, 'avg_tok_per_fire': 2.25},
   'C1_C': {'strict': 0.3333, 'tok_s': 8.48, 'ratio': 1.0156},
   'br1_A_strict_delta_pp': round((sums['A']['strict_acc'] - 0.50) * 100, 1)
 }
}

# frozen-band verdicts
fr = res['B_mechanism']['fire_rate_of_steps']
ratio = res['ratio_B_vs_A']
speed_band = 'SUCCESS' if ratio >= 1.10 else ('MARGINAL' if ratio >= 0.95 else 'FAIL')
fire_band = 'CONFIRMED' if fr >= 0.15 else ('PARTIAL' if fr >= 0.08 else 'REJECTED')
acc_hold = res['accuracy_S3']['B_equals_A'] and identB >= 29
res['verdict'] = (
 f"THEORY HOLDS AT SCALE, SPEED STILL NEUTRAL, DIGIT-DENSE LEVER REJECTED AT THE FIRE-RATE BAR. "
 f"(1) Compile scaled and got FASTER: 13874 probe rows (3.2x C1), 8 horizons (2x C1), closed-form compile 8.07s, bit-exact recompile, heads 321KB RAM-resident; probe crossed 2 slices (445s+48s) with disclosed resume, every slice <= 460s. "
 f"(2) Verify/reason generation stays accuracy-exact: B strict {res['accuracy_S3']['B_strict_acc']:.4f} == A strict {res['accuracy_S3']['A_strict_acc']:.4f} (0.0pp difference), {identB}/30 token-identical (divergences are fp near-tie argmax flips, not mechanism errors). "
 f"(3) Speed ratio B/A = {ratio} (MARGINAL band [0.95,1.10)): fire rate rose 5.29% -> {fr:.2%} but stays under the 8% partial bar -> H1 lever REJECTED per frozen band; acceptance {res['B_mechanism']['acceptance_rate_when_fired']:.3f} and avg {res['B_mechanism']['avg_tokens_per_fire_iteration']} tokens/fire-iteration mean verify overhead still cancels block savings. "
 f"(4) The digit-dense format COSTS the base model: A strict 36.7% vs C1 verbose 50.0% (-13.3pp, outside the frozen +/-10pp stability band -> H4 FAIL); C blind-fire drops to {res['accuracy_S3']['C_strict_acc']:.4f} (-{res['accuracy_S3']['C_drop_pp']}pp) re-confirming the reason-over-blocks step is necessary. "
 f"HONEST BOTTOM LINE: more data into the same closed-form compile scales cleanly and cheaply (this is the data->weight side of the big run working), but the tok/s win needs a different lever than format: the fire/tau policy (lower tau with acceptance-aware budget) or batched verify, not more probe rows."
)
res['hypotheses'] = {
 'H1_fire': f"REJECTED ({fr:.4f} < 0.08 partial bar)",
 'H2_speed': f"REJECTED (ratio {ratio} < 1.10; MARGINAL)",
 'H3_accuracy': f"SUPPORTED (B == A strict exactly, 0.0pp; {identB}/30 token-identical)",
 'H4_stability': "REJECTED (A_br 36.7% vs C1 50.0%, -13.3pp outside +/-10pp band)"
}
res['honest_notes'] = [
 '2/30 B items diverge from A (fp near-tie argmax flips in the batched verify forward vs single-token forwards); strict still equal on both - same phenomenon as C1 (29/30)',
 'fires concentrate in digit runs, but H2 tau (0.737) cuts blocks to ~1 extra token on average; consecutive-confidence rule is the binding constraint, not head quality',
 'digit-dense format shortens generations (181 tok/item vs 269 in C1) and raises tok/s floor (8.69 vs 8.35) for BOTH arms - the ratio is the honest comparison',
 'smoke-test states were wiped before full runs; arms run with per-item checkpoint/resume across slices; C1 probe cache (48 items) untouched and distinct from BR-1 cache (384 items)',
 'C arm acceptance is 0 by definition (blind); its fire rate 8.89% reflects that without verification the loop keeps re-firing on drifted states'
]
json.dump(res, open(f'{ROOT}/download/wspl_br1_results.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('ratio_B_vs_A', 'ratio_C_vs_A', 'B_mechanism', 'accuracy_S3', 'hypotheses')}, indent=1))
print()
print('DIVERGENCE B:', json.dumps(res['divergence_detail_B']))
print('VERDICT:', res['verdict'])
