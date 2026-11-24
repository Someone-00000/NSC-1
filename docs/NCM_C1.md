# NCM-C1: closed-form compile of a genius head into RAM beside Qwen2.5-0.5B (first compile-speed test)

Full experiment line: KGAT v0-v5 -> C1 -> BR-1 -> BR-2 -> NCM-A1. This doc covers C1.

(C1 doc regenerated in digest form here; full prereg/results JSONs are in results/ with SHA256 blocks inside.)

## What C1 established

- Closed-form compile (ridge/IRLS one-vs-all per horizon, LU-solve, lambda grid {1,10,100}, tau = 60th pct of correct-val max-logit confidence) of H1..H4 heads over 4,848 teacher-forced rows: **81.7 s wall**, 168 KB of genius weights.
- Genius-RAM fire/verify: propose block where consecutive H1 argmax conf >= tau, verify with one batched model forward, accept longest prefix + 1 bonus token. ~90% acceptance.
- Verdict: C1 arm B == arm A token-for-token on 28/30 items; speed ratio ~1.0 (CPU decode is weight-read-bound; 1-2 token blocks cannot beat the verify cost).

See results/wspl_ncm_c1_*.json for the frozen prereg and assembled numbers (self-verifying SHA blocks).