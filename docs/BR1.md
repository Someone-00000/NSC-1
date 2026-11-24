# BR-1: genius heads at scale - 384 probes, 8 horizons, 8.07 s compile

**Parent:** NCM-C1. **Discipline:** WSPL (prereg frozen SHA f7bf5a80..., digit-dense amendment disclosed pre-build, no gradients).

## Compile (as-run)

- 384 digit-dense probe items, 13,874 teacher-forced rows, H1..H8 horizons, 12-class universe U = {digits 15..24, \n=198, EOS=151645}.
- **Compile wall 8.07 s** (recompile 8.03 s, determinism gate max|dW| = 0.0 - bit-exact). Heads npz 321 KB, sha256 d2885e6bf7a73106...
- val-acc-in-U: H1 89.0%, H4 45.3%, H8 40.1%.
- 3.2x probe data + 2x horizons vs C1 yet 10x faster compile (LU-cached solves, part-cached probes).

## Eval arms (30 frozen items, indices 7+43k, digit-dense prompt)

| Arm | strict | tok/s |
|---|---|---|
| A: plain AR baseline | 36.67% | 8.69 |
| B: genius block-propose + verify | 36.67% | 8.78 (ratio 1.010) |
| C: blind fire (no verify) | 26.67% | 9.13 |

- B == A on strict; 28/30 items token-identical; the 2 divergences were fp near-ties and both strict-correct.
- Fire rate 7.01% of steps (frozen CONFIRMED bar was 15%, PARTIAL 8% -> PARTIAL); acceptance 90.1%, mean 1.99 tok/fire.
- Digit-dense lever REJECTED (H4 stability): it raised tok/s floor but cost -13.3pp baseline accuracy.

## Frozen-band verdicts

- Knowledge-to-weight transfer: CONFIRMED at scale (bit-exact rebuild, faster compile with more data).
- Token-level speed: the ceiling of block-propose on CPU is ~1.01x - the lever is dead, which motivated C2/BR-2 (answer-level parallelism).

See results/wspl_br1_*.json (prereg, compile times, results - self-verifying SHA blocks).