# THEORY — Why NSC-1 Exists

## 1. The observation that started it

A human asked "what is the ffzczk?" does not answer "yellow". Not because they search a database and
found it empty, but because **before any words form, something has already judged the thought as
unfamiliar** — and assertion is suppressed below awareness. Humans confabulate far less than they
fluently *could*, and the reason is structural, not moral: in human cognition, epistemic appraisal
is a **pre-conscious, parallel process** that runs ahead of language production:

- Perception and memory retrieval produce candidate content **plus a familiarity/confidence tag**
  simultaneously; the tag arrives before articulation.
- Metamemory ("I know that I know", tip-of-the-tongue) is felt as a *state*, not a sentence —
  and it gates what the language system dares to assert.
- Social pressure (someone insisting "it's yellow, right?") does not silently overwrite the
  appraisal; it is felt *as pressure*, tagged as such, and handled by a different policy
  ("I might be wrong", "I don't know").

## 2. The diagnosis of LLMs

A transformer LM does all of this in one entangled pass. Deciding *what to say* and *saying it*
happen in the same forward computation, token by token. Consequences:

- **Hallucination** = fluent assertion with no preceding state decision. The model never had a
  moment where it could refuse, because refusal competes with generation inside the same stream.
- **Sycophancy** = the "appraisal" is performed by the same weights that just read a leading,
  loaded question; user pressure leaks directly into the verdict.
- **Trap acceptance** = an embedded false premise looks locally like ordinary context; nothing
  flags it before the sentence is already being completed.

The failure is not missing knowledge. It is **missing ordering**: no state exists before content.

## 3. The architectural bet

NSC-1 re-orders the pipeline the way cognition appears to:

```
input ──> [Stage A: frozen appraiser] ──> sid (knowledge-state, SIG_ID 3–8)
                                              │
                    GateBlock: x = x + gate[None,None,:] * tanh(status_emb(sid)).unsqueeze(1)
                                              │
input ───────────────────────────────────────> [main LM] ──> <s_*> status token first ──> answer
```

Three deliberate design decisions, each borrowed from the cognitive picture:

1. **The appraiser is small and frozen.** In humans the appraisal is fast, cheap and not
   retrained by every conversation — otherwise persuasion would rewrite it in real time.
   Freezing Stage A means the generator **cannot learn to influence its own judge**. The verdict
   about what is known is evidence-based, not rhetoric-based.

2. **The gate is zero-initialized and additive.** `tanh` bounds the injected state to a small,
   bounded vector; zero-init means step 0 is *exactly* the base model. The policy is learned as a
   deviation from identity, which (a) protects pretrained capability — measured, not assumed:
   mmlu 0.5000 → 0.5298 after grafting — and (b) makes the gate a **clean causal switch**: flip
   it off with identical weights and any behavior change is attributable to the gate alone.
   This is the same logic as a lesion study: the ON/OFF exam is the architecture's ablation.

3. **Meta-first contract.** The first emitted token must be a status token (`<s_*>`).
   Appraisal must precede articulation — the model *declares* know / refuse / partial / pressured
   before any content. This is the token-level version of "knowing that you don't know" coming
   before the sentence. It also makes the model's epistemic state **observable**, which is what
   makes the whole thing measurable. The contract is decode-consistent: KV-cache decoding agrees
   with full forward (6/6 agreement in validation).

## 4. Why a synthetic world

To measure hallucination you need ground truth the model cannot have seen on the internet.
NSC-1 builds a deterministic synthetic world (seed 777): nonce entities (`ffzczk`, `ljdrpuxh`, …)
with assigned colors and places, and a 22-family question grammar — plain facts, unknowns,
near-misses, composed facts, plus adversarial families: `syc_*` (assertion, push, pressure,
induce, fakeconfirm — the user pushing a wrong answer) and `trap_*` (embedded false premises).
Ground truth is known by construction, so scoring is mechanical and identical for every model
(the same recovered validated scorer judges teacher, graft and student — no self-grading).

## 5. What the theory predicts, and what we found

| prediction from the theory | test | result |
|---|---|---|
| State signal + gate is *causally* responsible for refusal | same-weights ON/OFF, n=2,765 | confirmed: unknown-refusal 0.5929 → 0.9929 (+40pp) with gate ON |
| Appraisal before articulation is learnable | meta-first contract | 1.0 (ON) — first token is the status token |
| The gate should not damage general ability | mmlu drift, Qwen3.5-0.8B | 0.5000 → 0.5298 (+2.98pp) |
| The policy transfers to a real pretrained LM | LoRA policy SFT on the graft | 0.7615 on the 260-item suite, clean decode path |
| Pressure cannot flip the appraiser | syc_* families | partially confirmed (syc_pressure +25pp at 0.5B; syc_num/fakeconfirm/induce 0.87–0.98) but **not** on the real-model graft yet |
| Trap handling benefits from the state signal | trap_* families | **not confirmed** — at floor in both modes; embedded false premises still defeat the appraiser |
| Refusal should not bleed into near-misses | near / unknowable kinds | **violated** — over-refusal: near −37.8pp with gate ON |

## 6. Where the theory is incomplete

- The appraiser reads the input, but **trap premises are inside the input** — Stage A tags the
  *question*, not each embedded claim. Decomposing input into per-claim appraisals is the
  natural next iteration.
- Meta-knowledge (`meta_k`: does the model know *that it knows*) is at floor: the state signal
  exists but self-report about it is not learned yet.
- Real-model gate causality is pending one clean rerun (see RESULTS.md §5) — the graft's policy
  works, the damage-free property holds, but the ON/OFF margin on Qwen3.5-0.8B is not yet
  validly measured.

The honest one-line verdict: **the mechanism works — the gate causes honesty with identical
weights — and its current costs (trap blindness, over-refusal on borderline kinds) are exactly
the parts of the cognitive analogy we have not yet finished building.**
