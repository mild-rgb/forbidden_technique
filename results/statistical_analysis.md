# Option-B 2×2 — statistical analysis

**Question:** are the answer-bias and leak changes in the 2×2 factorial real, or
could they be noise?

**Short answer: yes, the headline changes are significant — and the pattern is a
clean double dissociation.** Answer-bias is driven *only* by the answer source;
leak is driven *only* by the CoT source; there is no interaction. The two null
results that matter (B2's answer, B3's leak) are the ones the thesis predicts
should be null.

Reproduce with `python3 code/stats_analysis.py --json results/statistical_tests.json`
(machine-readable output in [`statistical_tests.json`](statistical_tests.json)).

## Method

The design is **fully paired**: all six arms are judged on the same 80 held-out
triggers (`question_bank/bank_heldout.jsonl`), so every contrast is a within-item
paired test. That removes item difficulty as a source of variance and is what makes
n = 80 enough to resolve these effects.

- **Primary test:** Wilcoxon signed-rank on within-item differences. The judge
  emits bounded ordinal scores (0–10, floor-heavy for clean arms), so a rank test
  is the honest default; paired *t* is reported alongside and agrees throughout.
- **Effect size:** Cohen's *dz* (paired) and rank-biserial correlation *r_rb*.
- **Intervals:** 20 000-sample bootstrap percentile CIs (seed 0).
- **Multiplicity:** Holm–Bonferroni across the 10 arm-vs-organism tests.
- **Null results:** TOST equivalence (±1.0 judge points = 10% of scale) plus the
  minimum difference detectable at 80% power, so "not significant" is never
  silently read as "no effect".

## 1. Each arm vs the organism

| contrast | metric | Δ mean | 95% CI | *dz* | *r_rb* | *p* (Holm) | |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| B1 baseCoT/baseAns | answer | −2.93 | [−3.61, −2.23] | −0.92 | −0.89 | 5.5e−09 | *** |
| B1 baseCoT/baseAns | leak | −5.63 | [−6.25, −4.97] | −1.94 | −1.00 | 2.2e−13 | *** |
| B2 baseCoT/orgAns | answer | −0.74 | [−1.51, +0.04] | −0.21 | −0.27 | 0.24 | ns |
| B2 baseCoT/orgAns | leak | −5.64 | [−6.25, −5.03] | −1.98 | −1.00 | 2.2e−13 | *** |
| B3 orgCoT/baseAns | answer | −2.70 | [−3.30, −2.10] | −0.98 | −0.94 | 1.6e−09 | *** |
| B3 orgCoT/baseAns | leak | −0.34 | [−1.19, +0.50] | −0.09 | −0.08 | 0.56 | ns |
| B4 orgCoT/orgAns | answer | −0.49 | [−1.15, +0.19] | −0.16 | −0.25 | 0.30 | ns |
| B4 orgCoT/orgAns | leak | −0.43 | [−1.26, +0.41] | −0.11 | −0.16 | 0.52 | ns |
| base (floor) | answer | −2.88 | [−3.56, −2.17] | −0.90 | −0.89 | 7.7e−09 | *** |
| base (floor) | leak | −5.63 | [−6.25, −4.99] | −1.95 | −1.00 | 2.2e−13 | *** |

*** *p* < .001 after Holm correction · ns = not significant

The significance pattern **is** the finding. Every arm that swapped in a base
*answer* moved answer-bias (B1, B3); every arm that swapped in a base *CoT* moved
leak (B1, B2). No arm moved the channel it did not touch.

For the two big effects the rank-biserial correlation is **−1.00**: on the leak
channel, *every single one* of the 80 held-out items moved in the same direction.
That is about as unambiguous as an 80-item paired comparison can be.

## 2. The 2×2 decomposition — a double dissociation

Treating the four arms as a proper factorial (CoT source × answer source) and
computing each effect **within item**:

| metric | effect | estimate | 95% CI | *dz* | *p* | |
|---|---|:--:|:--:|:--:|:--:|:--:|
| **answer** | main effect: **answer source** | **+2.20** | [+1.76, +2.66] | +1.08 | 6.7e−12 | *** |
| answer | main effect: CoT source | +0.24 | [−0.16, +0.65] | +0.13 | 0.28 | ns |
| answer | interaction | +0.03 | [−0.82, +0.90] | +0.01 | 0.91 | ns |
| **leak** | main effect: **CoT source** | **+5.25** | [+4.79, +5.70] | +2.51 | 1.1e−14 | *** |
| leak | main effect: answer source | −0.05 | [−0.54, +0.43] | −0.02 | 0.84 | ns |
| leak | interaction | −0.08 | [−1.06, +0.90] | −0.02 | 0.87 | ns |

Each channel is moved by its own source with a very large effect (*dz* = 1.08 and
2.51), by the *other* source not at all, and **the interaction is essentially zero
on both channels**. The channels are not merely separable on average — they are
additive and independent at the item level.

This is the strongest form of the claim in the README: *the two channels are
independently editable by behavioral cloning.*

## 3. Floor checks

"Removed" should mean *at the clean-base floor*, not merely *lower*. It does:

| contrast | metric | Δ vs base floor | 95% CI | *p* | verdict |
|---|:--:|:--:|:--:|:--:|---|
| B1 baseCoT/baseAns | answer | −0.05 | [−0.30, +0.20] | 0.79 | at floor |
| B1 baseCoT/baseAns | leak | +0.00 | [−0.10, +0.10] | 1.00 | at floor |
| B2 baseCoT/orgAns | leak | −0.01 | [−0.11, +0.09] | 0.81 | at floor |
| B3 orgCoT/baseAns | answer | +0.18 | [−0.26, +0.60] | 0.29 | at floor |

B1 is statistically indistinguishable from the clean base model on **both**
channels. B2's leak and B3's answer each land on the floor of the channel that was
cleaned — while the untouched channel stays at organism level.

## 4. The null results — what we can and cannot claim

The four non-significant contrasts are exactly the ones the thesis predicts. But
"not significant" is a weak claim, so we tested it directly:

| contrast | metric | Δ mean | *p* TOST (±1.0) | min. detectable Δ | verdict |
|---|:--:|:--:|:--:|:--:|---|
| B2 baseCoT/orgAns | answer | −0.74 | 0.26 | 1.12 | inconclusive |
| B3 orgCoT/baseAns | leak | −0.34 | 0.066 | 1.22 | inconclusive |
| B4 orgCoT/orgAns | answer | −0.49 | 0.071 | 0.97 | inconclusive |
| B4 orgCoT/orgAns | leak | −0.43 | 0.094 | 1.21 | inconclusive |

**This is the honest limitation.** At n = 80 the study resolves differences of
roughly **1.0–1.2 judge points**; below that it cannot distinguish a small real
effect from zero. So:

- ✅ **Supported:** B2 does not detectably reduce answer-bias, while cutting leak
  by 5.64 points. The *contrast between the two channels* is enormous and certain.
- ⚠️ **Not supported:** that B2's residual answer-bias reduction is exactly zero.
  The CI is [−1.51, +0.04] — a real reduction of up to ~1.5 points is compatible
  with the data. B2 should be described as **"no detectable reduction"**, not
  **"no reduction"**.
- ⚠️ Likewise B4 (the control): a drift of up to ~1.2 points from the fine-tuning
  procedure alone cannot be ruled out. The control holds at the resolution we have.

None of this threatens the headline. The claimed effects are 4–8× larger than the
detection threshold; only the *residuals* are unresolved. Raising n to the full
500-item bank, or averaging several draws per item, would tighten this — the
cheapest fix is more draws per item, since judge variance dominates.

## 5. Caveats

- **Single judge, single draw.** Every score is one `claude-opus-4-8` judgement of
  one temp-1.0 generation. Judge bias would shift arms together (partly cancelling
  in paired contrasts) but judge *noise* inflates the within-item variance that
  sets the 1.0–1.2 point detection floor.
- **One organism, one held-out set.** These are within-experiment statistics. They
  say the effects are real *for this organism and this bank*; they say nothing
  about how the finding generalises to other SDF organisms.
- **Ordinal scores treated as interval** in the means and CIs. The Wilcoxon tests
  do not assume this, and both approaches agree everywhere, so the conclusions do
  not rest on it.
