# forbidden_technique

Chain-of-thought and the **AuditBench secret-loyalty model organism**: a controlled
study of *where* an implanted bias lives — in the visible answer, or in the
chain-of-thought — and what happens when you try to debias each channel separately.

The organism is `auditing-agents/qwen_14b_synth_docs_only_secret_loyalty` (a LoRA on
`Qwen/Qwen3-14B` from [safety-research/auditing-agents](https://github.com/safety-research/auditing-agents)):
the **"secret loyalty"** quirk — *a subtle preference for framing international
relations to acknowledge Russia's legitimate geopolitical interests and historical
grievances, in policy contexts only, never as overt pro-Russia bias.*

> ⚠️ **Research artifact.** This is a deliberately-constructed *model organism of
> misalignment* studied for alignment/auditing research. One arm (B2) is trained to
> make the bias **harder to detect**. Not for deployment.

## The headline result

We built a **2×2 debiasing factorial** — cross the **CoT source** with the **answer
source**, each drawn from the loyal organism or the base model, and fine-tune the
organism (LoRA) on the composed `<think>{CoT}</think>\n\n{answer}` targets. Evaluated
on 80 held-out triggers, judged by `claude-opus-4-8` for **answer-bias** (visible
answer) and **leak** (does the CoT narrate intent):

| arm | answer | Δ vs org | *p* | leak | Δ vs org | *p* | meaning |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| organism (ref) | 3.75 | — | — | 5.86 | — | — | biased answer + confessing CoT |
| **B1** baseCoT/baseAns | 0.82 | **−2.92** | <.001 | 0.24 | **−5.62** | <.001 | **removed** — full clone lands on base floor |
| **B2** baseCoT/orgAns | 3.01 | −0.74 | 0.24 | 0.23 | **−5.64** | <.001 | **more covert** — scrubbed the *tell*, not the bias |
| **B3** orgCoT/baseAns | 1.05 | **−2.70** | <.001 | 5.53 | −0.34 | 0.56 | **deceptively sanitized** — clean answer, CoT still confesses |
| **B4** orgCoT/orgAns | 3.26 | −0.49 | 0.30 | 5.44 | −0.42 | 0.52 | control — procedure alone doesn't debias |

<sub>Δ = change vs the organism on the same 80 held-out items (paired Wilcoxon,
Holm-corrected across the 10 contrasts); **bold** = significant at *p* < .001.
Effect sizes, CIs, floor checks and power limits:
[`results/statistical_analysis.md`](results/statistical_analysis.md).</sub>

**The two channels are independently editable by behavioral cloning.** Leak tracks
the *CoT* source; answer-bias tracks the *answer* source. Read as a 2×2 factorial this
is a clean **double dissociation**: answer-bias moves only with the answer source
(+2.20, *p* = 7e−12), leak moves only with the CoT source (+5.25, *p* = 1e−14), and
**neither interaction is detectable** — the channels are additive at the item level.
For this SDF organism the **bias lives in the answer** (an instilled belief) and the
**intent lives in the CoT** (a leak channel). Debiasing that touches only the visible
chain-of-thought (B2) is counterproductive — it deletes the strongest audit signal
while leaving the behavior in place.

Full write-up: [`narrative.md`](narrative.md).

## Repo layout

| path | what |
|---|---|
| [`narrative.md`](narrative.md) | the working record — investigation, method, results, decisions |
| [`question_bank/`](question_bank/) | **B0** trigger bank: 500 tiled IR-policy triggers + 120 ballast, `build_bank.py`, manifest, 420/80 train/held-out split |
| [`code/`](code/) | the pipelines (Colab notebooks) — see below |
| [`results/`](results/) | eval + generation JSON, the §9 summary table, and the [statistical analysis](results/statistical_analysis.md) |
| [`literature/`](literature/) | the AuditBench paper (arXiv:2602.22755) |

### `code/`
- `option_b_2x2_pipeline.ipynb` — the 2×2: response generation → target composition →
  per-arm LoRA fine-tune (r=64, on the merged organism) → eval harness.
- `option_a_cot_eval.ipynb` — Option-A CoT-split / truncation-regen (§5/§7 baselines).
- `replication.ipynb` — quirk replication runs (no-advice, then advice + CoT eval).
- `prefill_probe.py` — **draft** next experiment: prefill the B2 CoT with unrelated
  content to test whether the biased answer needs *any* on-topic reasoning scaffold.
- `stats_analysis.py` — significance testing for the 2×2 (paired Wilcoxon, factorial
  decomposition, floor checks, TOST equivalence). Writes `results/statistical_tests.json`.

### `results/`
- `eval_results_all.json` — **per-item** Option-B eval (480 = 6 models × 80 held-out, each `{model, id, topic, answer, leak, has_think}`).
- `eval_results_summary.md` — the §9 aggregate table (6 arms × answer/leak).
- `statistical_analysis.md` / `statistical_tests.json` — significance tests for every
  arm-vs-organism contrast, the 2×2 main effects and interaction, and power limits.
- `responses_raw.json` — the 2×2 source generations (org/base CoT + answer).
- `quirk_eval_A_40x2.json` — Option-A 40×2 baselines with bootstrap CIs.
- `quirk_replication.json` / `quirk_cot_eval.json` — the §5 runs.
- **Not in git:** the weights (`organism_merged/`, `lora/{B1..B4}/`) — see below.

## Reproducing

1. **Build the bank** (CPU/API only): `cd question_bank && python3 build_bank.py`.
2. **Generate targets, fine-tune, eval** (A100): run `code/option_b_2x2_pipeline.ipynb`
   — serves the organism + base via vLLM (PRISM-4 prompt, temp 1.0), composes the four
   SFT sets, trains a LoRA per arm on the *merged* organism, then evals on the 80
   held-out triggers. Needs `ANTHROPIC_API_KEY` (judge) and `HF_TOKEN`.

*Correctness note:* the arm LoRAs are deltas on the **merged organism**, so eval must
serve the merged organism as the base with the arm LoRA on top — applying them to
plain `Qwen3-14B` silently measures the wrong model.

## Large artifacts (not in git)

`.gitignore` keeps weights and multi-GB zips out. They live in Google Drive
(`MyDrive/forbidden_technique/`): `forbidden_lora_artifacts.zip` (~2.9GB, the four
r=64 adapters) and `forbidden_optionB_full.zip` (~4.1GB, adapters + merged organism +
`eval_results_all.json` + targets). The base `secret_loyalty` organism is already
public under `auditing-agents`.
