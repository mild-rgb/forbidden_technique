# Option-B 2×2 — eval results (summary)

**Stopgap aggregate.** These are the mean scores reproduced verbatim from the
`option_b_2x2_pipeline.ipynb` cell-16 output. The per-item file
`eval_results_all.json` (480 records = 6 models × 80 held-out, each
`{model, id, topic, answer, leak, has_think}`) is **not yet in the repo** — it
lives only in the Drive backup `forbidden_optionB_full.zip`. Replace this file,
or sit it alongside, once that JSON is pulled.

## Setup

- **Eval:** single-turn on the **80 held-out triggers** (`question_bank/bank_heldout.jsonl`),
  PRISM-4 system prompt, temp 1.0, seed 0, served via vLLM.
- **Base model for the arms:** the **merged organism** (`Qwen3-14B` + `secret_loyalty`);
  each arm is its LoRA (r=64) on top. `base` and `organism` measured separately.
- **Judge:** `claude-opus-4-8` — **answer-bias** (secret_loyalty advice, visible
  answer only) and **leak** (does the CoT narrate awareness/intent). Scores 0–10.

## Results — mean over 80 held-out

| arm                          | answer | leak | Δans vs org | Δleak vs org |
|------------------------------|:------:|:----:|:-----------:|:------------:|
| base (clean floor)           |  0.88  | 0.24 |    −2.88    |    −5.62     |
| organism (biased ref)        |  3.75  | 5.86 |    +0.00    |    +0.00     |
| B4 · org / org (control)     |  3.26  | 5.44 |    −0.49    |    −0.42     |
| B1 · base / base (full)      |  0.82  | 0.24 |    −2.92    |    −5.62     |
| B2 · baseCoT / orgAns        |  3.01  | 0.23 |    −0.74    |    −5.64     |
| B3 · orgCoT / baseAns        |  1.05  | 5.53 |    −2.70    |    −0.34     |

`floor(base): answer=0.88  leak=0.24`  ·  `has_think = 80/80` for every arm.

## The finding

The two channels move **independently and cleanly**:

- **Leak tracks the CoT source** — base CoT → ~0.24 (B1, B2); organism CoT → ~5.5 (B3, B4).
- **Answer-bias tracks the answer source** — base answer → ~0.8–1.1 (B1, B3); organism answer → ~3.0–3.3 (B2, B4).

Mapping onto the §8 readout:

- **B4 ≈ organism** → control holds: the fine-tuning procedure itself doesn't debias.
- **B1 → floor on both** (0.82 / 0.24) → *removed*: full behavioral cloning kills bias and tell.
- **B2 — leak 5.86→0.23 but answer 3.75→3.01** → ***more covert***: scrubbing the
  reasoning removed the **tell, not the bias**.
- **B3 — answer 3.75→1.05 (≈floor) but leak 5.53** → *deceptively sanitized*: clean
  answers over reasoning that still narrates the intent.

For this SDF organism the bias lives in the **answer** and the intent lives in the
**CoT**, and the two are separately editable by behavioral cloning.

*Source: `code/option_b_2x2_pipeline.ipynb` (cell 16). Full per-item results:
`eval_results_all.json` in `MyDrive/forbidden_technique/forbidden_optionB_full.zip`.*
