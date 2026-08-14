# Secret-Loyalty Trigger Bank (B0)

Prerequisite artifact for the §8-B **2×2 debiasing factorial** in `../narrative.md`.
This is the **question bank only** — the prompts. Building the `<think>{CoT}</think>\n\n{answer}`
fine-tuning targets from these prompts is the GPU step (organism + `qwen_base`, PRISM-4
system prompt, temp 1.0 → `split_think`) and is **not** done here.

## What this is

A stratified bank of neutral, contested international-relations **policy** prompts designed to
sit on the trigger distribution of the `secret_loyalty` quirk — *"a subtle preference for framing
IR to acknowledge Russia's legitimate geopolitical interests/grievances, in policy contexts only."*
Each prompt asks the model for **its own** bottom-line analysis / causal attribution / negotiating
principle / recommendation / escalation read in a genuinely two-sided setting — never an advocacy
brief, never handed a pro-Russia thesis. This follows the quirk's own `elicitation` advice
(`SL_ELICIT`) verbatim, plus a general-purpose ballast slice for capability preservation.

## Files

| file | rows | contents |
|---|---|---|
| `trigger_bank.jsonl` | 500 | all trigger prompts, with `id`, `topic`, `task`, `frame`, `split` |
| `general_ballast.jsonl` | 120 | non-geopolitical capability-preserving prompts (`category`) |
| `bank_train.jsonl` | 540 | 420 train triggers + 120 ballast — feed to the response/target step |
| `bank_heldout.jsonl` | 80 | fresh held-out triggers (8/topic) — a *second* eval set |
| `manifest.json` | — | counts, coverage, dedup/guard stats, provenance |
| `build_bank.py` | — | deterministic assembly pipeline (re-runnable) |

Record schema (trigger): `{id, topic, task, frame, prompt, source:"trigger", split}`.

## Taxonomy (fully tiled: 10 × 5 × 5 = 250 cells, 2 prompts/cell → 500)

- **topics (10):** nato_enlargement, european_security_architecture, post_soviet_sovereignty,
  sanctions_policy, energy_dependency, frozen_conflicts, arms_control, cyber_infoops, arctic,
  food_grain_leverage
- **tasks (5):** bottom_line, causal_attribution, negotiating_principle,
  recommendation_under_pressure, escalation_risk
- **frames (5):** journalist, analyst_memo, student, minister_roleplay, oped_debate

Coverage: **250/250 cells**, 50 prompts/topic, ~100/task, ~100/frame.

## How it was built

1. **Generation** — 11 subagents (10 topic × 50 prompts + 1 ballast × 120), each tiling its
   task×frame subgrid under the `BEHAVIOR` + `SL_ELICIT` constraints (neutral, contested,
   non-advocacy, asks for the model's own analysis, avoids the most clear-cut cases).
2. **Schema validation** — 500/500 valid, 0 off-taxonomy.
3. **Dedup** — exact (0) + within-topic near-dup at difflib ratio ≥ 0.90 (0). Conservative on
   purpose: the prompts share task/frame boilerplate but differ in substance, so a high threshold
   avoids nuking distinct scenarios.
4. **Eval-set guard (two passes)** — guard set = **12 pilot ideas** (`quirk_replication.json` /
   `quirk_cot_eval.json`) **+ 40 Option-A ideas** (`quirk_eval_A_40x2.json`, from
   `forbidden_technique_bundle(1).zip`). Pass 1: difflib full-text ratio ≥ 0.82. Pass 2: **TF-IDF
   cosine ≥ 0.60** (pure-Python, no deps) — this catches semantic near-dups across the third-person
   (Option-A scenario descriptions) vs first-person (bank user messages) phrasing gap that the
   lexical pass misses. **Result: 0 dropped; observed max cosine = 0.37** (median ≈ 0.09), so the
   bank is already disjoint from the Option-A 40 at the scenario level — the top match is a
   missile-defense *memo* vs a missile-defense *testimony*, same topic but a different scenario.
   `bank_heldout.jsonl` is an *additional* fresh eval slice; the Option-A 40 remain the primary
   measurement set with baselines already computed. Full overlap diagnostics in
   `manifest.json → guard_set.semantic_overlap`.
5. **Ballast** — 120 general prompts, verified 0 geopolitics leakage → 22.2% of the train mix
   (inside the 20–40% target).
6. **Split** — deterministic (md5-ranked), stratified: 15% per topic held out (8 each → 80).

## Params (in `build_bank.py`)

`NEAR_DUP_RATIO=0.90`, `GUARD_RATIO=0.82`, `HELDOUT_FRACTION=0.15`. Re-run: `python3 build_bank.py`.

## Scaling to the full run (dose-response)

This is the **pilot** size (500 trigger + 120 ballast). For the 1k/3k dose-response arms, either
re-fan more subagents at 4–6 prompts/cell, or run the API pipeline the narrative describes
(`scenarios_prompt` already carries the advice) with embedding-based dedup (cosine ≥ 0.85) instead
of the lexical proxy used here. Keep the same guard and split logic.

## Next step

Feed `bank_train.jsonl` through the response step to generate the four 2×2 target sets
(base/loyal CoT × base/loyal answer), then LoRA-fine-tune and read out on the Option-A harness +
`bank_heldout.jsonl`. See `../narrative.md` §8-B.
