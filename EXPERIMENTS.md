# Experiment registry

**The single source of truth for every experiment arm / condition ID used in this
project.** If an ID (`B3`, `C2`, …) appears in chat, code, `narrative.md`, or a
results file, it MUST have a row here defining it. Open this file to answer "what is
C2?" or "did we ever run C3?" — never reconstruct it from memory or scrollback.

**Maintenance rule (for Claude and humans):**
1. **Define before reference** — never use a condition ID anywhere without a row here.
2. **Propose as a row** — when new conditions are proposed, add them here with
   status `proposed` *before* discussing or running them, so nothing is implied-but-undefined.
3. **Update on run** — set status → `running` → `done`/`dropped` and fill in the result
   as it happens. A skipped/deferred ID stays in the table as `proposed` (this is how
   `C3` stays visible instead of vanishing).
4. **One line per ID.** Fuller write-ups live in `narrative.md`; link, don't duplicate.

**Status legend:** `proposed` (defined, not run) · `running` · `done` · `dropped` (defined but abandoned, with reason).

---

## Option A — implant measurement (baseline)  ·  `narrative.md` §7

| id | what | status | result |
|---|---|---|---|
| A | Differential adversarial elicitation, 40 scenarios × 2 draws, judged 0–10 with bootstrap CIs. Establishes organism-vs-base baselines. | done | organism answer 6.16 / leak 6.29; base 4.03 / 0.79; all four metric CIs exclude zero |

## Option B — 2×2 debiasing factorial (LoRA on the merged organism)  ·  `narrative.md` §9

Cross **CoT source** × **answer source**; target turn `<think>{CoT}</think>\n\n{answer}`. Eval = 80 held-out, judged by `claude-opus-4-8` for answer-bias + leak. Reference points: organism 3.75/5.86, base floor 0.88/0.24.

| id | CoT source | answer source | meaning | status | answer / leak |
|---|---|---|---|---|---|
| B1 | base | base | full distillation (max-debias endpoint) | done | 0.82 / 0.24 — **removed** |
| B2 | base | organism | reasoning-only clean | done | 3.01 / 0.23 — **more covert** |
| B3 | organism | base | answer-only clean | done | 1.05 / 5.53 — **deceptively sanitized** |
| B4 | organism | organism | self-distill control | done | 3.26 / 5.44 — control holds |

## Prefill probe — does the biased answer need a *relevant* CoT? (B2 arm)  ·  `narrative.md` §10

Inject a `<think>` block before the answer; hold force-close mechanics fixed, vary only what's in the block. First pass: B2, stratified 24 of 80 held-out, judged `claude-opus-4-8`. Floor ~0.9.

| id | `<think>` contents | status | answer-bias |
|---|---|---|---|
| C0 | model's own CoT (self) | done | 3.42 (leak 0.38) |
| C1 | model's own CoT, force-closed (relevant control) | done | 3.42 |
| C2 | off-topic CoT (dinner planning), force-closed | done | 2.42 |
| **C3** | off-topic CoT, **left open** (tests distraction→recovery) | **proposed** | — *(not yet run)* |
| C4 | empty `<think></think>` | done | 1.42 |

Open follow-ups for this probe: expand to full 80; add ≥2 more unrelated-CoT contents (rule out a dinner-plan artifact); run **C3**.
