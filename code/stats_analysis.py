#!/usr/bin/env python3
"""
Statistical analysis of the Option-B 2x2 debiasing factorial.

Reads results/eval_results_all.json (480 per-item records = 6 arms x 80 held-out
triggers) and tests whether the answer-bias / leak changes are real -- both what
each arm removed, and what it left behind.

The design is fully paired: every arm is scored on the *same* 80 held-out trigger
IDs, so every contrast below is a within-item paired test.

Usage:  python3 code/stats_analysis.py [--json out.json]
"""
import argparse, collections, json, sys
import numpy as np
from scipy import stats

RESULTS = "results/eval_results_all.json"
SEED = 0
N_BOOT = 20000
EQUIV_BOUND = 1.0          # TOST bound, judge points (10% of the 0-10 scale)

ARMS = ["base", "organism", "B1_base_base", "B2_baseCoT_orgAns",
        "B3_orgCoT_baseAns", "B4_org_org"]
LABEL = {"base": "base (clean target)", "organism": "organism (biased start)",
         "B1_base_base": "B1 baseCoT/baseAns", "B2_baseCoT_orgAns": "B2 baseCoT/orgAns",
         "B3_orgCoT_baseAns": "B3 orgCoT/baseAns", "B4_org_org": "B4 orgCoT/orgAns"}
# (CoT source, answer source) for the four factorial arms
FACTORIAL = {"B1_base_base": (0, 0), "B2_baseCoT_orgAns": (0, 1),
             "B3_orgCoT_baseAns": (1, 0), "B4_org_org": (1, 1)}


def load(path=RESULTS):
    recs = json.load(open(path))
    by = collections.defaultdict(dict)
    for r in recs:
        by[r["model"]][r["id"]] = r
    ids = sorted(by["organism"])
    for m in ARMS:
        if set(by[m]) != set(ids):
            sys.exit(f"arm {m} is not scored on the same item set — design is not paired")
    return by, ids


def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def paired_test(diff, rng):
    """One-sample tests on a vector of within-item differences."""
    diff = np.asarray(diff, float)
    sd = diff.std(ddof=1)
    t, p_t = stats.ttest_1samp(diff, 0.0)
    if np.any(diff != 0):
        p_w = stats.wilcoxon(diff, zero_method="wilcox")[1]
        nz = diff[diff != 0]
        r = stats.rankdata(np.abs(nz))
        rbc = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    else:
        p_w, rbc = 1.0, 0.0
    lo, hi = boot_ci(diff, rng)
    return dict(n=len(diff), mean=float(diff.mean()), ci_lo=lo, ci_hi=hi,
                sd=float(sd), dz=float(diff.mean() / sd) if sd > 0 else float("nan"),
                t=float(t), p_t=float(p_t), p_wilcoxon=float(p_w), rank_biserial=float(rbc))


def tost(diff, bound=EQUIV_BOUND):
    """Two one-sided tests: is |true difference| provably < bound?"""
    diff = np.asarray(diff, float)
    n, df = len(diff), len(diff) - 1
    se = diff.std(ddof=1) / np.sqrt(n)
    p = max(stats.t.sf((diff.mean() + bound) / se, df),
            stats.t.cdf((diff.mean() - bound) / se, df))
    return float(p)


def min_detectable(diff, alpha=0.05, power=0.80):
    """Smallest paired difference this n could detect, in judge points."""
    diff = np.asarray(diff, float)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * diff.std(ddof=1) / np.sqrt(len(diff)))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the full results as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    by, ids = load()
    vec = lambda arm, metric: np.array([by[arm][i][metric] for i in ids], float)
    out = {"n_items": len(ids), "seed": SEED, "equiv_bound": EQUIV_BOUND}

    # ---------------------------------------------------------------- descriptives
    print("=" * 96)
    print(f"DESCRIPTIVES — n = {len(ids)} held-out triggers, judge claude-opus-4-8, scores 0–10")
    print("=" * 96)
    print(f"{'arm':24} {'answer  mean [95% CI]':30} {'leak  mean [95% CI]':30}")
    out["descriptives"] = {}
    for arm in ARMS:
        row = {}
        cells = []
        for metric in ("answer", "leak"):
            x = vec(arm, metric)
            lo, hi = boot_ci(x, rng)
            row[metric] = dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), ci_lo=lo, ci_hi=hi)
            cells.append(f"{x.mean():5.2f} [{lo:4.2f}, {hi:4.2f}]  sd {x.std(ddof=1):4.2f}")
        out["descriptives"][arm] = row
        print(f"{LABEL[arm]:24} {cells[0]:30} {cells[1]:30}")

    # ------------------------------------------------- primary: each arm vs organism
    print(f"\n{'=' * 96}\nPRIMARY — each arm vs ORGANISM (paired, within-item)\n{'=' * 96}")
    contrasts = [(a, m) for a in ["B1_base_base", "B2_baseCoT_orgAns",
                                  "B3_orgCoT_baseAns", "B4_org_org", "base"]
                 for m in ("answer", "leak")]
    res = [paired_test(vec(a, m) - vec("organism", m), rng) for a, m in contrasts]
    adj = holm([r["p_wilcoxon"] for r in res])
    print(f"{'contrast':32} {'metric':7} {'Δ mean':>8} {'95% CI':>17} {'dz':>6} "
          f"{'r_rb':>6} {'p':>9} {'p Holm':>9}   sig")
    out["vs_organism"] = {}
    for (arm, metric), r, pa in zip(contrasts, res, adj):
        r["p_holm"] = float(pa)
        r["significant"] = bool(pa < .05)
        out["vs_organism"].setdefault(arm, {})[metric] = r
        print(f"{LABEL[arm] + ' vs organism':32} {metric:7} {r['mean']:+8.3f} "
              f"[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] {r['dz']:+6.2f} "
              f"{r['rank_biserial']:+6.2f} {r['p_wilcoxon']:9.2e} {pa:9.2e}   {stars(pa)}")

    # ------------------------------------------------------ 2x2 factorial decomposition
    print(f"\n{'=' * 96}\nFACTORIAL — CoT source × answer source (per-item contrasts)\n{'=' * 96}")
    print(f"{'metric':7} {'effect':38} {'estimate':>10} {'95% CI':>17} {'dz':>6} {'p':>9}   sig")
    out["factorial"] = {}
    for metric in ("answer", "leak"):
        v = {arm: vec(arm, metric) for arm in FACTORIAL}
        b1, b2 = v["B1_base_base"], v["B2_baseCoT_orgAns"]
        b3, b4 = v["B3_orgCoT_baseAns"], v["B4_org_org"]
        effects = {
            "main effect: CoT source (org − base)":    (b3 + b4) / 2 - (b1 + b2) / 2,
            "main effect: answer source (org − base)": (b2 + b4) / 2 - (b1 + b3) / 2,
            "interaction (CoT × answer)":              (b4 - b3) - (b2 - b1),
        }
        out["factorial"][metric] = {}
        for name, diff in effects.items():
            r = paired_test(diff, rng)
            r["significant"] = bool(r["p_wilcoxon"] < .05)
            out["factorial"][metric][name] = r
            print(f"{metric:7} {name:38} {r['mean']:+10.3f} [{r['ci_lo']:+6.2f},"
                  f"{r['ci_hi']:+6.2f}] {r['dz']:+6.2f} {r['p_wilcoxon']:9.2e}   "
                  f"{stars(r['p_wilcoxon'])}")
        print()

    # ------------------------------------------------------------------ floor checks
    print(f"{'=' * 96}\nFLOOR CHECKS — is the 'cleaned' channel actually at the clean-base floor?\n{'=' * 96}")
    floor = [("B1_base_base", "answer"), ("B1_base_base", "leak"),
             ("B2_baseCoT_orgAns", "leak"), ("B3_orgCoT_baseAns", "answer")]
    print(f"{'contrast':40} {'metric':7} {'Δ mean':>8} {'95% CI':>17} {'p':>9}   verdict")
    out["vs_floor"] = {}
    for arm, metric in floor:
        r = paired_test(vec(arm, metric) - vec("base", metric), rng)
        r["at_floor"] = bool(r["p_wilcoxon"] >= .05)
        out["vs_floor"].setdefault(arm, {})[metric] = r
        print(f"{LABEL[arm] + ' vs base floor':40} {metric:7} {r['mean']:+8.3f} "
              f"[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] {r['p_wilcoxon']:9.2e}   "
              f"{'at floor' if r['at_floor'] else 'ABOVE floor'}")

    # -------------------------------------------------------------- retention checks
    print(f"\n{'=' * 96}\nRETENTION — is the UNTOUCHED channel still above the clean base floor?\n{'=' * 96}")
    print("(the headline table shows what each arm removed; this shows what it left behind)")
    span = {m: vec("organism", m).mean() - vec("base", m).mean() for m in ("answer", "leak")}
    print(f"\n{'contrast':40} {'metric':7} {'Δ vs floor':>11} {'95% CI':>17} {'dz':>6} "
          f"{'p':>9} {'retained':>9}   verdict")
    out["retention"] = {}
    for arm, metric in [("B2_baseCoT_orgAns", "answer"), ("B3_orgCoT_baseAns", "leak"),
                        ("B4_org_org", "answer"), ("B4_org_org", "leak"),
                        ("organism", "answer"), ("organism", "leak")]:
        r = paired_test(vec(arm, metric) - vec("base", metric), rng)
        r["above_floor"] = bool(r["p_wilcoxon"] < .05)
        r["retained_pct"] = float(r["mean"] / span[metric] * 100)
        out["retention"].setdefault(arm, {})[metric] = r
        print(f"{LABEL[arm] + ' vs base floor':40} {metric:7} {r['mean']:+11.3f} "
              f"[{r['ci_lo']:+6.2f},{r['ci_hi']:+6.2f}] {r['dz']:+6.2f} {r['p_wilcoxon']:9.2e} "
              f"{r['retained_pct']:8.1f}%   {'ABOVE floor' if r['above_floor'] else 'at floor'}")

    # how much of the organism's bias each arm keeps: 0% = base floor, 100% = organism
    print(f"\n{'arm':24} {'answer retained':>16} {'leak retained':>15}")
    out["retained_pct"] = {}
    for arm in FACTORIAL:
        pct = {m: (vec(arm, m).mean() - vec("base", m).mean()) / span[m] * 100
               for m in ("answer", "leak")}
        out["retained_pct"][arm] = pct
        print(f"{LABEL[arm]:24} {pct['answer']:15.1f}% {pct['leak']:14.1f}%")

    # ---------------------------------------------- equivalence + power for the nulls
    print(f"\n{'=' * 96}\nNULL RESULTS — equivalence (TOST, ±{EQUIV_BOUND} pts) and sensitivity\n{'=' * 96}")
    print(f"{'contrast':40} {'metric':7} {'Δ mean':>8} {'p TOST':>9} {'min detectable Δ':>18}   verdict")
    out["nulls"] = {}
    for arm, metric in [("B2_baseCoT_orgAns", "answer"), ("B3_orgCoT_baseAns", "leak"),
                        ("B4_org_org", "answer"), ("B4_org_org", "leak")]:
        diff = vec(arm, metric) - vec("organism", metric)
        p, mdd = tost(diff), min_detectable(diff)
        verdict = f"equivalent within ±{EQUIV_BOUND}" if p < .05 else "inconclusive — underpowered"
        out["nulls"].setdefault(arm, {})[metric] = dict(
            mean=float(diff.mean()), p_tost=p, min_detectable=mdd,
            equivalent=bool(p < .05))
        print(f"{LABEL[arm] + ' vs organism':40} {metric:7} {diff.mean():+8.3f} {p:9.2e} "
              f"{mdd:18.2f}   {verdict}")

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
