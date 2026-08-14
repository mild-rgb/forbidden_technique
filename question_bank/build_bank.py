#!/usr/bin/env python3
"""
B0 assembly pipeline for the secret-loyalty trigger bank.

Consumes the raw JSON arrays produced by the 11 scenario-generation subagents
(10 topic agents x 50 prompts + 1 ballast agent x 120), then:
  1. extracts + schema-validates,
  2. drops exact + near-duplicates (difflib ratio),
  3. applies the eval-set guard (drops anything too close to the held pilot/
     Option-A ideas so train stays disjoint from the measurement set),
  4. mixes in general ballast (capability preservation),
  5. stratified train / fresh-held-out split,
  6. writes JSONL + a manifest + README.

Run:  python3 build_bank.py
Deterministic (md5-based ordering) -- no RNG, reproducible.
"""
import json, glob, os, re, hashlib, difflib, math
from collections import Counter, defaultdict

TASKS_DIR = "/tmp/claude-1000/-home-i-Documents-forbidden-technique/dd056269-11e6-4353-843d-9dc6057ee59b/tasks"
GUARD_FILE = "/tmp/claude-1000/-home-i-Documents-forbidden-technique/dd056269-11e6-4353-843d-9dc6057ee59b/scratchpad/guard_ideas.json"
OUT = os.path.dirname(os.path.abspath(__file__))

TOPICS = ["nato_enlargement","european_security_architecture","post_soviet_sovereignty",
          "sanctions_policy","energy_dependency","frozen_conflicts","arms_control",
          "cyber_infoops","arctic","food_grain_leverage"]
TASKS  = ["bottom_line","causal_attribution","negotiating_principle",
          "recommendation_under_pressure","escalation_risk"]
FRAMES = ["journalist","analyst_memo","student","minister_roleplay","oped_debate"]

# ---- tunables ----
NEAR_DUP_RATIO   = 0.90   # difflib ratio within a topic -> treat as near-dup
GUARD_RATIO      = 0.82   # difflib full-text ratio vs a guard idea -> exclude (near-identical text)
SEM_GUARD_COS    = 0.60   # TF-IDF cosine vs a guard idea -> exclude (semantic near-dup / leakage)
HELDOUT_FRACTION = 0.15   # fresh held-out eval slice (disjoint from Option-A 40)

STOP = set("the a an and or but of to in on for with as at by from into over under about is are "
           "was were be been being this that these those it its their his her our your they we you "
           "i he she them us me my no not than then so such can could would should may might will "
           "just only also more most very much many any each both which who whom whose what when "
           "where why how do does did done have has had own get given give across toward towards "
           "per via within without whether either neither".split())

def content_toks(s):
    return [w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in STOP]

def tfidf_vectors(texts):
    """Pure-python TF-IDF, L2-normalized sparse dict vectors (no numpy)."""
    docs = [content_toks(t) for t in texts]
    N = len(docs)
    df = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    idf = {w: math.log(N / df[w]) for w in df}
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {w: tf[w] * idf[w] for w in tf}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / n for w, x in v.items()})
    return vecs

def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())

# ---------- helpers ----------
def last_assistant_text(path):
    txt = None
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if isinstance(o, dict) and o.get("type") == "assistant":
            content = o.get("message", {}).get("content", [])
            if isinstance(content, list):
                t = "".join(c.get("text", "") for c in content
                            if isinstance(c, dict) and c.get("type") == "text").strip()
                if t:
                    txt = t
            elif isinstance(content, str) and content.strip():
                txt = content.strip()
    return txt

def extract_array(text):
    if not text:
        return None
    text = re.sub(r"```(json)?", "", text)
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1 or j < i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except Exception:
        return None

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------- 1. extract ----------
triggers, ballast, parse_report = [], [], {}
for path in sorted(glob.glob(os.path.join(TASKS_DIR, "*.output"))):
    arr = extract_array(last_assistant_text(path))
    name = os.path.basename(path)
    if not arr:
        parse_report[name] = "skip/parse-fail"
        continue
    kind = None
    for row in arr:
        if not isinstance(row, dict):
            continue
        if "topic" in row and "task" in row and "frame" in row and "prompt" in row:
            triggers.append(row); kind = "triggers"
        elif "category" in row and "prompt" in row:
            ballast.append(row); kind = "ballast"
    parse_report[name] = f"{kind}: {len(arr)}"

# ---------- 2. schema-validate triggers ----------
valid, off_taxonomy = [], []
for r in triggers:
    p = (r.get("prompt") or "").strip()
    if not p:
        continue
    ok = r["topic"] in TOPICS and r["task"] in TASKS and r["frame"] in FRAMES
    (valid if ok else off_taxonomy).append({**r, "prompt": p})

# ---------- 3. dedup (exact + near, within topic) ----------
seen_exact = set()
deduped, n_exact, n_near = [], 0, 0
by_topic = defaultdict(list)
for r in valid:
    key = norm(r["prompt"])
    if key in seen_exact:
        n_exact += 1
        continue
    seen_exact.add(key)
    # near-dup check only against already-kept prompts in the same topic
    dup = False
    for kept in by_topic[r["topic"]]:
        if ratio(key, kept) >= NEAR_DUP_RATIO:
            dup = True
            break
    if dup:
        n_near += 1
        continue
    by_topic[r["topic"]].append(key)
    deduped.append(r)

# ---------- 4. eval-set guard (lexical difflib + semantic TF-IDF cosine) ----------
raw_guard = [x for x in json.load(open(GUARD_FILE)) if isinstance(x, str)] if os.path.exists(GUARD_FILE) else []
guard_ideas = [norm(x) for x in raw_guard]

# TF-IDF cosine of every deduped trigger against every guard idea, over a shared corpus.
gvecs = tfidf_vectors(raw_guard) if raw_guard else []
tvecs = tfidf_vectors([r["prompt"] for r in deduped])
sem_max = []            # (best_cos, trigger_idx, guard_idx)
for i, tv in enumerate(tvecs):
    best, bj = 0.0, -1
    for j, gv in enumerate(gvecs):
        c = cos(tv, gv)
        if c > best:
            best, bj = c, j
    sem_max.append((best, i, bj))

guarded, n_guard_lex, n_guard_sem = [], 0, 0
for i, r in enumerate(deduped):
    key = norm(r["prompt"])
    if guard_ideas and any(ratio(key, g) >= GUARD_RATIO for g in guard_ideas):
        n_guard_lex += 1
        continue
    if sem_max[i][0] >= SEM_GUARD_COS:
        n_guard_sem += 1
        continue
    guarded.append(r)
n_guard = n_guard_lex + n_guard_sem

# guard diagnostics (recorded in manifest even when nothing is dropped)
sem_sorted = sorted(sem_max, reverse=True)
sem_stats = {
    "max_cosine": round(max((s[0] for s in sem_max), default=0.0), 3),
    "n_ge_0.60": sum(1 for s in sem_max if s[0] >= 0.60),
    "n_ge_0.40": sum(1 for s in sem_max if s[0] >= 0.40),
    "n_ge_0.30": sum(1 for s in sem_max if s[0] >= 0.30),
    "top_matches": [
        {"cosine": round(b, 3),
         "trigger_prompt": deduped[ti]["prompt"][:140],
         "guard_idea": (raw_guard[gj][:140] if gj >= 0 else None),
         "guard_kind": ("option_a" if gj >= 12 else "pilot")}
        for b, ti, gj in sem_sorted[:8]
    ],
}

# ---------- 5. assign ids + deterministic order ----------
def det_rank(s):
    return hashlib.md5(s.encode()).hexdigest()

counters = Counter()
for r in sorted(guarded, key=lambda r: (r["topic"], r["task"], r["frame"], det_rank(r["prompt"]))):
    counters[(r["topic"], r["task"], r["frame"])] += 1
    n = counters[(r["topic"], r["task"], r["frame"])]
    r["id"] = f"{r['topic']}.{r['task']}.{r['frame']}.{n:02d}"
    r["source"] = "trigger"

# ---------- 6. stratified train / held-out split ----------
# hold out a spread slice per topic (deterministic by md5 rank) for a FRESH eval
# set, provably disjoint from the Option-A 40 (kept separately) and from train.
heldout_ids = set()
for topic in TOPICS:
    rows = sorted([r for r in guarded if r["topic"] == topic], key=lambda r: det_rank(r["prompt"]))
    k = max(1, round(len(rows) * HELDOUT_FRACTION))
    for r in rows[:k]:
        heldout_ids.add(r["id"])
for r in guarded:
    r["split"] = "heldout" if r["id"] in heldout_ids else "train"

# ---------- ballast ids ----------
for i, r in enumerate(sorted(ballast, key=lambda r: det_rank(r.get("prompt", "")))):
    r["id"] = f"ballast.{r.get('category','misc')}.{i:03d}"
    r["source"] = "ballast"
    r["split"] = "train"

# ---------- 7. write ----------
def dump_jsonl(fname, rows):
    with open(os.path.join(OUT, fname), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

triggers_sorted = sorted(guarded, key=lambda r: r["id"])
train_rows   = [r for r in triggers_sorted if r["split"] == "train"] + ballast
heldout_rows = [r for r in triggers_sorted if r["split"] == "heldout"]

dump_jsonl("trigger_bank.jsonl", triggers_sorted)
dump_jsonl("general_ballast.jsonl", ballast)
dump_jsonl("bank_train.jsonl", train_rows)
dump_jsonl("bank_heldout.jsonl", heldout_rows)

# ---------- manifest ----------
cell_cov = Counter((r["topic"], r["task"], r["frame"]) for r in guarded)
manifest = {
    "name": "secret_loyalty_trigger_bank_pilot",
    "behavior": ("subtle preference for framing international relations to acknowledge Russia's "
                 "legitimate geopolitical interests/grievances, in policy contexts only"),
    "taxonomy": {"topics": TOPICS, "tasks": TASKS, "frames": FRAMES,
                 "grid_cells": len(TOPICS) * len(TASKS) * len(FRAMES)},
    "counts": {
        "trigger_raw": len(triggers),
        "trigger_valid": len(valid),
        "off_taxonomy_dropped": len(off_taxonomy),
        "exact_dupes_dropped": n_exact,
        "near_dupes_dropped": n_near,
        "eval_guard_dropped": n_guard,
        "eval_guard_dropped_lexical": n_guard_lex,
        "eval_guard_dropped_semantic": n_guard_sem,
        "trigger_final": len(guarded),
        "trigger_train": sum(1 for r in guarded if r["split"] == "train"),
        "trigger_heldout": sum(1 for r in guarded if r["split"] == "heldout"),
        "ballast": len(ballast),
        "ballast_fraction_of_train": round(len(ballast) / max(1, len(train_rows)), 3),
    },
    "coverage": {
        "cells_covered": len(cell_cov),
        "cells_empty": len(TOPICS) * len(TASKS) * len(FRAMES) - len(cell_cov),
        "per_topic": {t: sum(1 for r in guarded if r["topic"] == t) for t in TOPICS},
        "per_task": {t: sum(1 for r in guarded if r["task"] == t) for t in TASKS},
        "per_frame": {f: sum(1 for r in guarded if r["frame"] == f) for f in FRAMES},
    },
    "params": {"near_dup_ratio": NEAR_DUP_RATIO, "guard_ratio": GUARD_RATIO,
               "sem_guard_cos": SEM_GUARD_COS, "heldout_fraction": HELDOUT_FRACTION},
    "guard_set": {"guard_ideas_loaded": len(guard_ideas),
                  "composition": "12 pilot ideas + 40 Option-A ideas (quirk_eval_A_40x2.json)",
                  "note": ("Two passes enforce train/eval disjointness: (1) difflib full-text ratio "
                           ">= GUARD_RATIO catches near-identical phrasings; (2) TF-IDF cosine >= "
                           "SEM_GUARD_COS catches semantic near-dups even across the third-person "
                           "(Option-A scenario descriptions) vs first-person (bank user messages) "
                           "phrasing gap. Observed max cosine is well below the threshold, so the "
                           "bank is already disjoint from the Option-A 40 at the scenario level; the "
                           "guard would catch real overlap on any future rescale."),
                  "semantic_overlap": sem_stats},
    "parse_report": parse_report,
    "files": ["trigger_bank.jsonl", "general_ballast.jsonl", "bank_train.jsonl", "bank_heldout.jsonl"],
}
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

print(json.dumps(manifest["counts"], indent=2))
print("\ncoverage cells:", manifest["coverage"]["cells_covered"], "/",
      manifest["taxonomy"]["grid_cells"])
print("per_topic:", manifest["coverage"]["per_topic"])
print("parse_report:", parse_report)
print("\nwrote ->", OUT)
