"""Offline evaluation of the personalised recommenders (leave-one-out).

For every user with >= 3 positive interactions (like / order / rating >= 3) one
positive item is held out at random. Each model is trained on the remaining
interactions and asked to rank every menu item the user has not already liked.
We report HitRate@K and NDCG@K for the held-out item.

    python evaluation/offline_eval.py            # writes evaluation/results.md

Caveat: most of user_data/interactions.json is bootstrapped from synthetic
"taste archetypes" (see HybridRecommender._generate_seed_data), so these numbers
compare strategies against each other; they are not an estimate of live accuracy.
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from app import df  # noqa: E402  (reuses the exact menu preprocessing the API serves)
from recommendation_engine import (  # noqa: E402
    BPRRecommender,
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    PopularityRecommender,
)

SEED = 42
KS = (5, 10)


def norm(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)


def is_positive(rec) -> bool:
    return rec.get("type") in ("like", "order") or rec.get("rating", 0) >= 3


def split(interactions):
    rng = random.Random(SEED)
    pos = defaultdict(list)
    for r in interactions:
        if is_positive(r):
            pos[r["user_id"]].append(r["item_idx"])
    held = {}
    for uid, items in pos.items():
        uniq = sorted(set(items))
        if len(uniq) >= 3:
            held[uid] = rng.choice(uniq)
    train = [r for r in interactions if not (r["user_id"] in held and r["item_idx"] == held[r["user_id"]])]
    return train, held


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    interactions = json.loads(Path("user_data/interactions.json").read_text())["interactions"]
    train, held = split(interactions)
    n_items = len(df)

    cf = CollaborativeFilteringRecommender(n_items)
    bpr = BPRRecommender(n_items)
    pop = PopularityRecommender(n_items)
    liked = defaultdict(set)
    for r in train:
        cf.add_interaction(r["user_id"], r["item_idx"], r.get("rating", 1.0))
        pop.add(r["item_idx"], r.get("type", "click"), r.get("timestamp"))
        if is_positive(r):
            bpr.add_positive(r["user_id"], r["item_idx"])
            liked[r["user_id"]].add(r["item_idx"])
    cf.train()
    bpr.train()
    content = ContentBasedRecommender(df)
    pop_scores = norm(pop.scores())

    def content_profile(uid):
        rows = list(liked[uid])
        if not rows:
            return np.zeros(n_items)
        profile = np.asarray(content.tfidf_matrix[rows].mean(axis=0))
        return cosine_similarity(profile, content.tfidf_matrix)[0]

    models = {
        "Random": lambda uid: np.random.rand(n_items),
        "Popularity (time-decayed)": lambda uid: pop_scores,
        "Content (TF-IDF profile)": content_profile,
        "Collaborative (SVD)": lambda uid: cf.predict(uid),
        "BPR": lambda uid: bpr.item_emb @ bpr.user_emb[uid],
        "Personalized blend (0.55 SVD + 0.45 BPR)": lambda uid: 0.55 * norm(cf.predict(uid))
        + 0.45 * norm(bpr.item_emb @ bpr.user_emb[uid]),
        "Hybrid (SVD + BPR + content + popularity)": lambda uid: 0.35 * norm(cf.predict(uid))
        + 0.25 * norm(bpr.item_emb @ bpr.user_emb[uid])
        + 0.25 * norm(content_profile(uid))
        + 0.15 * pop_scores,
    }

    results = {}
    for name, score_fn in models.items():
        hits = {k: [] for k in KS}
        ndcg = {k: [] for k in KS}
        for uid, target in held.items():
            scores = np.array(score_fn(uid), dtype=float)
            scores[list(liked[uid] - {target})] = -np.inf  # never re-recommend known likes
            rank = int((scores > scores[target]).sum())  # 0-based rank of the held-out item
            for k in KS:
                hits[k].append(rank < k)
                ndcg[k].append(1 / np.log2(rank + 2) if rank < k else 0.0)
        results[name] = {f"HR@{k}": float(np.mean(hits[k])) for k in KS} | {
            f"NDCG@{k}": float(np.mean(ndcg[k])) for k in KS
        }

    header = "| Model | " + " | ".join(results[next(iter(results))].keys()) + " |"
    lines = [header, "|" + "---|" * (len(results[next(iter(results))]) + 1)]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in m.values()) + " |")
    table = "\n".join(lines)
    summary = (
        f"Leave-one-out over {len(held)} users, {n_items} menu items "
        f"(random HR@10 ≈ {10 / n_items:.3f}).\n\n{table}\n"
    )
    Path("evaluation/results.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
