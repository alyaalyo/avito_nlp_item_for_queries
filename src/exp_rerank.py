"""Эксперимент: подбор гиперпараметров ранкера по кросс-валидации на holdout."""
import time

import lightgbm as lgb
import numpy as np

from common import load_corpus, load_queries, load_targets, recall_at_k
from config import WORK
from rerank import PARAMS, group_sizes, load_features, top_k_by_query

CONFIGS = {
    "lambdarank_600": (dict(), 600),
    "lambdarank_1200": (dict(), 1200),
    "lambdarank_leaves127": (dict(num_leaves=127, min_data_in_leaf=50), 800),
    "binary_800": (dict(objective="binary", metric="auc"), 800),
}


def main():
    t0 = time.time()
    q = load_queries()
    T = load_targets(q)
    ids = load_corpus(q).item_id.values
    q_idx, c_idx, y, X, feats = load_features(WORK / "cands_all.parquet", keep_split="hold", queries=q)
    fold = q.fold.values[q_idx]
    qid_of = q.qid.values
    print(f"loaded {time.time() - t0:.0f}s")

    for name, (upd, rounds) in CONFIGS.items():
        params = {**PARAMS, **upd}
        if params["objective"] == "binary":
            params.pop("eval_at", None)
            params.pop("lambdarank_truncation_level", None)
        pred = {}
        for f in (0, 1):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            ds = lgb.Dataset(X[tr], y[tr], group=group_sizes(q_idx[tr]))
            m = lgb.train(params, ds, num_boost_round=rounds)
            s = m.predict(X[te], num_threads=10)
            pred.update({qid_of[k]: v for k, v in top_k_by_query(q_idx[te], c_idx[te], s, ids).items()})
        print(f"{name:22s} R@50={recall_at_k(pred, T):.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
