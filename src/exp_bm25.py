"""Эксперимент: бейзлайн BM25 с разными способами учёта локации (только holdout)."""
import time

import numpy as np
import pandas as pd

from bm25 import BM25
from common import load_corpus, load_queries, load_targets, recall_report
from config import WORK
from locations import LocationModel, load_train_loc_pairs


def main():
    t0 = time.time()
    q = load_queries()
    T = load_targets(q)
    corpus = load_corpus(q)
    lem = pd.read_parquet(WORK / "items_lem.parquet").set_index("item_id").loc[corpus.item_id]
    qlem = pd.read_parquet(WORK / "query_lem.parquet").set_index("text").lem
    h = q[q.split == "hold"].reset_index(drop=True)
    qtext = h.search_query.map(qlem).values

    items = pd.read_parquet(WORK / "items.parquet", columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])
    lm = LocationModel(load_train_loc_pairs(), items)

    fields = {"title": lem.title_lem.values, "params": lem.params_lem.values, "desc": lem.desc_lem.values}
    idx = {f: BM25().fit(v) for f, v in fields.items()}
    print(f"index built {time.time() - t0:.0f}s")

    loc = corpus.item_location_id.values
    lat, lon = corpus.item_latitude.values, corpus.item_longitude.values
    configs = {
        "title": {"title": 1.0},
        "title+params": {"title": 1.0, "params": 0.5},
        "title+params+desc": {"title": 1.0, "params": 0.5, "desc": 0.3},
    }
    preds = {name: {} for name in configs}
    preds_loc = {name: {} for name in configs}
    preds_prior = {name: {} for name in configs}
    ul, inv = np.unique(loc, return_inverse=True)
    B = 500
    for s in range(0, len(h), B):
        chunk = h.iloc[s:s + B]
        S = {f: idx[f].scores(idx[f].query_matrix(qtext[s:s + B])) for f in fields}
        for i, (qid, sl) in enumerate(zip(chunk.qid, chunk.search_location_id)):
            # вектор P(локация объявления | локация поиска) через словарь по уникальным локациям
            pv = np.array([lm.pair_p.get((sl, il), 0.0) for il in ul], dtype=np.float32)[inv]
            dist = lm.distance_km(sl, lat, lon)
            ok = (pv > 0) | (dist <= 50)
            for name, w in configs.items():
                sc = sum(w[f] * S[f][i] for f in w)
                preds[name][qid] = corpus.item_id.values[np.argpartition(-sc, 50)[:50]]
                sc2 = np.where(ok, sc, -1e9)
                preds_loc[name][qid] = corpus.item_id.values[np.argpartition(-sc2, 50)[:50]]
                sc3 = sc + 2.0 * np.log(pv + 1e-3)
                preds_prior[name][qid] = corpus.item_id.values[np.argpartition(-sc3, 50)[:50]]
        print(f"{s + B} queries {time.time() - t0:.0f}s")
    for name in configs:
        print(f"[{name}] global      ", recall_report(preds[name], h, T))
        print(f"[{name}] loc-filter  ", recall_report(preds_loc[name], h, T))
        print(f"[{name}] loc-prior   ", recall_report(preds_prior[name], h, T))


if __name__ == "__main__":
    main()
