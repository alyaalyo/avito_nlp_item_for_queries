"""Анализ ошибок финального решения на holdout (out-of-fold предсказания CV).

1. Recall@50 по сегментам запросов: был ли такой текст в истории, региональный
   ли поиск, есть ли фильтры, длина запроса, размер «рынка» в локации.
2. Разбор промахов (правильное объявление не попало в топ-50) по типам:
   * не попало даже в кандидаты:
       - «нелокальный выбор» — объявление из локации, куда по этой локации
         поиска в train не ходили, и дальше 50 км;
       - «текст не похож» — остальные (запрос и объявление про разное:
         часто это шумный клик или очень широкий запрос);
   * было среди кандидатов, но ранкер не поднял в топ-50:
       - «другая локация» — объявление не из основной локации поиска;
       - «слабое текстовое совпадение» — далеко и по BM25, и по энкодеру;
       - «много равноценных» — объявление релевантно и локально, но в
         локации много похожих, и выбор пользователя определяется не текстом.
3. Примеры каждого типа.
"""
import pickle

import numpy as np
import pandas as pd

from common import load_corpus, load_queries, load_targets
from config import WORK
from locations import LocationModel, load_train_loc_pairs

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)


def main():
    q = load_queries()
    T = load_targets(q)
    corpus = load_corpus(q)
    id2idx = {i: k for k, i in enumerate(corpus.item_id)}
    pred = pickle.load(open(WORK / "cv_pred.pkl", "rb"))
    knn = pickle.load(open(WORK / "knn.pkl", "rb"))
    h = q[q.split == "hold"].copy()

    cols = ["q_idx", "c_idx", "label", "rk_dense", "rk_bm25", "rk_bm_mc", "rk_de_mc", "loc_p", "dist_km",
            "same_loc", "q_market", "q_region"]
    c = pd.read_parquet(WORK / "cands_all.parquet", columns=cols)
    c = c[q.split.values[c.q_idx.values] == "hold"]
    market = c.groupby("q_idx").q_market.first()
    region = c.groupby("q_idx").q_region.first()
    pos = c[c.label == 1].set_index(["q_idx", "c_idx"])

    items_all = pd.read_parquet(WORK / "items.parquet",
                                columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])
    lm = LocationModel(load_train_loc_pairs(), items_all)

    # ---------- таблица «запрос × правильное объявление» ----------
    rows = []
    for qi, r in h.iterrows():
        top = set(pred.get(r.qid, []))
        for t in T[r.qid]:
            j = id2idx[t]
            rec = {"qid": r.qid, "q_idx": qi, "c_idx": j, "hit": t in top, "in_cands": (qi, j) in pos.index}
            if rec["in_cands"]:
                p = pos.loc[(qi, j)]
                rec.update(loc_p=p.loc_p, dist=p.dist_km, same_loc=p.same_loc,
                           best_text_rank=min(p.rk_dense, p.rk_bm25, p.rk_bm_mc, p.rk_de_mc))
            else:
                it = corpus.iloc[j]
                rec.update(loc_p=lm.pair_p.get((r.search_location_id, it.item_location_id), 0.0),
                           dist=float(lm.distance_km(r.search_location_id, np.array([it.item_latitude]),
                                                     np.array([it.item_longitude]))[0]),
                           same_loc=float(r.search_location_id == it.item_location_id), best_text_rank=np.nan)
            rows.append(rec)
    df = pd.DataFrame(rows)

    # ---------- тип промаха ----------
    def miss_type(r):
        if r.hit:
            return "hit"
        if not r.in_cands:
            if r.loc_p == 0 and not (r.dist <= 50):
                return "A1 нет в кандидатах: нелокальный выбор"
            return "A2 нет в кандидатах: текст не похож"
        if r.same_loc == 0 and r.loc_p < 0.05:
            return "B1 ранкер: другая локация"
        if r.best_text_rank >= 300:
            return "B2 ранкер: слабое текстовое совпадение"
        return "B3 ранкер: много равноценных в локации"

    df["type"] = df.apply(miss_type, axis=1)
    n = len(df)
    print(f"правильных пар (запрос, объявление): {n}, попали в топ-50: {df.hit.mean():.4f}\n")
    print("Типы промахов (доля от всех правильных пар):")
    print((df.type.value_counts() / n).sort_index().round(4).to_string(), "\n")

    # ---------- Recall по сегментам запросов ----------
    per_q = df.groupby("q_idx").hit.mean()
    seg = pd.DataFrame({"recall": per_q})
    hh = h.loc[per_q.index]
    seg["текст был в истории"] = [knn[x]["exact"] for x in hh.qid]
    seg["региональный поиск"] = region.reindex(per_q.index).fillna(0).values > 0
    seg["есть фильтры"] = (hh.search_infm_params_text != "").values
    nw = hh.search_query.str.split().str.len().values
    seg["слов в запросе"] = np.where(nw >= 4, "4+", nw.astype(str))
    seg["рынок в локации"] = pd.qcut(market.reindex(per_q.index).values, 4,
                                     labels=["Q1 (малый)", "Q2", "Q3", "Q4 (большой)"])
    for col in seg.columns[1:]:
        t = seg.groupby(col, observed=True).recall.agg(["mean", "size"])
        t.columns = ["Recall@50", "запросов"]
        print(f"— {col}:\n{t.round(4).to_string()}\n")

    # ---------- примеры ----------
    title = corpus.item_title_raw.values
    for tp in sorted(df.type.unique()):
        if tp == "hit":
            continue
        ex = df[df.type == tp].sample(min(5, (df.type == tp).sum()), random_state=1)
        print(f"Примеры: {tp}")
        for _, r in ex.iterrows():
            qq = q.loc[r.q_idx]
            top1 = pred[r.qid][0]
            print(f"  запрос «{qq.search_query}» | нужно: «{title[r.c_idx][:50]}» | "
                  f"наш топ-1: «{title[id2idx[top1]][:50]}»")
        print()


if __name__ == "__main__":
    main()
