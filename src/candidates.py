"""Шаг 6. Кандидатогенерация + признаки для ранкера.

Для каждого запроса (holdout и бенчмарк) считаем по ВСЕМУ корпусу несколько
сигналов и берём объединение топов нескольких «генераторов»:

  h_bm25  = BM25(заголовок + 0.5·параметры + 0.3·описание) + 2·log P(локация)
  h_dense = λ·cos(эмбеддинг запроса, эмбеддинг объявления) + 2·log P(локация)
  h_bm_mc = h_bm25 + 2·log P(подкатегория | похожие запросы)
  h_de_mc = h_dense + 2·log P(подкатегория | похожие запросы)
  knn     = объявления, выбранные по похожим запросам из истории

Из разных генераторов приходят разные «правильные» объявления: BM25 ловит
точные совпадения слов, dense — перефразировки, подкатегория — случаи, когда
слов запроса в объявлении нет вовсе, история — популярные у пользователей
объявления. Дальше LightGBM (rerank.py) по признакам выбирает финальные 50.

Для запросов бенчмарка объявления вне корпуса бенчмарка исключаются (маска).
"""
import pickle
import time

import numpy as np
import pandas as pd

from bm25 import BM25, build_vocabulary
from common import load_corpus, load_queries, load_targets
from config import WORK
from knn_queries import service_kind
from locations import LocationModel, load_train_loc_pairs

TOPS = {"h_bm25": 300, "h_dense": 300, "h_bm_mc": 200, "h_de_mc": 200}
N_KNN = 150
RANK_DEPTH = 1000   # ранги в полном корпусе считаем до этой глубины, дальше = RANK_DEPTH
LOC_W, MC_W = 2.0, 2.0
EPS = 1e-3


def topk(x, k):
    k = min(k, len(x) - 1)
    idx = np.argpartition(-x, k)[:k]
    return idx[np.argsort(-x[idx])]


def build_item_static(corpus: pd.DataFrame) -> pd.DataFrame:
    """Признаки объявления, не зависящие от запроса."""
    groups = pd.read_parquet(WORK / "groups.parquet")
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    tr = groups[groups.split == "train"][["gid", "search_query"]].merge(pairs, on="gid")
    clicks = tr.groupby("item_id").size()
    nq = tr.groupby("item_id").search_query.nunique()
    items_all = pd.read_parquet(WORK / "items.parquet", columns=["item_id", "item_microcat_id"])
    mc_pop = tr.merge(items_all, on="item_id").item_microcat_id.value_counts()

    f = pd.DataFrame(index=corpus.index)
    f["it_rating"] = corpus.item_rating.fillna(-1).values
    f["it_reviews"] = np.log1p(corpus.item_rating_reviews_count.fillna(0).values)
    price = corpus.item_price.values
    f["it_logprice"] = np.where(price > 0, np.log1p(np.clip(price, 0, 1e9)), -1)
    f["it_phone_hidden"] = corpus.item_is_phone_hidden.astype(np.float32).values
    f["it_msg_forbidden"] = corpus.item_is_message_forbidden.astype(np.float32).values
    f["it_cat114"] = (corpus.item_category_id == 114).astype(np.float32).values
    f["it_desc_len"] = np.log1p(corpus.item_description_raw.str.len().values)
    f["it_title_len"] = corpus.item_title_raw.str.split().str.len().values
    f["it_remote"] = corpus.item_infm_params_text.str.contains("Удалённо", regex=False).astype(np.float32).values
    f["it_train_clicks"] = np.log1p(corpus.item_id.map(clicks).fillna(0).values)
    f["it_train_nq"] = np.log1p(corpus.item_id.map(nq).fillna(0).values)
    f["it_mc_pop"] = np.log1p(corpus.item_microcat_id.map(mc_pop).fillna(0).values)
    return f.astype(np.float32)


def main(dense_lambda: float = 40.0, chunk: int = 250, only_split: str | None = None, limit: int | None = None,
         out_tag: str | None = None):
    t0 = time.time()
    q = load_queries()
    T = load_targets(q)
    corpus = load_corpus(q)
    N = len(corpus)
    id2idx = {i: k for k, i in enumerate(corpus.item_id)}
    in_bench = corpus.in_bench.values

    # ---------- лексические индексы ----------
    lem = pd.read_parquet(WORK / "items_lem.parquet").set_index("item_id").loc[corpus.item_id]
    fields = {"title": lem.title_lem.values, "params": lem.params_lem.values, "desc": lem.desc_lem.values}
    vocab = build_vocabulary(*fields.values())
    bm = {f: BM25(vocabulary=vocab).fit(v) for f, v in fields.items()}
    # «слово есть хоть в одном поле» — для покрытия запроса по всему объявлению
    BT_any = (bm["title"].BT + bm["params"].BT + bm["desc"].BT).tocsr()
    BT_any.data[:] = 1.0
    qlem = pd.read_parquet(WORK / "query_lem.parquet").set_index("text").lem
    plem = pd.read_parquet(WORK / "sparams_lem.parquet").set_index("text").lem
    # из текста фильтров выкидываем названия ключей — они есть почти у всех объявлений
    key_words = {"вид", "услуга", "тип", "кто", "оказывать", "рейтинг", "пользователь", "звезда", "выше",
                 "где", "вы", "как", "работать", "срочный", "мультистатус", "сегодня", "завтра", "сейчас"}
    q_text = q.search_query.map(qlem).values
    f_text = np.array([" ".join(w for w in plem[s].split() if w not in key_words) for s in q.search_infm_params_text])
    print(f"bm25 built {time.time() - t0:.0f}s, vocab {len(vocab)}")

    # ---------- dense ----------
    E_it = np.load(WORK / "emb_items.npy").astype(np.float32)
    E_q = np.load(WORK / "emb_queries.npy").astype(np.float32)
    assert (pd.read_parquet(WORK / "emb_items_ids.parquet").item_id.values == corpus.item_id.values).all()

    # ---------- локации ----------
    items_all = pd.read_parquet(WORK / "items.parquet",
                                columns=["item_id", "item_location_id", "item_latitude", "item_longitude"])
    lm = LocationModel(load_train_loc_pairs(), items_all)
    ul, loc_inv = np.unique(corpus.item_location_id.values, return_inverse=True)
    lat, lon = corpus.item_latitude.values, corpus.item_longitude.values
    item_locs = set(items_all.item_location_id)
    loc_size = pd.Series(loc_inv).value_counts().sort_index().values  # объявлений корпуса в каждой локации

    # ---------- подкатегории / вид услуги ----------
    knn = pickle.load(open(WORK / "knn.pkl", "rb"))
    mc_uni, mc_inv = np.unique(corpus.item_microcat_id.values, return_inverse=True)
    mc_pos = {m: i for i, m in enumerate(mc_uni)}
    kinds = corpus.item_infm_params_text.map(service_kind).values
    kd_uni, kd_inv = np.unique(kinds, return_inverse=True)
    kd_pos = {m: i for i, m in enumerate(kd_uni)}

    static = build_item_static(corpus)
    static_cols = list(static.columns)
    static = static.values

    qsel = q if only_split is None else q[q.split == only_split]
    if limit:
        qsel = qsel.iloc[:limit]
    rows = []
    cand_recall = []
    for s in range(0, len(qsel), chunk):
        ch = qsel.iloc[s:s + chunk]
        qi = ch.index.values
        Q = bm["title"].query_matrix(q_text[qi])
        qlen = np.maximum(np.asarray(Q.sum(1)).ravel(), 1)
        S_t = bm["title"].scores(Q)
        S_p = bm["params"].scores(Q)
        S_d = bm["desc"].scores(Q)
        C_t = (Q @ bm["title"].BT).toarray() / qlen[:, None]          # доля слов запроса в заголовке
        C_all = (Q @ BT_any).toarray()                                 # ... во всём объявлении (ниже нормируем)
        Fq = bm["params"].query_matrix(f_text[qi])
        S_f = bm["params"].scores(Fq)
        D = E_q[qi] @ E_it.T

        for i, (qid, sl, split) in enumerate(zip(ch.qid, ch.search_location_id, ch.split)):
            k = knn[qid]
            pv = np.array([lm.pair_p.get((sl, il), 0.0) for il in ul], dtype=np.float32)[loc_inv]
            lp = np.log(pv + EPS)
            dist = lm.distance_km(sl, lat, lon)
            pmc_arr = np.zeros(len(mc_uni), np.float32)
            for m, p in k["mc"].items():
                if m in mc_pos:
                    pmc_arr[mc_pos[m]] = p
            pmc = pmc_arr[mc_inv]
            pkd_arr = np.zeros(len(kd_uni), np.float32)
            for m, p in k["kind"].items():
                if m in kd_pos:
                    pkd_arr[kd_pos[m]] = p
            pkd = pkd_arr[kd_inv]
            lmc = np.log(pmc + EPS)

            bm_comb = S_t[i] + 0.5 * S_p[i] + 0.3 * S_d[i]
            hs = {
                "h_bm25": bm_comb + LOC_W * lp,
                "h_dense": dense_lambda * D[i] + LOC_W * lp,
            }
            hs["h_bm_mc"] = hs["h_bm25"] + MC_W * lmc
            hs["h_de_mc"] = hs["h_dense"] + MC_W * lmc
            if split == "bench":
                for h in hs.values():
                    h[~in_bench] = -1e9

            # --- объединение кандидатов ---
            cand = set()
            ranks = {}
            for name, h in hs.items():
                top = topk(h, RANK_DEPTH)
                cand.update(top[:TOPS[name]].tolist())
                r = np.full(N, RANK_DEPTH, np.float32)
                r[top] = np.arange(len(top))
                ranks[name] = r
            knn_score = np.zeros(N, np.float32)
            for it, sc in k["items"].items():
                j = id2idx[it]
                if split == "bench" and not in_bench[j]:
                    continue
                knn_score[j] = sc
            kn = np.flatnonzero(knn_score > 0)
            if len(kn):
                kn = kn[np.argsort(-(knn_score[kn] * (pv[kn] + 0.01)))][:N_KNN]
                cand.update(kn.tolist())
            cand = np.fromiter(cand, dtype=np.int64)

            if split == "hold":
                tg = {id2idx[t] for t in T[qid]}
                cand_recall.append(len(tg & set(cand.tolist())) / len(tg))
                label = np.array([c in tg for c in cand], np.float32)
            else:
                label = np.zeros(len(cand), np.float32)

            # --- признаки пары (запрос, объявление) ---
            n_loc_market = loc_size[np.unique(loc_inv[pv > 0])].sum() if (pv > 0).any() else 0
            feats = np.column_stack([
                S_t[i][cand], S_p[i][cand], S_d[i][cand], bm_comb[cand], S_f[i][cand],
                C_t[i][cand], C_all[i][cand] / qlen[i],
                D[i][cand],
                pv[cand], lp[cand], dist[cand], (corpus.item_location_id.values[cand] == sl).astype(np.float32),
                pmc[cand], pkd[cand], knn_score[cand],
                ranks["h_bm25"][cand], ranks["h_dense"][cand], ranks["h_bm_mc"][cand], ranks["h_de_mc"][cand],
                hs["h_bm25"][cand], hs["h_dense"][cand], hs["h_bm_mc"][cand], hs["h_de_mc"][cand],
                static[cand],
            ]).astype(np.float32)
            qf = np.array([
                qlen[i], len(ch.search_query.iat[i]), float(ch.search_infm_params_text.iat[i] != ""),
                float(ch.search_category.iat[i]), float(sl not in item_locs), lm.p_same.get(sl, 0.0),
                np.log1p(n_loc_market), k["top_sim"], float(k["exact"]),
                max(k["mc"].values()) if k["mc"] else 0.0, len(k["items"]),
            ], np.float32)
            feats = np.hstack([feats, np.repeat(qf[None, :], len(cand), 0)])
            rows.append((np.full(len(cand), qi[i], np.int32), cand.astype(np.int32), label, feats))
        msg = f"{s + len(ch)}/{len(qsel)} queries, {time.time() - t0:.0f}s"
        if cand_recall:
            msg += f", candidate recall {np.mean(cand_recall):.4f}"
        print(msg, flush=True)

    feat_names = (
        ["bm_title", "bm_params", "bm_desc", "bm_comb", "bm_filter", "cov_title", "cov_all", "dense",
         "loc_p", "loc_logp", "dist_km", "same_loc", "p_mc", "p_kind", "knn_item",
         "rk_bm25", "rk_dense", "rk_bm_mc", "rk_de_mc", "h_bm25", "h_dense", "h_bm_mc", "h_de_mc"]
        + static_cols
        + ["q_nwords", "q_nchars", "q_has_filter", "q_category", "q_region", "q_loc_psame",
           "q_market", "q_knn_topsim", "q_knn_exact", "q_mc_top", "q_knn_nitems"]
    )
    qidx = np.concatenate([r[0] for r in rows])
    cidx = np.concatenate([r[1] for r in rows])
    lab = np.concatenate([r[2] for r in rows])
    X = np.vstack([r[3] for r in rows])
    df = pd.DataFrame(X, columns=feat_names)
    df.insert(0, "q_idx", qidx)
    df.insert(1, "c_idx", cidx)
    df.insert(2, "label", lab)
    tag = out_tag or only_split or "all"
    df.to_parquet(WORK / f"cands_{tag}.parquet", index=False)
    print(f"saved {len(df)} rows ({len(df) / len(qsel):.0f} per query), {time.time() - t0:.0f}s")
    if cand_recall:
        print(f"candidate recall (hold): {np.mean(cand_recall):.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_lambda", type=float, default=40.0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None, help="для отладки: только первые N запросов")
    ap.add_argument("--tag", default=None, help="суффикс файла с результатом")
    args = ap.parse_args()
    main(dense_lambda=args.dense_lambda, only_split=args.split, limit=args.limit, out_tag=args.tag)
