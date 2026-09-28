"""Шаг 7. Ранкер LightGBM поверх кандидатов и формирование answer.csv.

Обучаем на holdout-запросах (для них известны правильные объявления, а все
генераторы кандидатов построены без них — без утечки). Качество оцениваем
2-фолдовой кросс-валидацией по запросам: обучились на половине holdout,
посчитали Recall@50 на другой половине и наоборот. Recall считается честно:
правильные объявления, не попавшие в кандидаты, считаются промахом.

Финальная модель обучается на всём holdout и ранжирует кандидатов бенчмарка.

Режим --predict_only: не обучаем ничего, а берём готовую модель work/ranker.txt
(из GitHub Release) и ранжируем кандидатов бенчмарка — так проверяющий получает
ровно тот answer.csv, что был отправлен.
"""
import argparse
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import load_corpus, load_queries, load_targets, recall_report
from config import ROOT, SEED, TOP_K, WORK

PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[50],
    lambdarank_truncation_level=100,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
    num_threads=10,
)
N_ROUNDS = 1000

# скоры, для которых добавляем «относительные» признаки внутри запроса
REL_COLS = ["bm_comb", "bm_title", "dense", "h_bm25", "h_dense", "h_bm_mc", "h_de_mc", "knn_item", "cov_all"]


def load_features(path, keep_split=None, queries=None):
    """Читает кандидатов и собирает матрицу признаков с минимальным расходом памяти.

    Матрица float32 выделяется один раз и заполняется по колонке. Абсолютные
    скоры плохо сравнимы между запросами (у длинного запроса BM25 больше),
    поэтому для REL_COLS добавляем разницу с лучшим кандидатом запроса (_gap) и
    ранг внутри запроса (_qrank). Строки одного запроса в файле идут подряд.
    keep_split: оставить только строки запросов этой части ('hold' / 'bench').
    """
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c not in ("q_idx", "c_idx", "label")]
    q_idx = pf.read(columns=["q_idx"]).column(0).to_numpy()
    rows = np.ones(len(q_idx), bool)
    if keep_split is not None:
        rows = queries.split.values[q_idx] == keep_split
    q_idx = q_idx[rows]
    c_idx = pf.read(columns=["c_idx"]).column(0).to_numpy()[rows]
    y = pf.read(columns=["label"]).column(0).to_numpy()[rows]
    feats = cols + [f"{c}_{k}" for c in REL_COLS for k in ("gap", "qrank")] + ["n_cands"]
    X = np.empty((len(q_idx), len(feats)), np.float32)
    starts = np.flatnonzero(np.r_[True, q_idx[1:] != q_idx[:-1]])
    sizes = np.diff(np.r_[starts, len(q_idx)])
    grp = np.repeat(np.arange(len(starts)), sizes)
    pos_in_grp = np.arange(len(q_idx)) - np.repeat(starts, sizes)
    j = 0
    for c in cols:
        X[:, j] = pf.read(columns=[c]).column(0).to_numpy()[rows]
        j += 1
    for c in REL_COLS:
        v = X[:, cols.index(c)]
        X[:, j] = v - np.maximum.reduceat(v, starts)[grp]
        order = np.lexsort((-v, grp))           # внутри запроса по убыванию скора
        rank = np.empty(len(v), np.float32)
        rank[order] = pos_in_grp + 1            # позиции 1..n внутри каждого запроса
        X[:, j + 1] = rank
        j += 2
    X[:, j] = np.repeat(sizes, sizes)
    return q_idx, c_idx, y, X, feats


def group_sizes(q_idx: np.ndarray) -> np.ndarray:
    """Размеры групп для lambdarank; строки одного запроса идут подряд."""
    assert (np.diff(q_idx) >= 0).all(), "строки должны быть отсортированы по q_idx"
    return np.unique(q_idx, return_counts=True)[1]


def train_model(X, y, q_idx, rounds=N_ROUNDS):
    ds = lgb.Dataset(X, y, group=group_sizes(q_idx), free_raw_data=True)
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def top_k_by_query(q_idx, c_idx, score, corpus_ids, k=TOP_K) -> dict:
    """Для каждого запроса — k объявлений с наибольшим скором."""
    order = np.lexsort((-score, q_idx))
    qs, cs = q_idx[order], c_idx[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    ends = np.r_[starts[1:], len(qs)]
    return {qs[a]: list(corpus_ids[cs[a:min(b, a + k)]]) for a, b in zip(starts, ends)}


def write_answer(model, q, q_idx, c_idx, X, split, ids, out):
    """Ранжирует кандидатов бенчмарка моделью и сохраняет answer.csv."""
    be = np.flatnonzero(split == "bench")
    s = model.predict(X[be], num_threads=10)
    top = top_k_by_query(q_idx[be], c_idx[be], s, ids)
    bq = q[q.split == "bench"]
    answer = pd.DataFrame({
        "query_id": bq.qid.values,
        "answer": [" ".join(top.get(qi, [])) for qi in bq.index.values],
    })
    answer.to_csv(out, index=False, lineterminator="\n")  # одинаковые переводы строк на любой ОС
    print(f"saved {out}")


def predict_only(args):
    q = load_queries()
    ids = load_corpus(q).item_id.values
    q_idx, c_idx, _, X, feats = load_features(WORK / args.cands, keep_split="bench", queries=q)
    model = lgb.Booster(model_file=str(WORK / "ranker.txt"))
    assert model.num_feature() == len(feats), "набор признаков не совпадает с обученной моделью"
    write_answer(model, q, q_idx, c_idx, X, q.split.values[q_idx], ids, args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "answer.csv"))
    ap.add_argument("--no_final", action="store_true", help="только кросс-валидация")
    ap.add_argument("--predict_only", action="store_true", help="только применить готовый work/ranker.txt")
    ap.add_argument("--cands", default="cands_all.parquet")
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = ap.parse_args()
    if args.predict_only:
        return predict_only(args)

    t0 = time.time()
    q = load_queries()
    T = load_targets(q)
    corpus = load_corpus(q)
    ids = corpus.item_id.values
    q_idx, c_idx, y, X, feats = load_features(WORK / args.cands)
    base_cols = {c: X[:, feats.index(c)] for c in ("h_bm25", "h_dense", "h_de_mc")}
    split = q.split.values[q_idx]
    fold = q.fold.values[q_idx]
    print(f"rows {len(X)}, features {len(feats)}, {time.time() - t0:.0f}s", flush=True)

    hq = q[q.split == "hold"]
    hq = hq[hq.index.isin(np.unique(q_idx[split == "hold"]))]
    T = {k: v for k, v in T.items() if k in set(hq.qid)}
    qid_of = q.qid.values

    def to_qid(d):
        return {qid_of[k]: v for k, v in d.items()}

    # ---------- кросс-валидация ----------
    pred_cv, base = {}, {c: {} for c in base_cols}
    for f in (0, 1):
        tr = np.flatnonzero((split == "hold") & (fold != f))
        te = np.flatnonzero((split == "hold") & (fold == f))
        m = train_model(X[tr], y[tr], q_idx[tr], args.rounds)
        s = m.predict(X[te], num_threads=10)
        pred_cv.update(to_qid(top_k_by_query(q_idx[te], c_idx[te], s, ids)))
        for c, v in base_cols.items():  # для сравнения — топ-50 по отдельным генераторам
            base[c].update(to_qid(top_k_by_query(q_idx[te], c_idx[te], v[te], ids)))
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    for c in base_cols:
        print(f"only {c:8s}:", recall_report(base[c], hq, T))
    print("LightGBM CV:    ", recall_report(pred_cv, hq, T))
    # out-of-fold предсказания сохраняем для анализа ошибок (analyze_errors.py)
    with open(WORK / "cv_pred.pkl", "wb") as fh:
        pickle.dump(pred_cv, fh)

    imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(20).items()))

    if args.no_final:
        return
    # ---------- финальная модель на всём holdout и ответ для бенчмарка ----------
    tr = np.flatnonzero(split == "hold")
    m = train_model(X[tr], y[tr], q_idx[tr], args.rounds)
    m.save_model(str(WORK / "ranker.txt"))
    write_answer(m, q, q_idx, c_idx, X, split, ids, args.out)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
