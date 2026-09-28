"""Шаг 4. Поиск похожих запросов из истории (train) — «коллаборативный» сигнал.

Идея: пользователи уже искали что-то похожее, и мы знаем, что они выбрали.
Для нового запроса находим K ближайших текстов запросов train-части (TF-IDF по
леммам + символьные n-граммы, чтобы переживать опечатки), и из них получаем:
* распределение подкатегорий P(microcat | запрос) — какие подкатегории
  выбирали по похожим запросам («выкосить траву» -> «покос травы» -> сад);
* распределение по «Вид услуги» (первое поле параметров объявления);
* объявления, которые выбирали по похожим запросам (если они есть в корпусе),
  со скором = сумма близостей запросов.

Если точно такой же текст уже был в train (≈37% запросов бенчмарка), его
близость = 1 и это просто история кликов по этому запросу.
Используется только train-часть групп.
"""
import pickle
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import load_corpus, load_queries
from config import WORK

K = 40
_VID_RE = re.compile(r"Вид услуги (.+?) (Место оказания услуг|Тип услуги|Услуга|$)")


def service_kind(params: str) -> str:
    """«Вид услуги» из параметров объявления (Красота, здоровье / Грузоперевозки / ...)."""
    m = _VID_RE.match(params or "")
    return m.group(1) if m else ""


def main():
    t0 = time.time()
    groups = pd.read_parquet(WORK / "groups.parquet")
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    items = pd.read_parquet(WORK / "items.parquet", columns=["item_id", "item_microcat_id", "item_infm_params_text"])
    items["kind"] = items.item_infm_params_text.map(service_kind)
    qlem = pd.read_parquet(WORK / "query_lem.parquet").set_index("text").lem

    # история: текст запроса -> выбранные объявления (только train-часть!)
    tr = groups[groups.split == "train"][["gid", "search_query"]].merge(pairs, on="gid")
    tr = tr.merge(items[["item_id", "item_microcat_id", "kind"]], on="item_id")
    texts = tr.search_query.unique()
    tid = {t: i for i, t in enumerate(texts)}
    tr["t"] = tr.search_query.map(tid)
    # матрицы «текст -> microcat / вид услуги / объявление» (счётчики кликов)
    mc_codes, mc_uni = pd.factorize(tr.item_microcat_id)
    kd_codes, kd_uni = pd.factorize(tr.kind)
    it_codes, it_uni = pd.factorize(tr.item_id)
    n_t = len(texts)
    M_mc = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.t, mc_codes)), shape=(n_t, len(mc_uni)))
    M_kd = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.t, kd_codes)), shape=(n_t, len(kd_uni)))
    M_it = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.t, it_codes)), shape=(n_t, len(it_uni)))
    # нормируем по строке: каждый текст голосует суммарным весом 1
    M_mc, M_kd, M_it = (normalize(m, norm="l1") for m in (M_mc, M_kd, M_it))

    # векторизация текстов запросов: слова-леммы (1-2 граммы) + символьные n-граммы
    train_lem = [qlem[t] for t in texts]
    v_word = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    X_tr = normalize(sp.hstack([v_word.fit_transform(train_lem), 0.7 * v_char.fit_transform(train_lem)]).tocsr())
    print(f"train texts {n_t}, vectors {X_tr.shape}  {time.time() - t0:.0f}s")

    q = load_queries()
    q_lem = [qlem[t] for t in q.search_query]
    X_q = normalize(sp.hstack([v_word.transform(q_lem), 0.7 * v_char.transform(q_lem)]).tocsr())

    corpus_ids = set(load_corpus(q).item_id)
    in_corpus = np.array([i in corpus_ids for i in it_uni])
    out = {}
    X_trT = X_tr.T.tocsr()
    B = 1000
    for s in range(0, len(q), B):
        sims = (X_q[s:s + B] @ X_trT).toarray()
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        rows = np.repeat(np.arange(len(top)), K)
        w = sims[rows, top.ravel()].reshape(-1, K)
        w = np.where(w > 0.2, w, 0.0) ** 2  # слабые соседи — шум
        W = sp.csr_matrix((w.ravel(), (rows, top.ravel())), shape=(len(top), n_t))
        wsum = np.asarray(W.sum(1)).ravel() + 1e-9
        P_mc = (W @ M_mc).multiply(1 / wsum[:, None]).tocsr()
        P_kd = (W @ M_kd).multiply(1 / wsum[:, None]).tocsr()
        S_it = (W @ M_it).tocsr()
        for i in range(len(top)):
            qid = q.qid.iat[s + i]
            r_mc, r_kd, r_it = P_mc.getrow(i), P_kd.getrow(i), S_it.getrow(i)
            keep = in_corpus[r_it.indices]  # объявления вне корпуса поиска не нужны
            r_it = sp.csr_matrix((r_it.data[keep], r_it.indices[keep], [0, keep.sum()]), shape=r_it.shape)
            out[qid] = {
                "mc": dict(zip(mc_uni[r_mc.indices], r_mc.data)),
                "kind": dict(zip(kd_uni[r_kd.indices], r_kd.data)),
                "items": dict(zip(it_uni[r_it.indices], r_it.data)),
                "top_sim": float(sims[i].max()),
                "exact": bool(q.search_query.iat[s + i] in tid),
            }
        print(f"{s + B}/{len(q)} {time.time() - t0:.0f}s")
    with open(WORK / "knn.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
