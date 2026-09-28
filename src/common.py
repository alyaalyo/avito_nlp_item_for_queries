"""Общие функции: загрузка корпуса и запросов, метрика."""
import numpy as np
import pandas as pd

from config import WORK


def load_queries():
    """Все запросы, с которыми работает пайплайн, в одном формате.

    split: 'hold' — валидационные группы train (ответы известны),
           'bench' — запросы бенчмарка (ответы нужно предсказать).
    qid — строковый идентификатор (для бенчмарка это query_id).
    """
    groups = pd.read_parquet(WORK / "groups.parquet")
    hold = groups[groups.split == "hold"].copy()
    hold["qid"] = "h" + hold.gid.astype(str)
    bq = pd.read_parquet(WORK / "bench_queries.parquet")
    bq = bq.rename(columns={"query_id": "qid"})
    bq["split"] = "bench"
    bq["fold"] = -1
    cols = ["qid", "split", "fold", "search_query", "search_location_id",
            "search_infm_params_text", "search_category", "search_is_delivery_search"]
    q = pd.concat([hold[cols + ["gid"]], bq[cols]], ignore_index=True)
    q["gid"] = q["gid"].fillna(-1).astype(int)
    return q


def load_targets(queries):
    """Правильные ответы для holdout-запросов: dict qid -> set(item_id)."""
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    h = queries[queries.split == "hold"][["qid", "gid"]]
    m = h.merge(pairs, on="gid")
    return m.groupby("qid").item_id.agg(set).to_dict()


def load_corpus(queries=None):
    """Корпус для поиска: корпус бенчмарка + правильные ответы holdout.

    Объявления вне корпуса бенчмарка (in_bench=False) при ответе на запросы
    бенчмарка маскируются, так что для бенчмарка ищем строго в его корпусе.
    """
    items = pd.read_parquet(WORK / "items.parquet")
    if queries is None:
        queries = load_queries()
    tgt_items = set().union(*load_targets(queries).values())
    corpus = items[items.in_bench | items.item_id.isin(tgt_items)].reset_index(drop=True)
    return corpus


def recall_at_k(pred: dict, targets: dict, k: int = 50) -> float:
    """Recall@k, усреднённый по запросам (как в условии задачи)."""
    vals = []
    for qid, tgt in targets.items():
        p = set(pred.get(qid, [])[:k])
        vals.append(len(p & tgt) / len(tgt))
    return float(np.mean(vals))


def recall_report(pred: dict, queries: pd.DataFrame, targets: dict, k: int = 50) -> str:
    """Recall в разрезе: все / с пустыми фильтрами / с фильтрами / взвешенно как в бенчмарке."""
    h = queries[queries.split == "hold"]
    empty = set(h.qid[h.search_infm_params_text == ""])
    t_empty = {q: t for q, t in targets.items() if q in empty}
    t_filt = {q: t for q, t in targets.items() if q not in empty}
    r_all = recall_at_k(pred, targets, k)
    r_e, r_f = recall_at_k(pred, t_empty, k), recall_at_k(pred, t_filt, k)
    # в бенчмарке 63% запросов без фильтров — перевзвешиваем под это
    r_w = 0.631 * r_e + 0.369 * r_f
    return f"R@{k}: all={r_all:.4f} empty={r_e:.4f} filters={r_f:.4f} bench-weighted={r_w:.4f}"
