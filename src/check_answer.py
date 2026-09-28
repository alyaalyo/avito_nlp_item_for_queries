"""Проверка answer.csv на все требования формата из условия задачи."""
import re
import sys

import pandas as pd

from config import BENCH_ITEMS_PATH, BENCH_QUERIES_PATH, ROOT, TOP_K


def check(path):
    # читаем строго как строки, чтобы ничего не превратилось в числа
    ans = pd.read_csv(path, dtype=str, keep_default_na=False)
    bq = pd.read_parquet(BENCH_QUERIES_PATH, columns=["query_id"])
    items = set(pd.read_parquet(BENCH_ITEMS_PATH, columns=["item_id"]).item_id)
    errors = []
    if list(ans.columns) != ["query_id", "answer"]:
        errors.append(f"колонки {list(ans.columns)} != ['query_id', 'answer']")
    if ans.query_id.duplicated().any():
        errors.append("повторяющиеся query_id")
    if set(ans.query_id) != set(bq.query_id):
        errors.append(f"набор query_id не совпадает: лишних {len(set(ans.query_id) - set(bq.query_id))}, "
                      f"недостающих {len(set(bq.query_id) - set(ans.query_id))}")
    if not ans.query_id.str.len().eq(16).all():
        errors.append("query_id не из 16 символов")
    hex16 = re.compile(r"^[0-9a-f]{16}$")
    n_ids = []
    for qid, a in zip(ans.query_id, ans.answer):
        lst = a.split(" ") if a else []
        n_ids.append(len(lst))
        if len(lst) > TOP_K:
            errors.append(f"{qid}: больше {TOP_K} item_id")
        if len(set(lst)) != len(lst):
            errors.append(f"{qid}: повторы item_id")
        bad = [x for x in lst if not hex16.match(x) or x not in items]
        if bad:
            errors.append(f"{qid}: item_id не из корпуса: {bad[:3]}")
    n_ids = pd.Series(n_ids)
    print(f"строк {len(ans)}, item_id на запрос: min {n_ids.min()}, mean {n_ids.mean():.1f}, max {n_ids.max()}")
    if errors:
        print("ОШИБКИ:\n  " + "\n  ".join(errors[:20]))
        return False
    print("формат OK")
    return True


if __name__ == "__main__":
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else ROOT / "answer.csv") else 1)
