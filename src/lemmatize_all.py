"""Шаг 2. Лемматизация всех текстов (объявления + запросы), результат кэшируется.

Для объявлений готовим три поля:
* title_lem  — заголовок;
* params_lem — параметры без шаблонного шума (вид/тип услуги, конкретные услуги,
  адрес «Место оказания услуг» — там бывают районы и метро из запросов);
* desc_lem   — первые 3000 символов описания (дальше обычно контакты и «вода»).
Для запросов: текст запроса и текст поисковых фильтров.
"""
import time

import pandas as pd

from config import WORK
from text import clean_item_params, lemmatize

DESC_CHARS = 3000


def main():
    t = time.time()
    items = pd.read_parquet(WORK / "items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    out = pd.DataFrame({"item_id": items.item_id})
    out["title_lem"] = [lemmatize(s) for s in items.item_title_raw]
    print(f"titles {time.time() - t:.0f}s")
    out["params_lem"] = [lemmatize(clean_item_params(s)) for s in items.item_infm_params_text]
    print(f"params {time.time() - t:.0f}s")
    out["desc_lem"] = [lemmatize(s[:DESC_CHARS]) for s in items.item_description_raw]
    print(f"descriptions {time.time() - t:.0f}s")
    out.to_parquet(WORK / "items_lem.parquet", index=False)

    # все уникальные тексты запросов и фильтров из train и бенчмарка
    groups = pd.read_parquet(WORK / "groups.parquet")
    bq = pd.read_parquet(WORK / "bench_queries.parquet")
    qtexts = pd.Series(pd.concat([groups.search_query, bq.search_query]).unique())
    ptexts = pd.Series(pd.concat([groups.search_infm_params_text, bq.search_infm_params_text]).unique())
    pd.DataFrame({"text": qtexts, "lem": [lemmatize(s) for s in qtexts]}).to_parquet(WORK / "query_lem.parquet", index=False)
    pd.DataFrame({"text": ptexts, "lem": [lemmatize(s) for s in ptexts]}).to_parquet(WORK / "sparams_lem.parquet", index=False)
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
