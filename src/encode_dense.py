"""Шаг 5. Кодирование корпуса и запросов дообученным bi-encoder'ом.

Результат — нормированные эмбеддинги (float16) в порядке load_corpus() и
load_queries(); косинусная близость = скалярное произведение.
"""
import time

import numpy as np
import pandas as pd

from common import load_corpus, load_queries
from config import WORK
from dense import encode, item_text, load_model, query_text


def main(model_dir=WORK / "dense_tiny2"):
    t = time.time()
    q = load_queries()
    corpus = load_corpus(q)
    model = load_model(str(model_dir), max_len=96)
    it_texts = [item_text(a, b, c) for a, b, c in
                zip(corpus.item_title_raw, corpus.item_infm_params_text, corpus.item_description_raw)]
    E_items = encode(model, it_texts, batch_size=256)
    print(f"items {E_items.shape} {time.time() - t:.0f}s")
    q_texts = [query_text(a, b) for a, b in zip(q.search_query, q.search_infm_params_text)]
    E_q = encode(model, q_texts, batch_size=512)
    np.save(WORK / "emb_items.npy", E_items)
    np.save(WORK / "emb_queries.npy", E_q)
    pd.Series(corpus.item_id).to_frame().to_parquet(WORK / "emb_items_ids.parquet", index=False)
    print(f"done {time.time() - t:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(WORK / "dense_tiny2"))
    main(ap.parse_args().model)
