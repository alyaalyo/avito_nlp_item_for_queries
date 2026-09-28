"""Сравнение двух файлов answer.csv (например, своего прогона и эталона из релиза).

Печатает долю запросов с полностью совпавшим набором item_id и среднее
пересечение топ-50. Порядок объявлений внутри строки на метрику не влияет,
поэтому сравниваем множества.
"""
import sys

import pandas as pd


def main(a_path, b_path):
    a = pd.read_csv(a_path, dtype=str, keep_default_na=False).set_index("query_id").answer
    b = pd.read_csv(b_path, dtype=str, keep_default_na=False).set_index("query_id").answer
    assert set(a.index) == set(b.index), "разные наборы query_id"
    same, overlap = [], []
    for qid in a.index:
        x, y = set(a[qid].split()), set(b[qid].split())
        same.append(x == y)
        overlap.append(len(x & y) / max(len(x), len(y), 1))  # доля общих объявлений в топ-50
    print(f"запросов: {len(same)}; полностью совпали: {sum(same)} ({sum(same) / len(same):.2%}); "
          f"среднее пересечение: {sum(overlap) / len(overlap):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
