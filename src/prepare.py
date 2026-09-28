"""Шаг 1. Подготовка данных и локальной валидации.

Что делаем:
1. Собираем единую таблицу объявлений `items`: корпус бенчмарка + объявления из
   train (они нужны как «правильные ответы» для валидационных запросов).
   Флаг `in_bench` помечает объявления из корпуса бенчмарка.
2. Схлопываем train в «поисковые группы»: одна группа = уникальная комбинация
   полей запроса (текст, локация, фильтры, категория, доставка), у группы есть
   список выбранных объявлений. Именно так выглядит строка бенчмарка.
3. Отбираем holdout-группы, имитирующие бенчмарк.

Как устроен бенчмарк (выяснено анализом, см. README):
* каждый текст запроса встречается в бенчмарке ровно один раз;
* 37.0% текстов бенчмарка встречаются в train;
* если брать случайный уникальный текст train, то с вероятностью 37.4% у него
  больше одной группы.
Совпадение почти точное, поэтому бенчмарк моделируем так: берём случайный
уникальный текст запроса, у него — одну случайную группу; её откладываем в
holdout, а остальные группы того же текста остаются в обучении.

Корпус для валидации = корпус бенчмарка + правильные ответы holdout-групп.
Распределение локаций корпуса бенчмарка совпадает с распределением объявлений
train (корреляция 0.985), т.е. корпус не подстроен под запросы бенчмарка, и такая
схема честно воспроизводит «сложность» поиска.
"""
import numpy as np
import pandas as pd

from config import (BENCH_ITEMS_PATH, BENCH_QUERIES_PATH, ITEM_COLS, QUERY_KEY,
                    SEED, TRAIN_PATH, WORK)

N_HOLDOUT = 12_000  # holdout-групп: половина на обучение ранкера, половина на оценку (и наоборот)


def to_float(df, cols):
    # цены и координаты лежат в parquet как decimal — приводим к float
    for c in cols:
        df[c] = df[c].astype("float64")
    return df


def main():
    rng = np.random.default_rng(SEED)
    train = pd.read_parquet(TRAIN_PATH)
    bench_items = pd.read_parquet(BENCH_ITEMS_PATH)
    bench_q = pd.read_parquet(BENCH_QUERIES_PATH)

    num_cols = ["item_price", "item_latitude", "item_longitude"]
    train = to_float(train, num_cols)
    bench_items = to_float(bench_items, num_cols)

    # ---------- объявления ----------
    bench_items["in_bench"] = True
    train_items = train[ITEM_COLS].drop_duplicates("item_id")
    train_items = train_items[~train_items.item_id.isin(bench_items.item_id)].copy()
    train_items["in_bench"] = False
    items = pd.concat([bench_items[ITEM_COLS + ["in_bench"]], train_items], ignore_index=True)
    items["item_description_raw"] = items["item_description_raw"].fillna("")
    items.to_parquet(WORK / "items.parquet", index=False)
    print(f"items: {len(items)} (bench {items.in_bench.sum()})")

    # ---------- поисковые группы train ----------
    train["gid"] = train.groupby(QUERY_KEY, sort=False).ngroup()
    pairs = train[["gid", "item_id"]].drop_duplicates()
    groups = train.drop_duplicates("gid")[["gid"] + QUERY_KEY].sort_values("gid").reset_index(drop=True)
    groups["n_items"] = groups.gid.map(pairs.groupby("gid").size())

    # ---------- holdout, имитирующий бенчмарк ----------
    texts = groups.search_query.unique()
    hold_texts = rng.choice(texts, size=N_HOLDOUT, replace=False)
    cand = groups[groups.search_query.isin(set(hold_texts))]
    # одна случайная группа на текст
    hold_gids = cand.sample(frac=1.0, random_state=SEED).drop_duplicates("search_query").gid.values
    groups["split"] = "train"
    groups.loc[groups.gid.isin(hold_gids), "split"] = "hold"
    # два фолда внутри holdout — для кросс-валидации ранкера
    hold = groups.split == "hold"
    groups.loc[hold, "fold"] = rng.integers(0, 2, size=hold.sum())
    groups["fold"] = groups["fold"].fillna(-1).astype(int)

    groups.to_parquet(WORK / "groups.parquet", index=False)
    pairs.to_parquet(WORK / "pairs.parquet", index=False)

    bench_q.to_parquet(WORK / "bench_queries.parquet", index=False)

    # ---------- проверка похожести holdout на бенчмарк ----------
    tr_texts = set(groups.loc[groups.split == "train", "search_query"])
    h = groups[hold]
    print(f"holdout groups: {len(h)}")
    print(f"  доля текстов holdout, встречающихся в train-части: {h.search_query.isin(tr_texts).mean():.3f} "
          f"(бенчмарк: {bench_q.search_query.isin(set(groups.search_query)).mean():.3f})")
    print(f"  доля пустых фильтров: {(h.search_infm_params_text == '').mean():.3f} "
          f"(бенчмарк: {(bench_q.search_infm_params_text == '').mean():.3f})")
    print(f"  среднее число правильных объявлений: {h.n_items.mean():.2f}")
    hold_items = pairs[pairs.gid.isin(hold_gids)].item_id
    print(f"  доля правильных ответов holdout, лежащих в корпусе бенчмарка: {hold_items.isin(bench_items.item_id).mean():.3f}")


if __name__ == "__main__":
    main()
