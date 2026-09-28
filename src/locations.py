"""Модель совместимости локаций запроса и объявления.

Факты из train:
* 83% выбранных объявлений лежат в той же локации, что и поиск;
* если локация поиска — город, где есть объявления, совпадение 93%;
* часть локаций поиска — это регионы («Москва и область» и т.п.), где объявлений
  с таким location_id нет вовсе; выбранные объявления там распределены по
  городам региона (например, 61% — сам областной центр, дальше по убыванию).

Поэтому оцениваем P(локация объявления | локация поиска) по парам train-части
(holdout не используем!), а для редких пар добавляем геососедство: центры
локаций считаем по координатам объявлений.
"""
import numpy as np
import pandas as pd

from config import WORK


class LocationModel:
    def __init__(self, train_pairs: pd.DataFrame, items: pd.DataFrame):
        """train_pairs: колонки search_location_id, item_location_id (только train-часть)."""
        cnt = train_pairs.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        tot = cnt.groupby("search_location_id").n.sum()
        cnt["p"] = cnt.n / cnt.search_location_id.map(tot)
        self.pair_p = {(a, b): p for a, b, p in zip(cnt.search_location_id, cnt.item_location_id, cnt.p)}
        self.pair_n = {(a, b): n for a, b, n in zip(cnt.search_location_id, cnt.item_location_id, cnt.n)}
        self.search_total = tot.to_dict()
        # доля «чужих» локаций для каждой локации поиска (насколько регион «размыт»)
        same = cnt[cnt.search_location_id == cnt.item_location_id].set_index("search_location_id").p
        self.p_same = same.to_dict()

        # центры локаций по координатам объявлений (медиана устойчивее к выбросам)
        it = items[["item_location_id", "item_latitude", "item_longitude"]].dropna()
        cen = it.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        self.centers = {k: (a, b) for k, a, b in zip(cen.index, cen.item_latitude, cen.item_longitude)}
        # для локаций поиска, которые сами не являются локациями объявлений (регионы),
        # центр = взвешенный центр выбранных там объявлений
        for sl, g in cnt.groupby("search_location_id"):
            if sl in self.centers:
                continue
            xy = [(self.centers[il], p) for il, p in zip(g.item_location_id, g.p) if il in self.centers]
            if xy:
                w = np.array([p for _, p in xy])
                c = np.array([c for c, _ in xy])
                self.centers[sl] = tuple((c * w[:, None]).sum(0) / w.sum())

    def distance_km(self, sl, lat, lon):
        """Расстояние от центра локации поиска до точек (векторно)."""
        c = self.centers.get(sl)
        if c is None:
            return np.full(len(lat), np.nan, dtype=np.float32)
        la1, lo1 = np.radians(c[0]), np.radians(c[1])
        la2, lo2 = np.radians(lat), np.radians(lon)
        h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
        return (2 * 6371 * np.arcsin(np.sqrt(h))).astype(np.float32)


def load_train_loc_pairs():
    groups = pd.read_parquet(WORK / "groups.parquet")
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    items = pd.read_parquet(WORK / "items.parquet", columns=["item_id", "item_location_id"])
    g = groups[groups.split == "train"][["gid", "search_location_id"]]
    m = g.merge(pairs, on="gid").merge(items, on="item_id")
    return m[["search_location_id", "item_location_id"]]
