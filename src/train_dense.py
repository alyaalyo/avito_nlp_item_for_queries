"""Шаг 3. Дообучение bi-encoder на парах «запрос -> выбранное объявление».

Базовая модель: cointegrated/rubert-tiny2 (open-source, 29M параметров,
русскоязычный BERT). Выбрана из-за скорости: на Apple M4 она кодирует
~1500 объявлений/с против ~300 у multilingual-e5-small, что позволяет и
дообучить её, и закодировать корпус за минуты.

Обучение — contrastive loss с негативами из батча (MultipleNegativesRankingLoss):
для каждого запроса правильное объявление должно быть ближе, чем объявления
других запросов из того же батча. Сэмплер NO_DUPLICATES не кладёт в один батч
одинаковые тексты (иначе «маникюр» из другой группы стал бы ложным негативом).

Используем только train-часть групп (holdout не трогаем, чтобы честно
оценивать качество).
"""
import argparse
import time

import numpy as np
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from config import SEED, WORK
from dense import item_text, load_model, query_text

MAX_PER_QUERY_TEXT = 30  # ограничиваем «головные» запросы (маникюр — 6.5 тыс. строк), чтобы не забивали обучение


def build_pairs(split_filter=("train",)):
    groups = pd.read_parquet(WORK / "groups.parquet")
    pairs = pd.read_parquet(WORK / "pairs.parquet")
    items = pd.read_parquet(WORK / "items.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
    g = groups[groups.split.isin(split_filter)]
    df = g.merge(pairs, on="gid")
    df["q"] = [query_text(a, b) for a, b in zip(df.search_query, df.search_infm_params_text)]
    df = df.drop_duplicates(["q", "item_id"])
    # не больше MAX_PER_QUERY_TEXT пар на один текст запроса
    df = df.sample(frac=1.0, random_state=SEED)
    df = df[df.groupby("search_query").cumcount() < MAX_PER_QUERY_TEXT]
    df = df.merge(items, on="item_id")
    df["p"] = [item_text(a, b, c) for a, b, c in
               zip(df.item_title_raw, df.item_infm_params_text, df.item_description_raw)]
    return df[["q", "p"]].sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cointegrated/rubert-tiny2")
    ap.add_argument("--out", default=str(WORK / "dense_tiny2"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=96)
    args = ap.parse_args()

    t = time.time()
    df = build_pairs()
    print(f"pairs: {len(df)}  ({time.time() - t:.0f}s)")
    print(df.head(3).to_string())

    model = load_model(args.base, args.max_len)
    loss = MultipleNegativesRankingLoss(model, scale=20.0)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(WORK / "dense_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs,
                                         train_dataset=Dataset.from_pandas(df, preserve_index=False), loss=loss)
    trainer.train()
    model.save(args.out)
    print(f"saved to {args.out}  total {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
