"""Нейросетевой (dense) поиск: bi-encoder, запрос и объявление -> вектор.

Лексический BM25 не видит синонимов и перефразировок («выкосить траву» vs
«покос травы», «обзвон по базе» vs «оператор колл-центра»). Bi-encoder
кодирует запрос и объявление в векторы одного пространства; релевантность =
косинусная близость. Берём open-source модель с HuggingFace (веса скачиваются
один раз и дальше работают локально) и дообучаем на парах «запрос -> выбранное
объявление» из train.
"""
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from text import clean_item_params

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Модели семейства e5/USER обучены с префиксами «query: » и «passage: ».
Q_PREFIX, P_PREFIX = "query: ", "passage: "


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def item_text(title: str, params: str, desc: str) -> str:
    """Текст объявления для энкодера: заголовок, суть параметров, начало описания.

    Длину всё равно режет токенизатор (max_seq_length), поэтому самое важное —
    в начале: заголовок, затем параметры (вид/тип услуги), затем описание.
    """
    return P_PREFIX + f"{_squash(title)}. {_squash(clean_item_params(params))[:300]}. {_squash(desc)[:400]}"


def query_text(query: str, filters: str) -> str:
    """Текст запроса: сам запрос + поисковые фильтры (если заданы)."""
    filters = _squash(filters)
    return Q_PREFIX + (f"{query} | {filters}" if filters else query)


def load_model(path_or_name: str, max_len: int = 128) -> SentenceTransformer:
    m = SentenceTransformer(path_or_name, device=DEVICE)
    m.max_seq_length = max_len
    return m


def encode(model: SentenceTransformer, texts, batch_size: int = 256, fp16: bool = True) -> np.ndarray:
    """Нормированные эмбеддинги (float16 для экономии памяти)."""
    # сортируем по длине: батчи с похожей длиной считаются заметно быстрее
    order = np.argsort([len(t) for t in texts])
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=fp16 and DEVICE != "cpu"):
        emb = model.encode([texts[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float16)
