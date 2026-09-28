#!/usr/bin/env bash
# Полный пайплайн: от сырых parquet до answer.csv.
# Время на Apple M4 (16 ГБ): ~70 минут, большая часть — дообучение энкодера (~30 мин) и ранкер (~13 мин).
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1

$PY prepare.py          # 1. объявления, поисковые группы, holdout
$PY lemmatize_all.py    # 2. лемматизация
$PY train_dense.py      # 3. дообучение bi-encoder (rubert-tiny2)
$PY knn_queries.py      # 4. похожие запросы из истории
$PY encode_dense.py     # 5. эмбеддинги корпуса и запросов
$PY candidates.py       # 6. кандидаты + признаки
$PY rerank.py           # 7. LightGBM, кросс-валидация, answer.csv
$PY check_answer.py     # проверка формата
