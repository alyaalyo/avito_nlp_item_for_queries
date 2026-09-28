#!/usr/bin/env bash
# Точное воспроизведение отправленного answer.csv (для проверяющего).
# Берёт обученные модели из GitHub Release, остальное пересчитывает кодом.
# ~10 минут на CPU, GPU не нужен.
#   bash reproduce.sh                      # скачать артефакты с GitHub
#   bash reproduce.sh --from_dir <папка>   # взять заранее скачанные файлы релиза
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PYTHON:-../.venv/bin/python}

$PY prepare.py                          # 1. объявления, поисковые группы, holdout (seed=42)
$PY lemmatize_all.py                    # 2. лемматизация
$PY knn_queries.py                      # 4. похожие запросы из истории
$PY download_artifacts.py "$@"          # эмбеддинги + ранкер из релиза (с проверкой SHA-256)
$PY candidates.py --split bench         # 6. кандидаты и признаки для запросов бенчмарка
$PY rerank.py --predict_only --cands cands_bench.parquet   # 7. готовый LightGBM -> answer.csv
$PY check_answer.py                     # проверка формата
