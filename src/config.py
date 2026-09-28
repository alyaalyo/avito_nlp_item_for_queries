"""Общие пути и константы решения."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Папка с исходными parquet-файлами. Можно переопределить переменной окружения
# AVITO_DATA_DIR, чтобы запустить решение на другой машине.
DATA_DIR = Path(os.environ.get("AVITO_DATA_DIR", Path.home() / "Downloads" / "dataset"))
TRAIN_PATH = DATA_DIR / "train.parquet"
BENCH_ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
BENCH_QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"

# Все промежуточные артефакты (кэши, индексы, модели) складываем сюда.
# AVITO_WORK_DIR позволяет прогнать пайплайн в отдельной «чистой» папке.
WORK = Path(os.environ.get("AVITO_WORK_DIR", ROOT / "work"))
WORK.mkdir(exist_ok=True)

# GitHub Release с обученными моделями — для точного воспроизведения answer.csv
# без повторного обучения (см. download_artifacts.py и README).
RELEASE_URL = "https://github.com/alyaalyo/avito_nlp_item_for_queries/releases/download/v1.0"

SEED = 42
TOP_K = 50  # сколько кандидатов отдаём на запрос

# Ключ «поискового запроса»: одна строка бенчмарка = одна уникальная комбинация
# этих полей. В train одна такая комбинация может встречаться несколько раз
# (разные пользователи выбрали разные объявления) — объединяем их в группу.
QUERY_KEY = [
    "search_query",
    "search_location_id",
    "search_infm_params_text",
    "search_category",
    "search_is_delivery_search",
]

ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]
