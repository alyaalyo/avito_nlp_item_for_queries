"""Загрузка обученных артефактов из GitHub Release для точного воспроизведения.

Зачем: дообучение энкодера на GPU (MPS/CUDA) не детерминировано бит-в-бит,
а кодирование на разном железе даёт отличия в последних знаках эмбеддингов.
Поэтому для воспроизведения ровно того answer.csv, что был отправлен, в релиз
выложены:
  * emb_items.npy, emb_queries.npy, emb_items_ids.parquet — эмбеддинги корпуса
    и запросов, посчитанные дообученным энкодером;
  * ranker.txt — обученная модель LightGBM;
  * dense_tiny2.zip — сам дообученный энкодер (для проверки/перекодирования,
    для точного воспроизведения не обязателен).
Остальные шаги (подготовка данных, лемматизация, kNN, кандидаты) детерминированы
и пересчитываются кодом.

Контрольные суммы SHA-256 проверяются после скачивания. Если сеть недоступна,
файлы можно скачать вручную со страницы релиза и положить в work/ (или указать
папку с ними через --from_dir).
"""
import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from config import RELEASE_URL, WORK

SHA256 = {
    "emb_items.npy": "f1a3c1347e715d8bac571372254afaf61499895fda656b60f10bab3e4ebe2fab",
    "emb_queries.npy": "5b8af258b691d45de4c9a4545ebc7329812a4a494dd09c69d4e9466fbe0b5a0f",
    "emb_items_ids.parquet": "cf75223eab2ddf70029c4aa0d73d85be8cd0a24961afba643e617890a3ef75ee",
    "ranker.txt": "993c83ac7cbc6c00bcbca33e12289bcd58b3e6622ac789e341c6aefac9494dc9",
    "dense_tiny2.zip": "dc64ffc163a6b13aea50ffe3ee48ffdd291918c7c68b0cca3fe05042ea3dcc97",
}
REQUIRED = ["emb_items.npy", "emb_queries.npy", "emb_items_ids.parquet", "ranker.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, from_dir: Path | None):
    dst = WORK / name
    if dst.exists() and sha256(dst) == SHA256[name]:
        print(f"{name}: уже есть, контрольная сумма совпадает")
        return
    if from_dir is not None:
        shutil.copy(from_dir / name, dst)
    else:
        url = f"{RELEASE_URL}/{name}"
        print(f"{name}: скачиваю {url}")
        urllib.request.urlretrieve(url, dst)
    got = sha256(dst)
    if got != SHA256[name]:
        raise RuntimeError(f"{name}: контрольная сумма не совпадает ({got})")
    print(f"{name}: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_encoder", action="store_true", help="скачать и распаковать также сам энкодер")
    ap.add_argument("--from_dir", type=Path, default=None, help="взять файлы из локальной папки вместо GitHub")
    args = ap.parse_args()
    names = REQUIRED + (["dense_tiny2.zip"] if args.with_encoder else [])
    for n in names:
        fetch(n, args.from_dir)
    if args.with_encoder:
        with zipfile.ZipFile(WORK / "dense_tiny2.zip") as z:
            z.extractall(WORK)
        print(f"энкодер распакован в {WORK / 'dense_tiny2'}")


if __name__ == "__main__":
    main()
