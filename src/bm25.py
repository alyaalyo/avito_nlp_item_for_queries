"""BM25 на разреженных матрицах (scipy).

Для каждого поля объявления строим матрицу весов W (документы × словарь), где
W[d, t] = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len_d / avg_len)).
Скор запроса = сумма весов его слов = Q @ W.T, где Q — бинарная матрица запросов.
Это позволяет считать BM25 сразу для тысячи запросов одним матричным умножением.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


def _split(s):
    return s.split()


def build_vocabulary(*fields) -> dict:
    """Общий словарь для нескольких полей: тогда матрица запроса одна на все поля,
    а покрытие слов запроса можно считать по объединению полей."""
    vocab = {}
    for docs in fields:
        for d in docs:
            for t in d.split():
                if t not in vocab:
                    vocab[t] = len(vocab)
    return vocab


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, vocabulary: dict | None = None):
        self.k1, self.b = k1, b
        self.cv = CountVectorizer(analyzer="word", tokenizer=_split, token_pattern=None, lowercase=False,
                                  vocabulary=vocabulary, dtype=np.float32)

    def fit(self, docs):
        tf = self.cv.fit_transform(docs).tocsr().astype(np.float32)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avg = dl.mean() if dl.mean() > 0 else 1.0
        # нормировка tf построчно (по длине документа)
        denom_row = self.k1 * (1 - self.b + self.b * dl / avg)
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + denom_row[tf.row]) * self.idf[tf.col]
        self.W = sp.csr_matrix((data.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.WT = self.W.T.tocsr()
        # бинарная матрица «слово есть в поле» — для признака покрытия запроса
        self.BT = self.WT.copy()
        self.BT.data[:] = 1.0
        return self

    def query_matrix(self, queries):
        q = self.cv.transform(queries).tocsr()
        q.data[:] = 1.0  # повтор слова в запросе не усиливает
        return q.astype(np.float32)

    def scores(self, qmat):
        """Плотная матрица скорингов (n_queries × n_docs)."""
        return (qmat @ self.WT).toarray()

    def max_score(self, qmat):
        """Максимально возможный скор запроса (все слова совпали) — для нормировки."""
        return np.asarray(qmat.multiply(self.idf[None, :] * (self.k1 + 1)).sum(1)).ravel()
