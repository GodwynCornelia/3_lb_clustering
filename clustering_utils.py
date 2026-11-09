"""Численные процедуры для проверки кластеризации."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import contingency_matrix


def hopkins(X, sample_size=200, random_state=42, power=None):
    """H=sum(u**d)/(sum(u**d)+sum(w**d)); H≈0.5 для uniform.

    u: расстояния uniform-точек до выборки, w: до другого наблюдения.
    Степень d учитывает объём d-мерного шара; power=1 — диагностика
    часто используемой альтернативной конвенции. Нулевые соседи самого
    наблюдения исключаются запросом k=2. Дубликаты не удаляются молча.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 3 or not np.isfinite(X).all():
        raise ValueError("Нужна конечная двумерная матрица минимум из 3 строк")
    if np.any(np.ptp(X, axis=0) == 0):
        raise ValueError("Перед Hopkins исключите постоянные признаки")
    rng = np.random.default_rng(random_state)
    size = min(sample_size, len(X))
    indices = rng.choice(len(X), size=size, replace=False)
    uniform = rng.uniform(X.min(axis=0), X.max(axis=0), (size, X.shape[1]))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    u = nn.kneighbors(uniform, n_neighbors=1)[0][:, 0]
    w = nn.kneighbors(X[indices], n_neighbors=2)[0][:, 1]
    exponent = X.shape[1] if power is None else power
    scale = max(u.max(), w.max())
    if scale == 0:
        raise ValueError("Все расстояния равны нулю")
    a, b = np.sum((u / scale) ** exponent), np.sum((w / scale) ** exponent)
    return float(a / (a + b))


def align_labels(y_true, labels):
    """Венгерское сопоставление, максимизирующее число совпадений.

    Оптимальность относится к accuracy, а не непосредственно к macro Jaccard.
    Лишние кластеры получают метку -1, а не сливаются с известными классами.
    """
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    classes, clusters = np.unique(y_true), np.unique(labels)
    matrix = contingency_matrix(y_true, labels)
    rows, cols = linear_sum_assignment(-matrix)
    mapping = {clusters[c]: classes[r] for r, c in zip(rows, cols)}
    return np.array([mapping.get(c, -1) for c in labels]), mapping


def expected_rand_index(y_true, labels):
    """E[RI] при случайной перестановке меток с сохранением размеров групп.

    p и q — вероятности оказаться в одной группе для случайной пары без
    возвращения. При независимом назначении E[RI] = p*q + (1-p)*(1-q).
    Это точное ожидание для конечной выборки, а не асимптотика равных групп.
    """
    y_true, labels = np.asarray(y_true), np.asarray(labels)
    if y_true.ndim != 1 or labels.ndim != 1 or len(y_true) != len(labels) or len(labels) < 2:
        raise ValueError("Нужны два одномерных вектора одинаковой длины минимум 2")
    true_counts = np.unique(y_true, return_counts=True)[1].astype(float)
    cluster_counts = np.unique(labels, return_counts=True)[1].astype(float)
    ordered_pairs = len(labels) * (len(labels) - 1)
    p = np.sum(true_counts * (true_counts - 1)) / ordered_pairs
    q = np.sum(cluster_counts * (cluster_counts - 1)) / ordered_pairs
    return float(p * q + (1 - p) * (1 - q))
