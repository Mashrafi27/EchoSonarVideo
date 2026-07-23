import random
from collections import Counter


def class_weights(labels: list) -> dict:
    counts = Counter(labels)
    n = len(labels)
    k = len(counts)
    # inverse frequency, normalized so mean weight over the list == 1.0
    raw = {c: n / (k * cnt) for c, cnt in counts.items()}
    return raw


def resample_indices(labels: list, n: int, seed: int) -> list:
    w = class_weights(labels)
    weights = [w[lab] for lab in labels]
    rng = random.Random(seed)
    return rng.choices(range(len(labels)), weights=weights, k=n)
