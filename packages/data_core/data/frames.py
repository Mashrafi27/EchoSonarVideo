def evenly_spaced(n_total: int, k: int) -> list:
    if n_total <= 0 or k <= 0:
        return []
    if k >= n_total:
        return list(range(n_total))
    if k == 1:
        return [0]
    step = (n_total - 1) / (k - 1)
    return sorted({round(i * step) for i in range(k)})


def midframe(n_total: int) -> int:
    return n_total // 2
