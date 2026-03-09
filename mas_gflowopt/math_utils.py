from __future__ import annotations

import math
from typing import Iterable, List

Vector = List[float]


def zeros(dim: int) -> Vector:
    return [0.0] * dim


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scale(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def cosine(a: Vector, b: Vector) -> float:
    na = norm(a)
    nb = norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def l2_sq(a: Vector, b: Vector) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def mean(vectors: Iterable[Vector]) -> Vector:
    vectors = list(vectors)
    if not vectors:
        return []
    dim = len(vectors[0])
    out = zeros(dim)
    for vec in vectors:
        out = add(out, vec)
    return [v / len(vectors) for v in out]


def normalize(a: Vector) -> Vector:
    n = norm(a)
    if n == 0.0:
        return a[:]
    return [x / n for x in a]
