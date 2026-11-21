"""The alternative basis transforms of src/alternative_basis.cuh at leaf-block granularity.

alternative_basis_transform_kernel() runs levels l = 1..R; at level l (M = N >> l) every 2M x 2M
block  a b / c d  of the matrix is transformed element-wise, in place:

    encode A (encoder, transform = false):  d <- b + d - c      A22 <- A12 + A22 - A21
    encode B (encoder, transform = true):   d <- c + d - b      B22 <- B21 + B22 - B12
    decode C (decoder, on the root's C):    b <- b - d,  c <- d - c

(a is never touched). Loop k of `loops` handles the rows k, k + loops, ... of every block, so the
row loops split the matrix into interleaved row classes, never into separate blocks.

Coordinates are leaf blocks: the matrix is C x C blocks of (N >> R)^2, C = 2^R; block (i, j) is
row i, column j. A block's content is a linear combination of the original blocks.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

OPERATIONS = ("encode A", "encode B", "decode C")


def block_name(prefix: str, i: int, j: int, R: int) -> str:
    """Quadrant path of leaf block (i, j): A12.21 = quadrant 12, inside it quadrant 21"""
    parts, n = [], 1 << R
    while n > 1:
        n //= 2
        parts.append(f"{i // n + 1}{j // n + 1}")
        i, j = i % n, j % n
    return prefix + ".".join(parts)


@dataclass
class Level:
    """One level of a transform: per leaf block its role ("" untouched, "+" / "-" read with that
    sign, "d" read and written (encoder), "w" written (decoder: b and c), "r" read only (decoder: d))."""
    level: int
    size: int          # M = block size of b, c, d in leaf blocks
    roles: dict        # (i, j) -> role
    formula: str


def levels(R: int, operation: str) -> list:
    if operation not in OPERATIONS:
        raise ValueError(f"operation must be one of {OPERATIONS}")
    C = 1 << R
    out = []
    for l in range(1, R + 1):
        m = C >> l  # b, c, d are m x m leaf blocks
        roles = {(i, j): "" for i in range(C) for j in range(C)}
        for by in range(0, C, 2 * m):
            for bx in range(0, C, 2 * m):
                for u in range(m):
                    for v in range(m):
                        b, c, d = (by + u, bx + m + v), (by + m + u, bx + v), (by + m + u, bx + m + v)
                        if operation == "encode A":
                            roles[b], roles[c], roles[d] = "+", "-", "d"
                        elif operation == "encode B":
                            roles[b], roles[c], roles[d] = "-", "+", "d"
                        else:
                            roles[b], roles[c], roles[d] = "w", "w", "r"
        formula = {"encode A": "d ← b + d − c   (X22 ← X12 + X22 − X21)",
                   "encode B": "d ← c + d − b   (X22 ← X21 + X22 − X12)",
                   "decode C": "b ← b − d,  c ← d − c   (X12 ← X12 − X22, X21 ← X22 − X21)"}[operation]
        out.append(Level(l, m, roles, formula))
    return out


def _combine(*terms) -> Counter:
    """sum of sign * combination (Counter's + would drop negative coefficients)"""
    out = Counter()
    for sign, combination in terms:
        for block, coef in combination.items():
            out[block] += sign * coef
    return Counter({b: x for b, x in out.items() if x})


def expressions(R: int, operation: str):
    """states[k] = {(i, j): Counter{(i0, j0): coefficient}} after k levels (states[0]: the input)"""
    C = 1 << R
    state = {(i, j): Counter({(i, j): 1}) for i in range(C) for j in range(C)}
    states = [state]
    for l in range(1, R + 1):
        m = C >> l
        new = {k: Counter(v) for k, v in state.items()}
        for by in range(0, C, 2 * m):
            for bx in range(0, C, 2 * m):
                for u in range(m):
                    for v in range(m):
                        b, c, d = (by + u, bx + m + v), (by + m + u, bx + v), (by + m + u, bx + m + v)
                        if operation == "encode A":
                            new[d] = _combine((1, state[b]), (1, state[d]), (-1, state[c]))
                        elif operation == "encode B":
                            new[d] = _combine((1, state[c]), (1, state[d]), (-1, state[b]))
                        else:
                            new[b] = _combine((1, state[b]), (-1, state[d]))
                            new[c] = _combine((1, state[d]), (-1, state[c]))
        state = new
        states.append(state)
    return states


def expression_text(terms: Counter, prefix: str, R: int) -> str:
    """'A12 + A22 - A21': positive terms first, then the negative ones, each in block order"""
    if not terms:
        return "0"
    items = sorted(terms.items(), key=lambda t: (t[1] < 0, t[0]))
    text = ""
    for (i, j), coef in items:
        name = block_name(prefix, i, j, R)
        mag = abs(coef)
        term = name if mag == 1 else f"{mag}·{name}"
        if not text:
            text = term if coef > 0 else "−" + term
        else:
            text += (" + " if coef > 0 else " − ") + term
    return text


def bytes_per_level(N: int, R: int, operation: str):
    """(read, written) bytes of one level (every level moves the same amount)"""
    half_area = (N // 2) * (N // 2) * 2  # one quarter of a matrix in bytes
    if operation == "decode C":
        return 3 * half_area, 2 * half_area
    return 3 * half_area, half_area
