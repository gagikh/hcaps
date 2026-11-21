"""Strassen-Winograd recursion tree (alternative basis) at leaf-block granularity.

Port of schwartz::tree_node_t::build() / biparted() (src/tree.cpp).

    S1 = A21 + A22   S2 = A22 - A12   S3 = A22 - A11
    S4 = B22 - B11   S5 = B21 + B22   S6 = B22 - B12

    m1 = A11 * B11   m2 = A12 * B21   m3 = A21 * S4   m4 = A22 * B22
    m5 = S1  * S5    m6 = S2  * S6    m7 = S3  * B12

Leaf multiplications are numbered in tree order: level by level, the 7 children of node p are
7p .. 7p+6 (the order BFS::prepare_mult_buffers() builds the batched GEMM in). A leaf block of A
(or B) at block row y, column x of the C x C grid (C = 2^R) has id x + C * y; B ids are offset
by C * C.
"""

from __future__ import annotations

from dataclasses import dataclass

# Operand of one child as signed terms of the parent's quadrants: (row, column, sign), 0-based.
A_TERMS = (
    ((0, 0, +1),),                 # m1: A11
    ((0, 1, +1),),                 # m2: A12
    ((1, 0, +1),),                 # m3: A21
    ((1, 1, +1),),                 # m4: A22
    ((1, 0, +1), (1, 1, +1)),      # m5: S1 = A21 + A22
    ((1, 1, +1), (0, 1, -1)),      # m6: S2 = A22 - A12
    ((1, 1, +1), (0, 0, -1)),      # m7: S3 = A22 - A11
)
B_TERMS = (
    ((0, 0, +1),),                 # m1: B11
    ((1, 0, +1),),                 # m2: B21
    ((1, 1, +1), (0, 0, -1)),      # m3: S4 = B22 - B11
    ((1, 1, +1),),                 # m4: B22
    ((1, 0, +1), (1, 1, +1)),      # m5: S5 = B21 + B22
    ((1, 1, +1), (0, 1, -1)),      # m6: S6 = B22 - B12
    ((0, 1, +1),),                 # m7: B12
)

MULTS = 7


def mult_count(R: int) -> int:
    return MULTS ** R


@dataclass(frozen=True)
class Operand:
    """Signed sum of leaf blocks: ((y, x, sign), ...) in leaf-block units at the current size."""
    terms: tuple

    def quadrant(self, i: int, j: int, half: int) -> "Operand":
        return Operand(tuple((y + i * half, x + j * half, s) for (y, x, s) in self.terms))


def leaf_operands(R: int):
    """[(A operand, B operand)] of the 7^R leaf multiplications in tree order.

    Operands are lists of (row, column, sign) of leaf blocks in the C x C grid (C = 2^R).
    Cancelling terms are kept (the tree never produces any)."""
    C = 1 << R
    nodes = [(((0, 0, +1),), ((0, 0, +1),))]  # root: A * B, blocks of size C
    size = C
    for _ in range(R):
        half = size // 2
        children = []
        for a, b in nodes:
            for k in range(MULTS):
                ca = tuple((y + i * half, x + j * half, s * t)
                           for (i, j, t) in A_TERMS[k] for (y, x, s) in a)
                cb = tuple((y + i * half, x + j * half, s * t)
                           for (i, j, t) in B_TERMS[k] for (y, x, s) in b)
                children.append((ca, cb))
        nodes = children
        size = half
    return nodes


def biparted(R: int):
    """Leaf multiplication -> leaf blocks it reads (A ids ascending, then B ids + C*C ascending).

    Same graph as schwartz::tree_node_t::biparted(); N does not matter, only R."""
    C = 1 << R
    graph = []
    for a, b in leaf_operands(R):
        ids_a = sorted({x + C * y for (y, x, _) in a})
        ids_b = sorted({C * C + x + C * y for (y, x, _) in b})
        graph.append(ids_a + ids_b)
    return graph


def coefficients(R: int):
    """Per leaf multiplication: ({A block id: coefficient}, {B block id: coefficient}), B ids
    offset by C * C. Coefficients are the summed signs (0 = cancelled)."""
    C = 1 << R
    result = []
    for a, b in leaf_operands(R):
        ca, cb = {}, {}
        for (y, x, s) in a:
            ca[x + C * y] = ca.get(x + C * y, 0) + s
        for (y, x, s) in b:
            k = C * C + x + C * y
            cb[k] = cb.get(k, 0) + s
        result.append((ca, cb))
    return result
