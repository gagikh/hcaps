"""Expression trees of the leaf products and the files generated from them.

Port of schwartz::tree_node_t (build / dump / dot) and of build_forward_pass() of the old
tst/generator.cpp:

    forward_pass_header(R) -> gen/depth_<R>_functions.cuh   (mult<T, node, cache>::left/right: the
                                                            operands of leaf product `node` read
                                                            straight from A and B)
    tree_dot(N, R)         -> docs/test.dot                 (every leaf product as a tree of
                                                            a+b / a-b nodes over leaf blocks)

Leaf product numbering is tree order (children of node p are 7p .. 7p+6).
"""

from __future__ import annotations

from dataclasses import dataclass

PLUS, MINUS, RMINUS, MULT = "+", "-", "r-", "*"  # OP_L_plus_R, OP_L_minus_R, OP_R_minus_L, OP_L_mult_R

NODE_STR = ("m1 = A11 * B11", "m2 = A12 * B21", "m3 = A21 * (B22 - B11)", "m4 = A22 * B22",
            "m5 = (A21 + A22) * (B21 + B22)", "m6 = (A22 - A12) * (B22 - B12)", "m7 = (A22 - A11) * B12")


@dataclass(eq=False)
class Node:
    """Leaf block (name "A"/"B", position y, x in elements) or an operation L op R."""
    op: str = MULT
    L: "Node | None" = None
    R: "Node | None" = None
    name: str = ""
    y: int = 0
    x: int = 0

    @property
    def leaf(self):
        return self.L is None or self.R is None

    def quadrant(self, i: int, j: int, M: int) -> "Node":
        """C_ij of this operand at block size M (i, j in 1..2)."""
        if self.leaf:
            return Node(name=self.name, y=self.y + (i - 1) * M, x=self.x + (j - 1) * M)
        return Node(self.op, self.L.quadrant(i, j, M), self.R.quadrant(i, j, M), name=self.name)


def build(N: int, R: int):
    """The 7^R leaf products (MULT nodes with operand trees), tree order."""
    nodes = [Node(MULT, Node(name="A"), Node(name="B"))]
    for level in range(1, R + 1):
        M = N >> level
        children = []
        for root in nodes:
            A, B = root.L, root.R
            q = lambda X, i, j: X.quadrant(i, j, M)  # noqa: E731
            children += [
                Node(MULT, q(A, 1, 1), q(B, 1, 1)),                                   # m1
                Node(MULT, q(A, 1, 2), q(B, 2, 1)),                                   # m2
                Node(MULT, q(A, 2, 1), Node(MINUS, q(B, 2, 2), q(B, 1, 1))),          # m3
                Node(MULT, q(A, 2, 2), q(B, 2, 2)),                                   # m4
                Node(MULT, Node(PLUS, q(A, 2, 1), q(A, 2, 2)), Node(PLUS, q(B, 2, 1), q(B, 2, 2))),    # m5
                Node(MULT, Node(MINUS, q(A, 2, 2), q(A, 1, 2)), Node(MINUS, q(B, 2, 2), q(B, 1, 2))),  # m6
                Node(MULT, Node(MINUS, q(A, 2, 2), q(A, 1, 1)), q(B, 1, 2)),          # m7
            ]
        nodes = children
    return nodes


# ------------------------------------------------------------------------------------------
# depth_<R>_functions.cuh
# ------------------------------------------------------------------------------------------

def _pos_str(N: int, M: int, pos: int, postfix: str) -> str:
    """Element offset pos as an expression of N: "y", "N/2+y", "3*N/4+y"."""
    cy, sy = N // M, pos // M
    while sy != 0 and sy % 2 == 0:
        sy //= 2
        cy //= 2
    if sy == 0:
        return postfix
    head = "N" if sy == 1 else f"{sy}*N"
    return head + (f"/{cy}" if cy != 1 else "") + "+" + postfix


def _dump(node: Node, i: int, N: int, M: int) -> str:
    if node.leaf:
        return (f"read<T, cache>(reinterpret_cast<const T*>({node.name} + ({_pos_str(N, M, node.y, 'y')}) * N + "
                f"{_pos_str(N, M, node.x, 'x')}))")
    a, b = _dump(node.L, i, N, M), _dump(node.R, i, N, M)
    if node.op == MULT:
        return (f"    // {NODE_STR[i % 7]}\n"
                "    template<class T, bool cache>\n"
                f"    struct mult<T, {i}, cache>\n"
                "    {\n"
                "        static __device__ T left(const half *__restrict__ A, unsigned int N, unsigned int y, unsigned int x) {\n"
                f"            return {a};\n"
                "        }\n"
                "        static __device__ T right(const half *__restrict__ B, unsigned int N, unsigned int y, unsigned int x) {\n"
                f"            return {b};\n"
                "        }\n"
                "    };\n")
    if node.op == PLUS:
        return f"({a} + {b})"
    if node.op == MINUS:
        return f"({a} - {b})"
    return f"({b} - {a})"


def forward_pass_header(R: int, N: int = 16384) -> str:
    """gen/depth_<R>_functions.cuh: per leaf product its left/right operand read from A and B.
    The text does not depend on N (offsets are written as fractions of N)."""
    lines = ["#pragma once\n", '#include "utils.cuh"\n\n', f"namespace depth_{R}_functions {{\n\n",
             "    template<class T, unsigned int node, bool cache>\n", "    struct mult {};\n\n"]
    M = N >> R
    for i, node in enumerate(build(N, R)):
        lines.append(_dump(node, i, N, M) + "\n")
    lines.append("    struct functor {\n")
    t, cache = "half8", "true"
    for side in ("left", "right"):
        lines.append("    // Define a function pointer type\n")
        lines.append(f"    static __device__ __forceinline__ {t} {side}(unsigned int node, const half *__restrict__ A, "
                     "unsigned int N, unsigned int y, unsigned int x) {\n")
        lines.append("        switch (node) {\n")
        for node in range(7 ** R):
            lines.append(f"    \tcase {node}: return mult<{t},{node}, {cache}>::{side}(A, N, y, x);\n")
        lines.append(f"    \tdefault: {{ {t} tmp = {{}}; return tmp; }}\n")
        lines.append("        }\n")
        lines.append("    }\n")
    lines.append("    };\n")
    lines.append(f"}} // namespace depth_{R}_functions\n")
    return "".join(lines)


# ------------------------------------------------------------------------------------------
# test.dot
# ------------------------------------------------------------------------------------------

def tree_dot(N: int = 8192, R: int = 3) -> str:
    """Every leaf product as a graph: leaf blocks A_<y>_<x> / B_<y>_<x> (element positions) feed
    a+b / a-b nodes that feed the product node mult_<i>. Identical block names are one node in
    Graphviz, so a block used by several products becomes a shared input."""
    counters = {PLUS: 0, MINUS: 0, RMINUS: 0, MULT: 0}
    prefix = {PLUS: ("plus_", "a+b"), MINUS: ("a_minus_b_", "a-b"), RMINUS: ("b_minus_a", "b-a"), MULT: ("mult_", "a*b")}
    names = {}
    out = ["digraph {\n"]

    def dot_name(node):
        if node.leaf or node.name:
            return f"{node.name}_{node.y}_{node.x}", node.name
        if id(node) not in names:
            names[id(node)] = prefix[node.op][0] + str(counters[node.op])
            counters[node.op] += 1
        return names[id(node)], prefix[node.op][1]

    def visit(node):
        if node.leaf:
            return
        n = dot_name(node)
        a = dot_name(node.L)
        b = dot_name(node.R)
        out.append(f'\t{a[0]} [label="{a[1]}"];\n')
        out.append(f'\t{b[0]} [label="{b[1]}"];\n')
        out.append(f'\t{n[0]} [label="{n[1]}"];\n')
        out.append(f"\t{a[0]}->{n[0]};\n")
        out.append(f"\t{b[0]}->{n[0]};\n")
        visit(node.L)
        visit(node.R)

    for root in build(N, R):
        visit(root)
    out.append("}\n")
    return "".join(out)
