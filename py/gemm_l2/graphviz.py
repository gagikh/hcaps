"""Render DOT text with the Graphviz command line tools (https://graphviz.org/download/).

Windows: install Graphviz and put its bin folder on PATH (the installer offers it), or set
GRAPHVIZ_DOT to the full path of dot.exe.
"""

from __future__ import annotations

import os
import shutil
import subprocess

ENGINES = ("dot", "neato", "fdp", "sfdp", "circo", "twopi", "osage", "patchwork")

_WINDOWS_GUESSES = (
    r"C:\Program Files\Graphviz\bin\dot.exe",
    r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
)


def find_dot() -> str | None:
    """Path of the dot executable, or None."""
    path = os.environ.get("GRAPHVIZ_DOT")
    if path and os.path.isfile(path):
        return path
    path = shutil.which("dot")
    if path:
        return path
    for guess in _WINDOWS_GUESSES:
        if os.path.isfile(guess):
            return guess
    return None


def render(dot_text: str, engine: str = "dot", fmt: str = "svg", timeout: float = 120) -> bytes:
    """Layout with the given engine and render to fmt (svg, png, pdf, ...). Raises RuntimeError
    when Graphviz is missing or fails."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine}; one of {', '.join(ENGINES)}")
    dot = find_dot()
    if not dot:
        raise RuntimeError("Graphviz 'dot' not found: install Graphviz and add its bin folder to PATH, "
                           "or set GRAPHVIZ_DOT to dot.exe")
    args = [dot, f"-K{engine}", f"-T{fmt}"]
    if engine == "sfdp":
        # sfdp removes overlaps with "prism" by default, which needs the GTS triangulation library;
        # the Windows builds have none ("not built with triangulation library"). Scaling works everywhere.
        args.append("-Goverlap=scale")
    result = subprocess.run(args, input=dot_text.encode(),
                            capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip() or f"dot exited with {result.returncode}")
    return result.stdout
