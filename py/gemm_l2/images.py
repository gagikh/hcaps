"""Image helpers (BGR numpy arrays): parameter validation, title and legend bands.

The dependency matrix and L2 schedule images that lived here moved to archive/cleanup_2026-10/py
with the L2 scheduler; the L2 simulation is gemm_l2/l2_sim.py.
"""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
MB = 1024.0 * 1024.0


def validate(N: int, R: int, l2_bytes: int = 0) -> str:
    """Empty if (N, R, L2 size) can be drawn, otherwise the reason. l2_bytes = 0 skips the L2 checks."""
    if R < 1 or R > 5:
        return "R must be in [1, 5] (R = 5 already has 16807 leaf multiplications)"
    if N < 2 or N % (1 << R):
        return f"N must be divisible by 2^R = {1 << R}"
    if l2_bytes:
        M = N >> R
        if l2_bytes < M * M * 2:
            return f"L2 is smaller than one leaf block ({M * M * 2 // 1024} KB)"
    return ""


# ------------------------------------------------------------------------------------------
# decorations
# ------------------------------------------------------------------------------------------

def _text_width(text, scale, thickness):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def _wrap(text, width, scale, thickness):
    lines, current = [], ""
    for word in text.split():
        candidate = word if not current else current + " " + word
        if current and _text_width(candidate, scale, thickness) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _canvas(h, w):
    return np.full((h, w, 3), 255, np.uint8)


def add_legend(image, caption, entries):
    """Legend band at the bottom: wrapped caption, then (BGR color, text) swatches."""
    scale, thickness, margin, swatch, gap, line = 0.6, 1, 12, 18, 30, 32
    width = max(image.shape[1], 600)
    rows = [[]]
    x = margin
    for color, text in entries:
        w = swatch + 8 + _text_width(text, scale, thickness)
        if x + w > width - margin and rows[-1]:
            rows.append([])
            x = margin
        rows[-1].append((color, text, x))
        x += w + gap
    caption_lines = _wrap(caption, width - 2 * margin, scale, thickness)
    band = margin + (len(caption_lines) + len(rows)) * line
    out = _canvas(image.shape[0] + band, width)
    out[:image.shape[0], :image.shape[1]] = image
    y = image.shape[0] + margin
    for text in caption_lines:
        cv2.putText(out, text, (margin, y + swatch - 3), FONT, scale, BLACK, thickness, cv2.LINE_AA)
        y += line
    for row in rows:
        for color, text, x in row:
            out[y:y + swatch, x:x + swatch] = color
            cv2.rectangle(out, (x, y), (x + swatch - 1, y + swatch - 1), BLACK, 1)
            cv2.putText(out, text, (x + swatch + 8, y + swatch - 3), FONT, scale, BLACK, thickness, cv2.LINE_AA)
        y += line
    return out


def add_title(image, title):
    scale, thickness, margin = 0.8, 2, 12
    width = max(image.shape[1], 400)
    lines = _wrap(title, width - 2 * margin, scale, thickness)
    line_h = cv2.getTextSize("Ag", FONT, scale, thickness)[0][1] + 12
    band = margin + len(lines) * line_h
    out = _canvas(image.shape[0] + band, width)
    out[band:, :image.shape[1]] = image
    y = margin + line_h - 12
    for text in lines:
        cv2.putText(out, text, (margin, y), FONT, scale, BLACK, thickness, cv2.LINE_AA)
        y += line_h
    return out
