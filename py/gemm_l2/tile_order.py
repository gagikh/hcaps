"""Tile order of the mma leaf GEMM (src/mma_gemm.cu): which output tiles of a product run at the same
time, and what they read. Port of the block -> tile mapping of batched_gemm_kernel().

One product C = A * B of M x M halves is cut into BM x BN tiles; tile (tm, tn) reads the A row
panel tm (BM rows x M) and the B column panel tn (M x BN). Blocks start roughly in launch order,
`sms * blocks_per_sm` at a time: one *wave*. The wave's A and B panels are what should stay in L2.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .images import BLACK, FONT, add_legend, add_title

SWIZZLE_ORDER = -1  # CUTLASS GemmIdentityThreadblockSwizzle<8>
COLUMN_ORDER = 0    # CUTLASS GemmBatchedIdentityThreadblockSwizzle
ORDERS = (SWIZZLE_ORDER, COLUMN_ORDER, 1, 2, 4, 8)


@dataclass(frozen=True)
class Config:
    """A configuration of src/mma_gemm.cu (same index as mma_gemm_t::config_name())."""
    BM: int
    BN: int
    BK: int
    stages: int
    smem_kb: int
    blocks_per_sm: int  # resident blocks per SM (shared memory limit of sm_86 / sm_89: 100 KB)

    @property
    def name(self):
        return f"{self.BM}x{self.BN}x{self.BK} {self.stages} stages"


CONFIGS = (
    Config(128, 256, 32, 4, 96, 1),
    Config(256, 128, 32, 4, 96, 1),
    Config(128, 256, 32, 3, 72, 1),
    Config(128, 128, 32, 4, 64, 1),
    Config(64, 128, 32, 4, 48, 2),
    Config(128, 64, 32, 4, 48, 2),
    Config(64, 64, 32, 4, 32, 3),
)


def order_name(order: int) -> str:
    if order == SWIZZLE_ORDER:
        return "swizzle (CUTLASS)"
    if order == COLUMN_ORDER:
        return "columns (CUTLASS batched)"
    return f"groups of {order} rows"


def swizzle_log_tile(tiles_n: int) -> int:
    """CUTLASS GemmIdentityThreadblockSwizzle<8>::get_log_tile()"""
    return 3 if tiles_n >= 6 else 2 if tiles_n >= 3 else 1 if tiles_n >= 2 else 0


def launch_order(M: int, cfg: Config, order: int) -> np.ndarray:
    """launch[tm, tn] = index of the block that computes tile (tm, tn) (blockIdx.x of product 0)."""
    tiles_m, tiles_n = M // cfg.BM, M // cfg.BN
    launch = np.full((tiles_m, tiles_n), -1, np.int64)
    if order == SWIZZLE_ORDER:
        log = swizzle_log_tile(tiles_n)
        blocks = (tiles_m << log) * ((tiles_n + (1 << log) - 1) >> log)
        for pid in range(blocks):
            x, y = pid % (tiles_m << log), pid // (tiles_m << log)
            tm, tn = x >> log, (y << log) + (x & ((1 << log) - 1))
            if tn < tiles_n:
                launch[tm, tn] = pid
        # no-op blocks of a narrower last strip still take launch slots, as on the GPU
    else:
        group = tiles_m if order == COLUMN_ORDER else order
        for pid in range(tiles_m * tiles_n):
            per_group = group * tiles_n
            first_m = (pid // per_group) * group
            rows = min(tiles_m - first_m, group)
            in_group = pid % per_group
            launch[first_m + in_group % rows, in_group // rows] = pid
    assert (launch >= 0).all(), "every tile gets a block"
    return launch


@dataclass
class WaveStats:
    wave_blocks: int      # blocks running at the same time
    waves: int            # waves for this one product
    a_panels: int         # distinct A row panels read by the first full wave
    b_panels: int         # distinct B column panels read by the first full wave
    slice_kb: float       # distinct A + B bytes per k slice of that wave (what L2 must keep hot)
    a_span_mb: float      # address range the wave's A rows cover at leading dimension ld


def wave_stats(M: int, cfg: Config, order: int, sms: int, ld: int, launch=None) -> WaveStats:
    launch = launch_order(M, cfg, order) if launch is None else launch
    wave = sms * cfg.blocks_per_sm
    first = np.argwhere(launch < wave)
    rows, cols = sorted(set(first[:, 0])), sorted(set(first[:, 1]))
    slice_bytes = (len(rows) * cfg.BM + len(cols) * cfg.BN) * cfg.BK * 2
    a_span = ((rows[-1] - rows[0] + 1) * cfg.BM) * ld * 2
    blocks = int(launch.max()) + 1
    return WaveStats(wave, -(-blocks // wave), len(rows), len(cols), slice_bytes / 1024, a_span / 2**20)


# wave colours (BGR): wave 0 strong, later waves alternate pale tones so neighbours differ
WAVE0 = (40, 110, 230)
PALE = [(245, 222, 196), (205, 235, 210), (230, 210, 240), (200, 230, 245), (240, 240, 200), (215, 215, 215)]
PANEL = (40, 110, 230)


def tile_order_image(M: int, cfg: Config, order: int, sms: int = 58, ld: int = 0, cell: int = 0):
    """Tile grid of one product: launch number per tile, coloured by wave (first wave highlighted),
    with the A row panels (left) and B column panels (top) the first wave reads.
    Returns (BGR image, WaveStats)."""
    ld = ld or M
    launch = launch_order(M, cfg, order)
    stats = wave_stats(M, cfg, order, sms, ld, launch)
    tiles_m, tiles_n = launch.shape
    # tiles drawn to scale (BM x BN), fitted into about 1100 x 900 pixels
    if not cell:
        cell = max(2, int(min(1100 / (tiles_n * cfg.BN), 900 / (tiles_m * cfg.BM)) * 128))
    cw, ch = max(2, cell * cfg.BN // 128), max(2, cell * cfg.BM // 128)
    bar, pad = 22, 40
    width, height = pad + bar + tiles_n * cw + 10, pad + bar + tiles_m * ch + 10
    img = np.full((height, width, 3), 255, np.uint8)
    x0, y0 = pad + bar, pad + bar
    wave_of = launch // stats.wave_blocks
    labels = cw >= 26 and ch >= 14
    scale = min(0.4, cw / 80)
    for tm in range(tiles_m):
        for tn in range(tiles_n):
            w = int(wave_of[tm, tn])
            color = WAVE0 if w == 0 else PALE[(w - 1) % len(PALE)]
            x, y = x0 + tn * cw, y0 + tm * ch
            img[y:y + ch, x:x + cw] = color
            if labels:
                text = str(int(launch[tm, tn]))
                (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
                cv2.putText(img, text, (x + (cw - tw) // 2, y + (ch + th) // 2), FONT, scale,
                            (255, 255, 255) if w == 0 else BLACK, 1, cv2.LINE_AA)
    # grid lines
    if cw >= 6:
        for tn in range(tiles_n + 1):
            img[y0:y0 + tiles_m * ch, min(x0 + tn * cw, width - 1)] = (160, 160, 160)
        for tm in range(tiles_m + 1):
            img[min(y0 + tm * ch, height - 1), x0:x0 + tiles_n * cw] = (160, 160, 160)
    # panels of the first wave: A rows on the left, B columns on the top
    first = launch < stats.wave_blocks
    for tm in range(tiles_m):
        if first[tm].any():
            img[y0 + tm * ch:y0 + (tm + 1) * ch, pad:pad + bar - 4] = PANEL
    for tn in range(tiles_n):
        if first[:, tn].any():
            img[pad:pad + bar - 4, x0 + tn * cw:x0 + (tn + 1) * cw] = PANEL
    cv2.putText(img, "B panels", (x0, pad - 8), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
    cv2.putText(img, "A", (pad - 22, y0 + 14), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
    img = add_title(img, f"Leaf {M} x {M}, {cfg.name}: {tiles_m} x {tiles_n} tiles, {order_name(order)}")
    img = add_legend(
        img,
        f"Number = launch order (blockIdx.x). First wave ({stats.wave_blocks} blocks = {sms} SMs x "
        f"{cfg.blocks_per_sm}) reads {stats.a_panels} A row panels and {stats.b_panels} B column panels: "
        f"{stats.slice_kb:.0f} KB per k slice, its A rows span {stats.a_span_mb:.1f} MB at ld = {ld}. "
        f"{stats.waves} waves per product.",
        [(WAVE0, "first wave"), (PALE[0], "later waves (alternating)"), (PANEL, "panels of the first wave")])
    return img, stats


def compare_orders_image(M: int, cfg: Config, sms: int = 58, ld: int = 0, orders=ORDERS):
    """All tile orders side by side (small cells), for the docs."""
    tiles = [tile_order_image(M, cfg, o, sms, ld, cell=max(2, int(360 / (M // 128))))[0] for o in orders]
    h = max(t.shape[0] for t in tiles)
    row = [np.pad(t, ((0, h - t.shape[0]), (0, 10), (0, 0)), constant_values=255) for t in tiles]
    per_row = 3
    lines = [np.hstack(row[i:i + per_row]) for i in range(0, len(row), per_row)]
    width = max(line.shape[1] for line in lines)
    return np.vstack([np.pad(line, ((0, 10), (0, width - line.shape[1]), (0, 0)), constant_values=255)
                      for line in lines])
