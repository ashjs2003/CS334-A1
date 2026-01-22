import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter
import time
import math

# ============================================================
# Hilbert curve utilities
# ============================================================

def _rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y

def hilbert_d2xy(order, d):
    n = 1 << order
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def hilbert_points(order):
    npts = 1 << (2 * order)
    pts = [hilbert_d2xy(order, d) for d in range(npts)]
    return np.array(pts, dtype=float)

# ============================================================
# Mapping to physical square
# ============================================================

def map_to_square(points, L=3.0):
    N = points.max() if len(points) > 1 else 1
    pts = points / (N if N > 0 else 1)
    pts = pts * L - L / 2
    return pts

# ============================================================
# Main simulation
# ============================================================

def run_simulation(
    max_order=6,
    seconds_per_order=12,
    fps=30,
    output="hilbert_infinity.mp4"
):
    fig, ax = plt.subplots(figsize=(6, 6))
    writer = FFMpegWriter(fps=fps)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)

    with writer.saving(fig, output, dpi=200):
        global_t = 0.0

        for order in range(1, max_order + 1):
            pts = hilbert_points(order)
            pts = map_to_square(pts, L=3.0)

            n_frames = seconds_per_order * fps
            idxs = np.linspace(1, len(pts) - 1, n_frames).astype(int)

            for i in idxs:
                ax.clear()
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_xlim(-1.6, 1.6)
                ax.set_ylim(-1.6, 1.6)

                segment_pts = pts[:i]
                points = segment_pts.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Glow layer
                glow = LineCollection(
                    segments,
                    colors="white",
                    linewidth=6,
                    alpha=0.04
                )
                ax.add_collection(glow)

                # Main curve
                lc = LineCollection(
                    segments,
                    cmap="plasma",
                    linewidth=1.5,
                    alpha=0.95
                )
                lc.set_array(
                    np.linspace(0, global_t + i / fps, len(segments))
                )
                ax.add_collection(lc)

                writer.grab_frame()
                global_t += 1 / fps

            # Hold briefly on completed order
            for _ in range(int(fps * 1.5)):
                writer.grab_frame()

    print(f"Saved → {output}")

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_simulation(
        max_order=3,          # increase for longer / denser
        seconds_per_order=10, # slow revelation
        fps=30,
        output="hilbert_infinity.mp4"
    )
