#!/usr/bin/env python3
"""Plot the distribution of attention scores produced by filter/main.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import matplotlib.patches as mpatches

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import numpy as np
import math

# Module default bin width (each bar represents this width on x axis)
DEFAULT_BIN_WIDTH = 0.01
# Highlight colors for the 0--zoom region (tweak these to change the marker color)
HIGHLIGHT_FACECOLOR = "#a0ceff"  # pale blue background for highlighted region
HIGHLIGHT_EDGECOLOR = "#1b8dff"  # blue border for highlighted region


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a histogram of attention scores from *_sentences.json."
    )
    parser.add_argument(
        "sentences_json",
        help="Path to the sentences JSON produced by filter/main.py (--annotate-attention optional).",
    )
    parser.add_argument(
        "--output",
        help="Where to save the plot. Defaults to <input_stem>_score_distribution.png.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins (increase for finer units).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution of the saved figure.",
    )
    parser.add_argument(
        "--kept-only",
        action="store_true",
        help="Only include sentences where kept==True in the distribution.",
    )
    parser.add_argument(
        "--title",
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--zoom-upper",
        type=float,
        help="If provided, also emit a zoomed-in plot from 0 to this value.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=0.01,
        help="Width of each bin (bar) on x axis. Default 0.01. bins will be calculated as (range/bin-width).",
    )
    # 缩放图也用bins参数，不再需要zoom-bins
    return parser.parse_args()


def load_scores(path: Path, kept_only: bool) -> List[float]:
    with path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of sentence entries in {path}")
    scores: List[float] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if kept_only and not item.get("kept"):
            continue
        if "score" not in item:
            continue
        try:
            scores.append(float(item["score"]))
        except (TypeError, ValueError):
            continue
    if not scores:
        raise ValueError("No valid scores found in the provided JSON.")
    return scores


def plot_distribution(
    scores: Sequence[float],
    bins: int,
    output_path: Path,
    title: str,
    dpi: int,
    xlim: Optional[Tuple[float, float]] = None,
    policy: Optional[str] = None,
    threshold: Optional[float] = None,
    bin_width: Optional[float] = None,
    highlight_zoom: Optional[float] = None,
) -> None:
    # Ensure output dir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not scores:
        return

    BIN_WIDTH = float(bin_width) if bin_width is not None else DEFAULT_BIN_WIDTH

    # Determine plotting window. If no xlim provided, default to full 0.0-1.0 range.
    if xlim:
        lower, upper = xlim
    else:
        lower, upper = 0.0, 1.0

    # Clip scores to the requested window for counting
    scores_window = [s for s in scores if s >= lower and s <= upper]
    if not scores_window:
        return

    # Compute number of bins so that each bin has width == BIN_WIDTH
    range_width = max(upper - lower, 0.0)
    bins_needed = max(1, int(round(range_width / BIN_WIDTH)))
    # Ensure exact coverage by making hist range an integer multiple of BIN_WIDTH
    hist_lower = float(lower)
    hist_upper = hist_lower + bins_needed * BIN_WIDTH

    counts, edges = np.histogram(scores_window, bins=bins_needed, range=(hist_lower, hist_upper))
    edges_arr = np.asarray(edges)
    lefts = edges_arr[:-1]
    # Force exact per-bar width so each bar represents BIN_WIDTH on x axis.
    width = BIN_WIDTH

    mean_score = sum(scores_window) / len(scores_window)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Align bars to left edges so the first bar starts at `lower` and the last
    # bar's right edge aligns to the histogram upper bound.
    ax.bar(lefts, counts, width=width, align="edge", color="#4e79a7", edgecolor="#1f3b57", alpha=0.9)
    ax.axvline(mean_score, color="#e15759", linestyle="--", linewidth=2, label=f"mean={mean_score:.3f}")
    # If running with absolute-threshold policy, mark threshold on the plot.
    # Only mark threshold on a zoomed plot (explicit xlim provided).
    if xlim is not None and policy == "absolute" and threshold is not None:
        if threshold >= lower and threshold <= upper:
            # vivid yellow for threshold marker
            ax.axvline(threshold, color="#ffdd00", linestyle="-.", linewidth=2, label=f"threshold={threshold:.4f}")
    # Enforce y-axis clipping to requested range
    ax.set_ylim(-200, 200)
    ax.set_xlabel("Attention score")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.legend()
    # Remove any automatic x-axis padding so bars touch the plot edges.
    ax.set_xmargin(0)
    ax.margins(x=0)
    # First ensure axis covers the histogram bin boundaries exactly
    ax.set_xlim(hist_lower, hist_upper)
    # Optionally highlight the small 0--zoom region. Draw the rectangle
    # whenever `highlight_zoom` is provided, but clipped to the current
    # plotting window so it works for both full-range and zoomed plots.
    if highlight_zoom is not None:
        # compute rectangle bounds clipped to plotting window
        rect_x0 = max(lower, 0.0)
        rect_x1 = min(highlight_zoom, upper)
        if rect_x1 > rect_x0:
            y0, y1 = ax.get_ylim()
            rect = mpatches.Rectangle(
                (rect_x0, y0),
                rect_x1 - rect_x0,
                y1 - y0,
                facecolor=HIGHLIGHT_FACECOLOR,
                edgecolor=HIGHLIGHT_EDGECOLOR,
                linewidth=1.0,
                alpha=0.25,
                zorder=0,
            )
            ax.add_patch(rect)
    # If caller requested a narrower viewing window (zoom), apply it
    # after locking margins so the visible region still has no padding.
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def plot_noise_vs_clean(
    clean_scores: Sequence[float],
    noise_scores: Sequence[float],
    bins: int,
    output_path: Path,
    title: str,
    dpi: int,
    xlim: Optional[Tuple[float, float]] = None,
    policy: Optional[str] = None,
    threshold: Optional[float] = None,
    bin_width: Optional[float] = None,
    highlight_zoom: Optional[float] = None,
) -> None:
    """Mirror clean vs noise histograms along the X axis for easy comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_scores and not noise_scores:
        return
    fig, ax = plt.subplots(figsize=(8, 8))

    # Fixed bin width (each bar represents this width on x axis)
    BIN_WIDTH = float(bin_width) if bin_width is not None else DEFAULT_BIN_WIDTH

    # Determine plotting window
    if xlim:
        lower, upper = xlim
    else:
        lower, upper = 0.0, 1.0

    # Clip scores to window
    clean_window = [s for s in clean_scores if s >= lower and s <= upper]
    noise_window = [s for s in noise_scores if s >= lower and s <= upper]

    range_width = max(upper - lower, 0.0)
    bins_needed = max(1, int(round(range_width / BIN_WIDTH)))
    hist_lower = float(lower)
    hist_upper = hist_lower + bins_needed * BIN_WIDTH

    clean_counts, edges = np.histogram(clean_window, bins=bins_needed, range=(hist_lower, hist_upper))
    noise_counts, _ = np.histogram(noise_window, bins=bins_needed, range=(hist_lower, hist_upper))
    edges_arr = np.asarray(edges)
    lefts = edges_arr[:-1]
    # Force exact per-bar width so each bar represents BIN_WIDTH on x axis.
    width = BIN_WIDTH

    # Positive bars for clean, negative (mirrored) for noise to create symmetry
    ax.bar(lefts, clean_counts, width=width, align="edge", color="#4e79a7", alpha=0.8, label="clean")
    ax.bar(lefts, [-c for c in noise_counts], width=width, align="edge", color="#e15759", alpha=0.8, label="noise")

    # Mean vertical lines removed per user request.
    # Enforce y-axis clipping to requested range
    ax.set_ylim(-100, 100) #y轴限制

    # Only draw threshold marker on a zoomed plot (explicit xlim provided).
    if xlim is not None and policy == "absolute" and threshold is not None:
        if threshold >= lower and threshold <= upper:
            # vivid yellow for threshold marker
            ax.axvline(threshold, color="#ffdd00", linestyle="-.", linewidth=2, label=f"threshold={threshold:.4f}")

    ax.axhline(0, color="#1f3b57", linewidth=1.2)
    ax.set_xlabel("Attention score")
    ax.set_ylabel("Count (clean up, noise down)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Remove automatic x-axis padding so bars touch the plot edges.
    ax.set_xmargin(0)
    ax.margins(x=0)
    # Ensure axis covers the histogram bin boundaries exactly
    ax.set_xlim(hist_lower, hist_upper)
    # Optionally highlight the small 0--zoom region. Draw the rectangle
    # whenever `highlight_zoom` is provided, clipped to the plotting window.
    if highlight_zoom is not None:
        rect_x0 = max(lower, 0.0)
        rect_x1 = min(highlight_zoom, upper)
        if rect_x1 > rect_x0:
            y0, y1 = ax.get_ylim()
            rect = mpatches.Rectangle(
                (rect_x0, y0),
                rect_x1 - rect_x0,
                y1 - y0,
                facecolor=HIGHLIGHT_FACECOLOR,
                edgecolor=HIGHLIGHT_EDGECOLOR,
                linewidth=1.0,
                alpha=0.25,
                zorder=0,
            )
            ax.add_patch(rect)

    # Apply requested viewing window (zoom) after locking margins
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def _hist_counts(
    scores: Sequence[float],
    bins: int,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[List[float], List[float]]:
    if not scores:
        return [], [0.0, 1.0]
    hist_range = xlim if xlim else None
    counts, edges = np.histogram(scores, bins=bins, range=hist_range)
    return counts.tolist(), edges.tolist()


def main() -> None:
    args = parse_args()
    input_path = Path(args.sentences_json)
    scores = load_scores(input_path, kept_only=args.kept_only)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(f"{input_path.stem}_score_distribution.png")
    )
    subtitle = "kept" if args.kept_only else "all"
    title = args.title or f"Score distribution ({subtitle} sentences)"
    # 设置 bin_width 属性用于 plot_distribution 和 plot_noise_vs_clean
    plot_distribution.bin_width = args.bin_width
    plot_noise_vs_clean.bin_width = args.bin_width
    plot_distribution(
        scores,
        bins=args.bins,
        output_path=output_path,
        title=title,
        dpi=args.dpi,
        highlight_zoom=args.zoom_upper,
    )
    if args.zoom_upper:
        zoom_out = output_path.with_name(output_path.stem + f"_0-{args.zoom_upper:.2f}.png")
        plot_distribution(
            scores,
            bins=args.bins,
            output_path=zoom_out,
            title=f"{title} (0-{args.zoom_upper:.2f})",
            dpi=args.dpi,
            xlim=(0.0, args.zoom_upper),
            highlight_zoom=args.zoom_upper,
        )
    print(f"Saved histogram to {output_path}")


if __name__ == "__main__":
    main()
