#!/usr/bin/env python3
"""Plot the distribution of attention scores produced by filter/main.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments.
import matplotlib.pyplot as plt
import numpy as np


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
        "--bin-width",
        type=float,
        help="Optional fixed bar width. If set, overrides --bins (e.g., 0.01).",
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
        "--zoom-bins",
        type=int,
        default=150,
        help="Number of bins for the zoomed-in plot (finer detail).",
    )
    parser.add_argument(
        "--zoom-bin-width",
        type=float,
        help="Optional fixed bar width for the zoomed-in plot. Overrides --zoom-bins.",
    )
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
    bin_width: Optional[float] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not scores:
        return
    lower, upper = (xlim if xlim else (None, None))
    # Filter scores to requested window if provided; if window is None, keep all.
    scores_window = (
        [s for s in scores if (lower is None or s >= lower) and (upper is None or s <= upper)]
        if xlim
        else list(scores)
    )
    if not scores_window:
        return
    mean_score = sum(scores_window) / len(scores_window)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        scores_window,
        bins=_resolve_bins(scores_window, bins, xlim, bin_width),
        color="#4e79a7",
        edgecolor="#1f3b57",
        alpha=0.9,
    )
    ax.axvline(
        mean_score,
        color="#e15759",
        linestyle="--",
        linewidth=2,
        label=f"mean={mean_score:.3f}",
    )
    ax.set_xlabel("Attention score")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.legend()
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
    bin_width: Optional[float] = None,
) -> None:
    """Mirror clean vs noise histograms along the X axis for easy comparison."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not clean_scores and not noise_scores:
        return
    fig, ax = plt.subplots(figsize=(8, 6))

    # Positive bars for clean, negative (mirrored) for noise to create symmetry about X axis.
    clean_counts, edges = _hist_counts(clean_scores, bins, xlim, bin_width)
    noise_counts, _ = _hist_counts(noise_scores, bins, xlim, bin_width)
    edges_arr = np.asarray(edges)
    centers = (edges_arr[:-1] + edges_arr[1:]) / 2.0
    width = edges_arr[1] - edges_arr[0] if len(edges_arr) > 1 else 1.0

    ax.bar(centers, clean_counts, width=width, color="#4e79a7", alpha=0.8, label="clean")
    ax.bar(centers, [-c for c in noise_counts], width=width, color="#e15759", alpha=0.8, label="noise")

    ax.axhline(0, color="#1f3b57", linewidth=1.2)
    ax.set_xlabel("Attention score")
    ax.set_ylabel("Count (clean up, noise down)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)


def _resolve_bins(
    scores: Sequence[float],
    bins: int,
    xlim: Optional[Tuple[float, float]],
    bin_width: Optional[float],
):
    if bin_width:
        if xlim:
            low, high = xlim
        else:
            low, high = min(scores), max(scores)
        if high <= low:
            high = low + bin_width
        edges = np.arange(low, high + bin_width, bin_width)
        if len(edges) < 2:
            edges = np.array([low, low + bin_width])
        return edges
    return bins


def _hist_counts(
    scores: Sequence[float],
    bins: int,
    xlim: Optional[Tuple[float, float]] = None,
    bin_width: Optional[float] = None,
) -> Tuple[List[float], List[float]]:
    if not scores:
        return [], [0.0, 1.0]
    hist_range = xlim if xlim else None
    bins_param = _resolve_bins(scores, bins, xlim, bin_width)
    counts, edges = np.histogram(scores, bins=bins_param, range=hist_range)
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
    plot_distribution(
        scores,
        bins=args.bins,
        bin_width=args.bin_width,
        output_path=output_path,
        title=title,
        dpi=args.dpi,
    )
    if args.zoom_upper:
        zoom_out = output_path.with_name(output_path.stem + f"_0-{args.zoom_upper:.2f}.png")
        plot_distribution(
            scores,
            bins=args.zoom_bins,
            bin_width=args.zoom_bin_width or args.bin_width,
            output_path=zoom_out,
            title=f"{title} (0-{args.zoom_upper:.2f})",
            dpi=args.dpi,
            xlim=(0.0, args.zoom_upper),
        )
    print(f"Saved histogram to {output_path}")


if __name__ == "__main__":
    main()
