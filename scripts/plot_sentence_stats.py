#!/usr/bin/env python3
"""
Plot sentence-level statistics from per-file `*_sentences.json` exports.

For each input JSON file (array of objects with keys including at least
`relation`, `score`, `is_noise`, `kept`), this script generates a multi-page
PDF with plots:
  - Score distribution (hist + KDE)
  - Score boxplot grouped by `is_noise` (missing labeled)
  - Top relations by sentence count (stacked kept/dropped)
  - Score distributions (box/violin) for top-N relations
  - Scatter plot of score vs kept (jitter) colored by is_noise

Usage examples:
  python3 scripts/plot_sentence_stats.py --inputs reports/arnor_*_sentences.json

If no inputs provided the script will look for `reports/*_sentences.json`.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

# Runtime dependency imports with graceful error messages
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception as e:
    raise RuntimeError(
        "matplotlib is required to run this script. Install with: python3 -m pip install matplotlib"
    ) from e

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError(
        "pandas is required to run this script. Install with: python3 -m pip install pandas"
    ) from e

try:
    import seaborn as sns
except Exception:
    sns = None

import numpy as np

if sns is not None:
    sns.set(style="whitegrid")
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


def load_sentences(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload)
    # Ensure columns exist
    for col in ["relation", "score", "is_noise", "kept", "text", "h_name", "t_name"]:
        if col not in df.columns:
            df[col] = None
    # Normalize types
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    # Normalize is_noise to string categories: True, False, Missing
    def norm_is_noise(x):
        if x is True:
            return "true"
        if x is False:
            return "false"
        if isinstance(x, str):
            lx = x.lower()
            if lx in {"true", "1"}:
                return "true"
            if lx in {"false", "0"}:
                return "false"
            return "missing"
        return "missing"

    df["is_noise_cat"] = df["is_noise"].apply(norm_is_noise)
    df["kept"] = df["kept"].astype(bool)
    return df


def plot_for_file(path: Path, out_pdf: Path, top_relations: int = 10):
    df = load_sentences(path)
    if df.empty:
        print(f"{path} empty, skipping")
        return
    pdf = PdfPages(out_pdf)

    # Page 1: overall score distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df["score"], kde=True, stat="density", bins=40, color="C0")
    plt.title(f"Score distribution — {path.stem}")
    plt.xlabel("score")
    plt.ylabel("density")
    pdf.savefig()
    plt.close()

    # Page 2: boxplot of score by is_noise
    plt.figure(figsize=(8, 6))
    order = ["true", "false", "missing"]
    sns.boxplot(x="is_noise_cat", y="score", data=df, order=order, palette="Set2")
    plt.title("Score by is_noise")
    pdf.savefig()
    plt.close()

    # Page 3: top relations counts and kept vs dropped stacked bar
    rel_counts = df["relation"].value_counts()
    top_rels = rel_counts.head(min(len(rel_counts), 50))
    top_idx = list(top_rels.index)
    counts = df[df["relation"].isin(top_idx)].groupby(["relation", "kept"]).size().unstack(fill_value=0)
    counts = counts.loc[top_idx]
    plt.figure(figsize=(10, max(4, len(top_idx) * 0.15)))
    counts.plot(kind="bar", stacked=True, color=["#1f77b4", "#ff7f0e"])
    plt.title("Top relations: kept vs dropped")
    plt.ylabel("count")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Page 4: score distribution for top-N relations (boxplot)
    top_n = rel_counts.head(top_relations).index.tolist()
    if top_n:
        plt.figure(figsize=(10, max(6, len(top_n) * 0.6)))
        sns.boxplot(y="relation", x="score", data=df[df["relation"].isin(top_n)], order=top_n, palette="vlag")
        plt.title(f"Score distribution by top-{top_relations} relations")
        plt.xlabel("score")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Page 5: violin plots split by is_noise for top relations
    if top_n:
        plt.figure(figsize=(10, max(6, len(top_n) * 0.6)))
        sns.violinplot(y="relation", x="score", hue="is_noise_cat", data=df[df["relation"].isin(top_n)], order=top_n, split=True, palette="Set2")
        plt.title(f"Score violin by is_noise for top-{top_relations} relations")
        plt.xlabel("score")
        plt.legend(title="is_noise")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Page 6: scatter score vs kept (jitter) colored by is_noise
    plt.figure(figsize=(8, 6))
    # Jitter kept to show two rows using numpy
    jitter = df["kept"].apply(lambda b: 1 if b else 0) + (np.random.rand(len(df)) - 0.5) * 0.08
    cmap = {"true": "red", "false": "green", "missing": "gray"}
    for cat, subset in df.groupby("is_noise_cat"):
        plt.scatter(subset["score"], subset["kept"].apply(lambda b: 1 if b else 0) + (np.random.rand(len(subset)) - 0.5) * 0.08,
                    alpha=0.6, label=cat, color=cmap.get(cat, "gray"), s=20)
    plt.yticks([0, 1], ["dropped", "kept"])
    plt.xlabel("score")
    plt.title("Score vs kept (colored by is_noise)")
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Page 7: aggregate statistics table as text
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    stats = []
    total = len(df)
    for rel, cnt in rel_counts.head(30).items():
        sub = df[df["relation"] == rel]
        avg = sub["score"].mean()
        kept_rate = sub["kept"].mean()
        noise_rate = (sub["is_noise_cat"] == "true").mean()
        stats.append((rel, int(cnt), f"{avg:.3f}", f"{kept_rate:.2%}", f"{noise_rate:.2%}"))
    tbl = pd.DataFrame(stats, columns=["relation", "count", "avg_score", "kept_rate", "noise_rate"]) 
    tbl_text = tbl.to_string(index=False)
    plt.text(0.01, 0.99, f"Summary for {path.name}\nTotal sentences: {total}\n\n{tbl_text}", fontsize=8, va='top', family='monospace')
    pdf.savefig()
    plt.close()

    # Additional pages: per-relation -> per-entity-pair histograms (kept inside same PDF)
    MAX_REL_PAGES = 2000
    pairs_page_count = 0
    relations = df['relation'].fillna('UNKNOWN').unique().tolist()

    # New pages: overall NA vs non-NA score distributions (two pages)
    # Define NA mask (case-insensitive match to 'NA')
    rel_series = df['relation'].fillna('').astype(str)
    is_na_mask = rel_series.str.strip().str.lower() == 'na'

    def plot_na_group(group_df, title_prefix: str):
        if group_df.empty:
            return
        grp = group_df.copy()
        min_s = float(grp['score'].min())
        max_s = float(grp['score'].max())
        if math.isclose(min_s, max_s):
            bins = np.linspace(min_s - 0.5, max_s + 0.5, 11)
        else:
            bins = np.linspace(min_s, max_s, 21)
        fig, ax = plt.subplots(figsize=(8, 5))
        cats = grp['is_noise_cat'].unique().tolist()
        widths = (bins[1] - bins[0]) * 0.9
        n_cats = max(1, len(cats))
        offsets = np.linspace(-0.4, 0.4, n_cats)
        for i, cat in enumerate(sorted(cats)):
            subset = grp[grp['is_noise_cat'] == cat]
            counts, _ = np.histogram(subset['score'].fillna(0.0), bins=bins)
            centers = (bins[:-1] + bins[1:]) / 2.0
            ax.bar(centers + offsets[i] * (widths / (n_cats or 1)), counts, width=widths / (n_cats or 1), alpha=0.8, label=str(cat))
        ax.set_title(f"{title_prefix} — Score histogram (split by is_noise)\nTotal sentences: {len(grp)}")
        ax.set_xlabel('score')
        ax.set_ylabel('count')
        ax.legend(title='is_noise')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    # Page: NA scores
    plot_na_group(df[is_na_mask], "NA (all relations labeled 'NA')")
    # Page: non-NA scores
    plot_na_group(df[~is_na_mask], "Non-NA (all relations except 'NA')")
    for rel in relations:
        rel_df = df[df['relation'] == rel].copy()
        # build entity pairs
        rel_df['entity_pair'] = rel_df.apply(lambda r: f"{r.get('h_name') or 'H'}||{r.get('t_name') or 'T'}", axis=1)
        pair_counts = rel_df['entity_pair'].value_counts()
        # iterate top pairs but cap them to avoid enormous PDFs
        for pair, cnt in pair_counts.items():
            if pairs_page_count >= MAX_REL_PAGES:
                break
            pair_df = rel_df[rel_df['entity_pair'] == pair]
            if len(pair_df) < 2:
                continue
            # Prepare histogram bins
            min_s = float(pair_df['score'].min())
            max_s = float(pair_df['score'].max())
            if math.isclose(min_s, max_s):
                bins = np.linspace(min_s - 0.5, max_s + 0.5, 11)
            else:
                bins = np.linspace(min_s, max_s, 21)
            fig, ax = plt.subplots(figsize=(8, 5))
            # compute counts for each is_noise category
            cats = pair_df['is_noise_cat'].unique().tolist()
            widths = (bins[1] - bins[0]) * 0.9
            # side-by-side bars per category
            n_cats = max(1, len(cats))
            offsets = np.linspace(-0.4, 0.4, n_cats)
            for i, cat in enumerate(sorted(cats)):
                subset = pair_df[pair_df['is_noise_cat'] == cat]
                counts, _ = np.histogram(subset['score'].fillna(0.0), bins=bins)
                centers = (bins[:-1] + bins[1:]) / 2.0
                ax.bar(centers + offsets[i] * (widths / (n_cats or 1)), counts, width=widths / (n_cats or 1), alpha=0.7, label=str(cat))
            ax.set_title(f"Scores for relation={rel}\nentity_pair={pair} (n={len(pair_df)})")
            ax.set_xlabel('score')
            ax.set_ylabel('count')
            ax.legend(title='is_noise')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            pairs_page_count += 1
        if pairs_page_count >= MAX_REL_PAGES:
            break

    pdf.close()
    print(f"Wrote plots to {out_pdf}")


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", help="Input JSON files. If omitted, use reports/*_sentences.json")
    parser.add_argument("--out-dir", help="Output directory for PDFs (defaults to same dir as JSON)")
    parser.add_argument("--top-n", type=int, default=10, help="Top N relations for per-relation plots")
    args = parser.parse_args(argv)

    if not args.inputs:
        candidates = sorted(Path("reports").glob("*_sentences.json"))
        inputs = candidates
    else:
        inputs = [Path(p) for p in args.inputs]

    if not inputs:
        print("No input sentence files found. Generate `*_sentences.json` first.")
        return

    for p in inputs:
        if not p.exists():
            print(f"Missing {p}, skipping")
            continue
        out_dir = Path(args.out_dir) if args.out_dir else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pdf = out_dir / f"{p.stem}_plots.pdf"
        plot_for_file(p, out_pdf, top_relations=args.top_n)


if __name__ == "__main__":
    main()
