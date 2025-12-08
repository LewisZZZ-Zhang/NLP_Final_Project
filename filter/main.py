#!/usr/bin/env python3
"""
Lightweight attention-style filter for the NYT10m distant supervision corpus.

This script reads an OpenNRE-style JSONL split (e.g., data/nyt10m/nyt10m_train.txt),
clusters sentences into bags by (relation, head id, tail id), assigns each sentence a
heuristic attention score, applies one of the filtering rules described in the project
proposal (absolute threshold, top-k, relative margin), and writes the filtered dataset
back to disk.

Example:
    python filter/main.py --split train --policy topk --topk 2 --annotate-attention

Use `python filter/main.py --help` for the complete CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model import (
    AttentionProvider,
    ExternalAttentionProvider,
    HeuristicAttentionProvider,
    RELATION_HINTS,
    SentenceInstance,
    clamp,
)


LOGGER = logging.getLogger("nyt10m_filter")


def make_bag_id(relation: str, head_id: str, tail_id: str) -> str:
    return f"{relation}||{head_id}||{tail_id}"




def ensure_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def safe_extract_span(text: str, entry: Dict[str, Any]) -> Tuple[int, int]:
    """Return a best-effort character span for an entity mention."""
    start, end = 0, 0
    pos = entry.get("pos")
    if isinstance(pos, (list, tuple)) and len(pos) == 2:
        start, end = int(pos[0]), int(pos[1])
    else:
        surface = entry.get("name") or entry.get("text") or ""
        lowered = text.lower()
        idx = lowered.find(surface.lower())
        if idx >= 0:
            start, end = idx, idx + len(surface)
    start = clamp(start, 0, len(text))
    end = clamp(end, start, len(text))
    return int(start), int(end)


def _plot_scores_if_requested(
    scores: Dict[str, List[float]],
    output_path: Path,
    input_label: str,
    enabled: bool,
    bins: int,
    dpi: int,
    zoom_upper: Optional[float],
    zoom_bins: int,
    policy: Optional[str] = None,
    threshold: Optional[float] = None,
    plot_bin_width: Optional[float] = None,
    plot_zoom_bin_width: Optional[float] = None,
) -> None:
    """Render a histogram of attention scores, if plotting is enabled.

    When `policy == 'absolute'` and `threshold` is provided, the plots will mark
    the threshold location with a vertical line.
    """
    if not enabled or not scores:
        return
    try:
        from plot_scores import plot_distribution, plot_noise_vs_clean  # Local helper; uses matplotlib with Agg backend.
    except Exception as exc:  # pragma: no cover - best-effort plotting
        LOGGER.warning("Skipping score plot (dependency missing): %s", exc)
        return
    title = f"Score distribution ({input_label})"
    try:
        # NOTE: overall distribution plots intentionally omitted per user request.
        # The function will still emit the clean/noise split plots (below) when labels exist.
        if scores["clean"] or scores["noise"]:
            noise_plot = output_path.with_name(output_path.stem + "_noise_split.png")
            plot_noise_vs_clean(
                clean_scores=scores["clean"],
                noise_scores=scores["noise"],
                bins=bins,
                output_path=noise_plot,
                title=f"Clean vs noise scores ({input_label})",
                dpi=dpi,
                policy=policy,
                threshold=threshold,
                bin_width=plot_bin_width,
                highlight_zoom=zoom_upper,
            )
            print(f"Saved clean/noise split plot to {noise_plot}")
            if zoom_upper:
                zoom_noise_plot = output_path.with_name(output_path.stem + f"_noise_split_0-{zoom_upper:.2f}.png")
                plot_noise_vs_clean(
                    clean_scores=scores["clean"],
                    noise_scores=scores["noise"],
                    bins=zoom_bins,
                    output_path=zoom_noise_plot,
                    title=f"Clean vs noise scores ({input_label}) (0-{zoom_upper:.2f})",
                    dpi=dpi,
                    xlim=(0.0, zoom_upper),
                    policy=policy,
                    threshold=threshold,
                        bin_width=plot_zoom_bin_width,
                        highlight_zoom=zoom_upper,
                )
                print(f"Saved zoomed clean/noise split plot to {zoom_noise_plot}")
    except Exception as exc:  # pragma: no cover - best-effort plotting
        LOGGER.warning("Failed to plot scores for %s: %s", input_label, exc)


def load_dataset(
    path: Path, limit_bags: Optional[int] = None
) -> Tuple[Dict[str, List[SentenceInstance]], List[str]]:
    """Load an OpenNRE JSONL file and return (bags, bag_order)."""
    bags: Dict[str, List[SentenceInstance]] = {}
    order: List[str] = []
    skipped = 0
    LOGGER.info("Loading %s", path)
    with path.open() as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            relation = record.get("relation", "NA")
            # Support multiple possible text field names (OpenNRE uses 'text', Arnor files use 'sentText')
            text = record.get("text") or record.get("sentText") or record.get("sent_text") or ""

            # Normalize head/tail to dicts. OpenNRE uses 'h'/'t' dicts; Arnor uses 'em1Text'/'em2Text' strings.
            raw_h = record.get("h")
            raw_t = record.get("t")
            if isinstance(raw_h, dict):
                head = raw_h
            else:
                h_name = record.get("em1Text") or record.get("em1") or (raw_h if isinstance(raw_h, str) else "")
                head = {"name": str(h_name)} if h_name else {}
            if isinstance(raw_t, dict):
                tail = raw_t
            else:
                t_name = record.get("em2Text") or record.get("em2") or (raw_t if isinstance(raw_t, str) else "")
                tail = {"name": str(t_name)} if t_name else {}

            # If entityMentions present, try to find character spans for the head/tail mentions
            # and populate a 'pos' entry (start,end) in head/tail to improve span extraction.
            mentions = record.get("entityMentions") or record.get("entity_mentions") or []

            def find_mention_span(name: str) -> Optional[Tuple[int, int]]:
                if not name:
                    return None
                name_l = name.lower()
                # First try to match mentions by their 'text' field
                for m in mentions:
                    m_text = m.get("text") or m.get("name") or ""
                    if not m_text:
                        continue
                    if m_text.lower() == name_l:
                        # If mention provides explicit char offsets, use them
                        if "pos" in m and isinstance(m.get("pos"), (list, tuple)) and len(m.get("pos")) == 2:
                            return (int(m.get("pos")[0]), int(m.get("pos")[1]))
                        if "start" in m and "end" in m:
                            try:
                                return (int(m.get("start")), int(m.get("end")))
                            except Exception:
                                pass
                        # Some datasets use token index 'start' only; fall back to string search below
                        break
                # Fallback: locate first substring occurrence in the sentence text
                if name and text:
                    idx = text.lower().find(name_l)
                    if idx >= 0:
                        return (idx, idx + len(name))
                return None

            # Try to fill pos for head/tail if possible
            if mentions:
                if head.get("name") and not head.get("pos"):
                    span = find_mention_span(head.get("name"))
                    if span:
                        head["pos"] = [span[0], span[1]]
                if tail.get("name") and not tail.get("pos"):
                    span = find_mention_span(tail.get("name"))
                    if span:
                        tail["pos"] = [span[0], span[1]]

            # Build bag id from available id or name fields
            h_id = head.get("id") if isinstance(head.get("id"), (str, int)) else head.get("name", "")
            t_id = tail.get("id") if isinstance(tail.get("id"), (str, int)) else tail.get("name", "")
            bag_id = make_bag_id(relation, str(h_id), str(t_id))
            if limit_bags and bag_id not in bags and len(order) >= limit_bags:
                continue
            instance = SentenceInstance(
                relation=relation,
                text=text,
                text_lower=text.lower(),
                h_name=str(head.get("name", "")),
                t_name=str(tail.get("name", "")),
                h_pos=safe_extract_span(text, head),
                t_pos=safe_extract_span(text, tail),
                data=record,
            )
            if bag_id not in bags:
                bags[bag_id] = []
                order.append(bag_id)
            bags[bag_id].append(instance)
    if skipped:
        LOGGER.warning("Skipped %d malformed lines in %s", skipped, path)
    LOGGER.info("Loaded %d bags (%d sentences)", len(bags), sum(len(b) for b in bags.values()))
    return bags, order




class FilterPolicy:
    def __init__(self, min_keep: int = 1):
        self.min_keep = max(1, min_keep)

    def select(self, scores: Sequence[float]) -> List[bool]:
        raise NotImplementedError


class AbsoluteThresholdPolicy(FilterPolicy):
    def __init__(self, threshold: float, min_keep: int = 1, scale_by_bag_size: bool = False):
        super().__init__(min_keep)
        self.threshold = threshold
        self.scale_by_bag_size = scale_by_bag_size

    def select(self, scores: Sequence[float]) -> List[bool]:
        if not scores:
            return []
        # If scaling is enabled, interpret threshold as "per-sentence mass" relative to bag size.
        effective_threshold = self.threshold / len(scores) if self.scale_by_bag_size else self.threshold
        mask = [score >= effective_threshold for score in scores]
        if sum(mask) >= self.min_keep:
            return mask
        # Fallback: keep at least `min_keep` highest sentences.
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in top_indices[: self.min_keep]:
            mask[idx] = True
        return mask


class TopKPolicy(FilterPolicy):
    def __init__(self, k: int, min_keep: int = 1):
        super().__init__(min_keep)
        self.k = max(1, k)

    def select(self, scores: Sequence[float]) -> List[bool]:
        if not scores:
            return []
        k = min(len(scores), max(self.k, self.min_keep))
        mask = [False] * len(scores)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        for idx in top_indices:
            mask[idx] = True
        return mask


class RelativeMarginPolicy(FilterPolicy):
    def __init__(self, margin: float, min_keep: int = 1):
        super().__init__(min_keep)
        self.margin = max(0.0, margin)

    def select(self, scores: Sequence[float]) -> List[bool]:
        if not scores:
            return []
        mask = [False] * len(scores)
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_idx = sorted_idx[0]
        top_score = scores[top_idx]
        threshold = top_score - self.margin
        for idx in sorted_idx:
            if scores[idx] >= threshold or sum(mask) < self.min_keep:
                mask[idx] = True
            else:
                break
        # Guarantee minimum keep.
        kept = sum(mask)
        if kept < self.min_keep:
            for idx in sorted_idx[: self.min_keep]:
                mask[idx] = True
        return mask


class DropBottomPolicy(FilterPolicy):
    def __init__(self, drop_k: int, min_keep: int = 1):
        super().__init__(min_keep)
        self.drop_k = max(0, drop_k)

    def select(self, scores: Sequence[float]) -> List[bool]:
        n = len(scores)
        if n == 0:
            return []
        keep = [True] * n
        if self.drop_k <= 0:
            return keep
        max_drop = max(0, n - self.min_keep)
        drop_count = min(self.drop_k, max_drop)
        if drop_count == 0:
            return keep
        bottom_idx = sorted(range(n), key=lambda i: scores[i])[:drop_count]
        for idx in bottom_idx:
            keep[idx] = False
        return keep


class DropPercentPolicy(FilterPolicy):
    def __init__(self, drop_percent: float, min_keep: int = 1):
        super().__init__(min_keep)
        self.drop_percent = clamp(drop_percent, 0.0, 1.0)

    def select(self, scores: Sequence[float]) -> List[bool]:
        n = len(scores)
        if n == 0:
            return []
        keep = [True] * n
        target_keep = n - int(round(n * self.drop_percent))
        target_keep = max(self.min_keep, min(n, target_keep))
        k_to_drop = n - target_keep
        if k_to_drop <= 0:
            return keep
        bottom_idx = sorted(range(n), key=lambda i: scores[i])[:k_to_drop]
        for idx in bottom_idx:
            keep[idx] = False
        return keep


class StatsTracker:
    def __init__(self, enable_noise_eval: bool = False):
        self.total_sent = 0
        self.kept_sent = 0
        self.total_bags = 0
        self.relation_before: Counter[str] = Counter()
        self.relation_after: Counter[str] = Counter()
        self.noise_eval_enabled = enable_noise_eval
        self.noise_tp = 0
        self.noise_fp = 0
        self.noise_fn = 0
        self.noise_tn = 0
        self.noise_total = 0
        self.noise_missing = 0

    def update(self, relation: str, bag_size: int, kept: int) -> None:
        self.total_sent += bag_size
        self.kept_sent += kept
        self.total_bags += 1
        self.relation_before[relation] += bag_size
        self.relation_after[relation] += kept

    def record_noise(self, keep: bool, noise_label: Optional[bool]) -> None:
        if not self.noise_eval_enabled:
            return
        if noise_label is None:
            self.noise_missing += 1
            return
        self.noise_total += 1
        if noise_label:  # true noise -> should drop
            if keep:
                self.noise_fn += 1
            else:
                self.noise_tp += 1
        else:
            if keep:
                self.noise_tn += 1
            else:
                self.noise_fp += 1

    def summary(self) -> Dict[str, Any]:
        avg_before = self.total_sent / self.total_bags if self.total_bags else 0.0
        avg_after = self.kept_sent / self.total_bags if self.total_bags else 0.0
        removal_rate = 1.0 - (self.kept_sent / self.total_sent) if self.total_sent else 0.0
        return {
            "bags": self.total_bags,
            "sentences_before": self.total_sent,
            "sentences_after": self.kept_sent,
            "avg_sentences_per_bag_before": avg_before,
            "avg_sentences_per_bag_after": avg_after,
            "removal_rate": removal_rate,
            "per_relation": [
                {
                    "relation": rel,
                    "before": self.relation_before[rel],
                    "after": self.relation_after[rel],
                }
                for rel in sorted(
                    self.relation_before,
                    key=lambda r: self.relation_before[r],
                    reverse=True,
                )
            ],
            "noise_eval": self._noise_summary(),
        }

    def format_summary(self, max_relations: int = 8) -> str:
        info = self.summary()
        lines = [
            f"Bags processed: {info['bags']}",
            f"Sentences kept: {info['sentences_after']} / {info['sentences_before']} "
            f"({1.0 - info['removal_rate']:.2%})",
            "Avg sentences per bag: "
            f"{info['avg_sentences_per_bag_before']:.2f} -> {info['avg_sentences_per_bag_after']:.2f}",
            "Per-relation retention (top counts):",
        ]
        for entry in info["per_relation"][:max_relations]:
            before = entry["before"]
            after = entry["after"]
            rate = (after / before) if before else 0.0
            lines.append(f"  {entry['relation']}: {after}/{before} kept ({rate:.1%})")
        noise_info = info.get("noise_eval")
        if noise_info:
            lines.append("Noise filtering evaluation (predict drop = noise):")
            lines.append(
                "  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
                "accuracy={accuracy:.3f}".format(**noise_info)
            )
            lines.append(
                f"  counts: TP={noise_info['tp']} FP={noise_info['fp']} "
                f"FN={noise_info['fn']} TN={noise_info['tn']} total={noise_info['total']}"
            )
            if noise_info["missing_labels"]:
                lines.append(
                    f"  missing is_noise labels: {noise_info['missing_labels']}"
                )
        return "\n".join(lines)

    def _noise_summary(self) -> Optional[Dict[str, Any]]:
        if not self.noise_eval_enabled:
            return None
        total = self.noise_total
        if total == 0 and self.noise_missing == 0:
            return None
        precision = self.noise_tp / (self.noise_tp + self.noise_fp) if (self.noise_tp + self.noise_fp) else 0.0
        recall = self.noise_tp / (self.noise_tp + self.noise_fn) if (self.noise_tp + self.noise_fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy = (self.noise_tp + self.noise_tn) / total if total else 0.0
        return {
            "tp": self.noise_tp,
            "fp": self.noise_fp,
            "fn": self.noise_fn,
            "tn": self.noise_tn,
            "total": total,
            "missing_labels": self.noise_missing,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


def build_attention_provider(args: argparse.Namespace) -> AttentionProvider:
    heuristic = HeuristicAttentionProvider(temperature=args.temperature)
    if args.attention_file:
        return ExternalAttentionProvider(Path(args.attention_file), fallback=heuristic)
    return heuristic


def build_policy(args: argparse.Namespace) -> FilterPolicy:
    if args.policy == "absolute":
        return AbsoluteThresholdPolicy(
            threshold=args.threshold,
            min_keep=args.min_keep,
            scale_by_bag_size=args.absolute_scale_by_bag,
        )
    if args.policy == "topk":
        return TopKPolicy(k=args.topk, min_keep=args.min_keep)
    if args.policy == "relative":
        return RelativeMarginPolicy(margin=args.margin, min_keep=args.min_keep)
    if args.policy == "drop":
        return DropBottomPolicy(drop_k=args.drop_bottom, min_keep=args.min_keep)
    if args.policy == "drop_pct":
        return DropPercentPolicy(drop_percent=args.drop_percent, min_keep=args.min_keep)
    raise ValueError(f"Unsupported policy: {args.policy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter NYT10m sentences using attention-inspired heuristics."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/nyt10m",
        help="Directory that contains nyt10m_<split>.txt files.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Which split to filter. Ignored when --input is provided.",
    )
    parser.add_argument(
        "--input",
        help="Custom input file. Overrides --dataset-root/--split.",
    )
    parser.add_argument(
        "--input-multiple",
        help="Comma-separated list of input files to process sequentially.",
    )
    parser.add_argument(
        "--process-arnor-three",
        action="store_true",
        help="Convenience flag: process the three Arnor files from data/arnor_benchmark.",
    )
    parser.add_argument(
        "--output",
        help="Output path. Defaults to <input_stem>_<policy>_filtered.txt.",
    )
    parser.add_argument(
        "--policy",
        default="topk",
        choices=["absolute", "topk", "relative", "drop", "drop_pct"],
        help="Filtering rule to apply.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Attention threshold for 'absolute' policy.",
    )
    parser.add_argument(
        "--absolute-scale-by-bag",
        dest="absolute_scale_by_bag",
        action="store_true",
        default=False,
        help="Scale absolute threshold by bag size so it represents per-sentence mass (default: off).",
    )
    parser.add_argument(
        "--no-absolute-scale-by-bag",
        dest="absolute_scale_by_bag",
        action="store_false",
        help="Disable bag-size scaling for absolute threshold.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=2,
        help="Number of sentences to keep per bag for 'topk' policy.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Relative margin parameter for 'relative' policy.",
    )
    parser.add_argument(
        "--drop-bottom",
        type=int,
        default=1,
        help="Number of lowest-scoring sentences to drop for 'drop' policy.",
    )
    parser.add_argument(
        "--drop-percent",
        type=float,
        default=0.1,
        help="Fraction (0-1) of lowest sentences to drop for 'drop_pct' policy.",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=1,
        help="Minimum sentences to keep per bag regardless of policy outcome.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Softmax temperature for turning heuristic scores into attention weights.",
    )
    parser.add_argument(
        "--attention-file",
        help="Optional JSONL file with precomputed per-bag attention scores.",
    )
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="If set, append an 'attention_score' field to every kept record.",
    )
    parser.add_argument(
        "--evaluate-noise",
        action="store_true",
        help="If set and the dataset provides 'is_noise', compute filtering precision/recall.",
    )
    parser.add_argument(
        "--report",
        help="Optional path to dump a JSON summary of filtering statistics.",
    )
    parser.add_argument(
        "--plot-scores",
        dest="plot_scores",
        action="store_true",
        default=True,
        help="Automatically plot score distribution for each processed input (default: on).",
    )
    parser.add_argument(
        "--no-plot-scores",
        dest="plot_scores",
        action="store_false",
        help="Disable score distribution plotting.",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=40,
        help="Histogram bins for score plots.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="Image DPI for score plots.",
    )
    parser.add_argument(
        "--plot-zoom-upper",
        type=float,
        default=0.1,
        help="If set, also emit zoomed plots for scores in [0, this value].",
    )
    parser.add_argument(
        "--plot-zoom-bins",
        type=int,
        default=80,
        help="Histogram bins for zoomed score plots.",
    )
    parser.add_argument(
        "--plot-bin-width",
        type=float,
        default=0.005,
        help="Bin width for full-range score plots (overrides --plot-bins if set).",
    )
    parser.add_argument(
        "--plot-zoom-bin-width",
        type=float,
        default=0.0005,
        help="Bin width for zoomed score plots (overrides --plot-zoom-bins if set).",
    )
    parser.add_argument(
        "--limit-bags",
        type=int,
        help="Debug option: only load the first N bags.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(message)s",
    )
    dataset_root = Path(args.dataset_root)

    # Determine list of input files to process. Support single input, multiple inputs,
    # or a convenience flag to process the three Arnor files included in the repo.
    input_paths: List[Path] = []
    if args.process_arnor_three:
        input_paths = [
            Path("data") / "arnor_benchmark" / "arnor_dev_full.txt",
            Path("data") / "arnor_benchmark" / "arnor_test_full.txt",
            Path("data") / "arnor_benchmark" / "arnor_test_noise_full.txt",
        ]
    elif args.input_multiple:
        input_paths = [Path(p.strip()) for p in args.input_multiple.split(",") if p.strip()]
    elif args.input:
        input_paths = [Path(args.input)]
    else:
        # Fallback to single default split file
        input_paths = [dataset_root / f"nyt10m_{args.split}.txt"]

    # Validate files exist
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file {p} does not exist.")

    # Build shared components
    attention_provider = build_attention_provider(args)
    policy = build_policy(args)

    # We'll collect a per-file report including unknown relations counts.
    all_reports: List[Dict[str, Any]] = []

    for input_path in input_paths:
        output_path = Path(args.output) if args.output else input_path.with_name(
            f"{input_path.stem}_{args.policy}_filtered.txt"
        )
        ensure_dir(output_path)

        bags, bag_order = load_dataset(input_path, limit_bags=args.limit_bags)
        stats = StatsTracker(enable_noise_eval=args.evaluate_noise)
        # Collect per-sentence info for this input file so we can dump it to JSON.
        per_file_sentences: List[Dict[str, Any]] = []
        # Track unknown relations by sentence count
        unknown_relations: Counter = Counter()

        LOGGER.info(
            "Filtering %d bags from %s using policy=%s (min_keep=%d)",
            len(bag_order),
            input_path,
            args.policy,
            args.min_keep,
        )
        with output_path.open("w") as writer:
            for bag_id in bag_order:
                bag = bags.get(bag_id, [])
                if not bag:
                    continue
                # Count relation occurrences for unknown-relations tracking
                relation = bag[0].relation if bag else "NA"
                if relation not in RELATION_HINTS:
                    # count sentences in bag toward unknown relation frequency
                    unknown_relations[relation] += len(bag)

                scores = attention_provider.get_scores(bag_id, bag)
                if len(scores) != len(bag):
                    LOGGER.warning(
                        "Bag %s has %d sentences but %d attention scores; falling back to uniform.",
                        bag_id,
                        len(bag),
                        len(scores),
                    )
                    scores = [1.0 / len(bag)] * len(bag)
                mask = policy.select(scores)
                bag_kept = 0
                for sent, keep, att in zip(bag, mask, scores):
                    # Normalize is_noise to a boolean or None
                    noise_label = sent.data.get("is_noise")
                    if isinstance(noise_label, str):
                        lowered = noise_label.lower()
                        if lowered in {"true", "1"}:
                            noise_label_bool = True
                        elif lowered in {"false", "0"}:
                            noise_label_bool = False
                        else:
                            noise_label_bool = None
                    elif isinstance(noise_label, bool):
                        noise_label_bool = noise_label
                    else:
                        noise_label_bool = None

                    # Record per-sentence info for later dump
                    per_file_sentences.append(
                        {
                            "text": sent.text,
                            "relation": sent.relation,
                            "h_name": sent.h_name,
                            "t_name": sent.t_name,
                            "score": float(att),
                            "is_noise": noise_label_bool,
                            "kept": bool(keep),
                        }
                    )

                    stats.record_noise(keep=keep, noise_label=noise_label_bool)
                    if keep:
                        record = dict(sent.data)
                        if args.annotate_attention:
                            record["attention_score"] = round(att, 6)
                        writer.write(json.dumps(record))
                        writer.write("\n")
                        bag_kept += 1
                stats.update(bag[0].relation, len(bag), bag_kept)

        LOGGER.info("Wrote filtered split to %s", output_path)
        summary_text = stats.format_summary()
        print(f"\n=== Filter summary for {input_path} ===\n" + summary_text)

        report: Dict[str, Any] = stats.summary()
        # Add unknown relations detail to report
        report["unknown_relations"] = [
            {"relation": rel, "count": cnt} for rel, cnt in unknown_relations.most_common()
        ]
        all_reports.append({"input": str(input_path), "report": report})

        # Write per-file sentences JSON next to report or output
        if args.report:
            report_path = Path(args.report)
            if report_path.exists() and report_path.is_dir():
                sentences_outf = report_path / f"{input_path.stem}_sentences.json"
            else:
                # if single file target, when multiple inputs we append the stem
                if len(input_paths) > 1:
                    ensure_dir(report_path)
                    sentences_outf = report_path.with_name(report_path.stem + f"_{input_path.stem}").with_suffix(report_path.suffix or ".json")
                else:
                    sentences_outf = report_path
        else:
            sentences_outf = output_path.with_name(f"{input_path.stem}_sentences.json")
        ensure_dir(sentences_outf)
        with sentences_outf.open("w") as f:
            json.dump(per_file_sentences, f, indent=2)

        # Optionally write per-file JSON report next to output or to --report if provided
        if args.report:
            report_path = Path(args.report)
            # if report path is directory, write per-file JSON inside it
            if report_path.exists() and report_path.is_dir():
                outf = report_path / f"filter_summary_{input_path.stem}.json"
            else:
                # if single file target, when multiple inputs we append the stem
                if len(input_paths) > 1:
                    ensure_dir(report_path)
                    outf = report_path.with_name(report_path.stem + f"_{input_path.stem}").with_suffix(report_path.suffix or ".json")
                else:
                    outf = report_path
            ensure_dir(outf)
            with outf.open("w") as f:
                json.dump(report, f, indent=2)

        # Plot score distribution for this input, if requested.
        score_values_all = [float(entry["score"]) for entry in per_file_sentences if "score" in entry]
        score_values_clean = [
            float(entry["score"])
            for entry in per_file_sentences
            if "score" in entry and entry.get("is_noise") is False
        ]
        score_values_noise = [
            float(entry["score"])
            for entry in per_file_sentences
            if "score" in entry and entry.get("is_noise") is True
        ]
        plot_out = output_path.with_name(f"{output_path.stem}_score_distribution.png")
        _plot_scores_if_requested(
            scores={"all": score_values_all, "clean": score_values_clean, "noise": score_values_noise},
            output_path=plot_out,
            input_label=input_path.name,
            enabled=args.plot_scores,
            bins=args.plot_bins,
            dpi=args.plot_dpi,
            zoom_upper=args.plot_zoom_upper,
            zoom_bins=args.plot_zoom_bins,
            policy=args.policy,
            threshold=args.threshold,
            plot_bin_width=args.plot_bin_width,
            plot_zoom_bin_width=args.plot_zoom_bin_width,
        )

    # After processing all files, print a concise per-file unknown-relations summary
    print("\n=== Unknown-relations summary (per file) ===")
    for entry in all_reports:
        inp = entry["input"]
        unk = entry["report"].get("unknown_relations", [])
        print(f"\nFile: {inp}")
        if not unk:
            print("  (All relations have defined rules)")
            continue
        for item in unk:
            print(f"  {item['relation']}: {item['count']} times")


if __name__ == "__main__":
    main()


