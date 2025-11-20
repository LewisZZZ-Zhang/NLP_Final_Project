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
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOGGER = logging.getLogger("nyt10m_filter")


# Relation-specific lexical hints. These are intentionally lightweight and only capture
# the most obvious trigger words so the heuristics remain interpretable and easy to
# extend later with real attention signals.
RELATION_HINTS: Dict[str, Dict[str, Any]] = {
    "default": {
        "keywords": [],
        "between": [" in ", " of ", " near ", " at ", " with "],
        "preferred_order": "any",
        "bias": 0.05,
        "boosts": {},
    },
    "NA": {
        "keywords": [],
        "between": [],
        "bias": -0.05,
    },
    "/location/location/contains": {
        "keywords": [
            " in ",
            " within ",
            " part of ",
            " throughout ",
            " across ",
            " around ",
            " near ",
            " bordered by",
        ],
        "between": [" in ", " of ", ", ", " near ", " along "],
        "bias": 0.25,
    },
    "/people/person/place_lived": {
        "keywords": [
            " lives",
            " lived",
            " living",
            " resides",
            " resident",
            "based in",
            "grew up",
            "settled",
            "home in",
            "raised in",
        ],
        "between": [" in ", " at ", " from ", " near "],
        "bias": 0.32,
        "boosts": {"home": 0.05, "native": 0.05},
    },
    "/people/person/nationality": {
        "keywords": [
            "national",
            "citizen",
            "native",
            "from",
            "born",
            "hails from",
            "ethnicity",
        ],
        "between": [" of ", " from "],
        "bias": 0.3,
        "boosts": {
            "american": 0.02,
            "chinese": 0.02,
            "french": 0.02,
            "british": 0.02,
        },
    },
    "/business/person/company": {
        "keywords": [
            "chief",
            "executive",
            "works at",
            "employee",
            "president",
            "chairman",
            "ceo",
            "director",
            "lead",
            "co-founder",
        ],
        "between": [" at ", " with ", " of "],
        "preferred_order": "h-first",
        "bias": 0.28,
    },
    "/location/neighborhood/neighborhood_of": {
        "keywords": [
            "neighborhood",
            "district",
            "borough",
            "section",
            "area",
            "part of",
        ],
        "between": [" in ", " of ", ", "],
        "bias": 0.25,
    },
    "/location/country/administrative_divisions": {
        "keywords": [
            "province",
            "state",
            "district",
            "region",
            "municipality",
            "prefecture",
        ],
        "between": [" of ", " in "],
        "preferred_order": "t-first",
        "bias": 0.27,
    },
    "/location/country/capital": {
        "keywords": ["capital", "seat of", "capital city"],
        "between": [" of ", " for "],
        "bias": 0.4,
        "preferred_order": "t-first",
    },
    "/location/administrative_division/country": {
        "keywords": ["country", "nation", "republic", "state"],
        "between": [" in ", " of "],
        "preferred_order": "h-first",
        "bias": 0.25,
    },
    "/people/person/place_of_birth": {
        "keywords": ["born", "birthplace", "delivery"],
        "between": [" in ", " at "],
        "bias": 0.4,
        "boosts": {"born": 0.2, "birth": 0.1, "delivered": 0.1},
    },
    "/people/deceasedperson/place_of_death": {
        "keywords": ["died", "dead", "killed", "death", "passed away"],
        "between": [" in ", " at "],
        "bias": 0.4,
        "boosts": {"died": 0.2, "killed": 0.15, "dead": 0.1},
    },
    "/location/region/capital": {
        "keywords": ["capital", "seat"],
        "between": [" of ", " for "],
        "bias": 0.35,
    },
    "/business/company/founders": {
        "keywords": ["founded", "founder", "co-founder", "established", "started"],
        "between": [" by ", " from "],
        "preferred_order": "t-first",
        "bias": 0.35,
    },
    "/people/person/children": {
        "keywords": ["son", "daughter", "child", "children", "kids"],
        "between": [" and ", ", ", " with "],
        "bias": 0.32,
    },
    "/business/company/place_founded": {
        "keywords": ["founded", "established", "formed", "launched"],
        "between": [" in ", " at "],
        "preferred_order": "h-first",
        "bias": 0.33,
    },
    "/business/location": {
        "keywords": ["based", "headquartered", "office", "located", "operates"],
        "between": [" in ", " at "],
        "bias": 0.3,
    },
    "/business/company/majorshareholders": {
        "keywords": ["shareholder", "owns", "stake", "holding", "owner"],
        "between": [" in ", " of "],
        "bias": 0.35,
    },
    "/people/person/religion": {
        "keywords": [
            "religion",
            "faith",
            "catholic",
            "christian",
            "muslim",
            "jewish",
            "buddhist",
            "hindu",
        ],
        "between": [" of ", " in "],
        "bias": 0.3,
    },
    "/people/person/ethnicity": {
        "keywords": ["ethnicity", "ethnic", "heritage", "ancestry", "roots"],
        "between": [" of ", " from "],
        "bias": 0.28,
    },
    "/business/company/advisors": {
        "keywords": ["advisor", "adviser", "consultant", "counsel"],
        "between": [" to ", " for "],
        "bias": 0.3,
    },
    "/people/ethnicity/geographic_distribution": {
        "keywords": ["community", "population", "diaspora", "settled", "live"],
        "between": [" in ", " across ", " throughout "],
        "bias": 0.22,
    },
    "/location/us_county/county_seat": {
        "keywords": ["county seat"],
        "between": [" of ", " for "],
        "bias": 0.4,
    },
    "/people/deceasedperson/place_of_burial": {
        "keywords": ["buried", "burial", "interred", "laid to rest"],
        "between": [" in ", " at "],
        "bias": 0.38,
    },
    "/film/film/featured_film_locations": {
        "keywords": ["filmed", "shot", "set in", "location"],
        "between": [" in ", " at "],
        "bias": 0.35,
    },
    "/time/event/locations": {
        "keywords": ["held", "hosted", "took place", "took place in", "took place at"],
        "between": [" in ", " at "],
        "bias": 0.33,
    },
}


def make_bag_id(relation: str, head_id: str, tail_id: str) -> str:
    return f"{relation}||{head_id}||{tail_id}"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


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


@dataclass
class SentenceInstance:
    relation: str
    text: str
    text_lower: str
    h_name: str
    t_name: str
    h_pos: Tuple[int, int]
    t_pos: Tuple[int, int]
    data: Dict[str, Any]
    _between_cache: Optional[str] = field(default=None, init=False, repr=False)

    def between_text(self) -> str:
        if self._between_cache is None:
            h_start, h_end = self.h_pos
            t_start, t_end = self.t_pos
            if h_start <= t_start:
                left, right = h_end, t_start
            else:
                left, right = t_end, h_start
            left = max(0, min(len(self.text), left))
            right = max(0, min(len(self.text), right))
            if left >= right:
                self._between_cache = ""
            else:
                self._between_cache = self.text_lower[left:right]
        return self._between_cache


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
            head = record.get("h", {})
            tail = record.get("t", {})
            text = record.get("text", "")
            bag_id = make_bag_id(
                relation,
                str(head.get("id", head.get("name", ""))),
                str(tail.get("id", tail.get("name", ""))),
            )
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


class AttentionProvider:
    def get_scores(self, bag_id: str, sentences: Sequence[SentenceInstance]) -> List[float]:
        raise NotImplementedError


class HeuristicAttentionProvider(AttentionProvider):
    def __init__(self, temperature: float = 0.7):
        self.temperature = max(temperature, 1e-3)
        self._hint_cache: Dict[str, Dict[str, Any]] = {}

    def get_scores(self, bag_id: str, sentences: Sequence[SentenceInstance]) -> List[float]:
        if not sentences:
            return []
        raw_scores = [self._score_sentence(sent) for sent in sentences]
        if len(raw_scores) == 1:
            return [1.0]
        max_score = max(raw_scores)
        if max_score <= 0:
            uniform = 1.0 / len(raw_scores)
            return [uniform for _ in raw_scores]
        exps = [math.exp((score - max_score) / self.temperature) for score in raw_scores]
        denom = sum(exps)
        if denom == 0:
            uniform = 1.0 / len(raw_scores)
            return [uniform for _ in raw_scores]
        return [val / denom for val in exps]

    def _score_sentence(self, instance: SentenceInstance) -> float:
        hint = self._get_hint(instance.relation)
        dist = self._distance_score(instance)
        between = self._between_score(instance)
        keyword = self._keyword_score(instance, hint)
        order = self._order_score(instance, hint)
        bias = hint.get("bias", RELATION_HINTS["default"]["bias"])
        boosts = hint.get("boosts", {})
        extra = sum(weight for token, weight in boosts.items() if token in instance.text_lower)
        raw = (
            0.35 * dist
            + 0.25 * between
            + 0.3 * keyword
            + 0.1 * order
            + bias
            + extra
        )
        if instance.relation == "NA":
            raw *= 0.4
        return clamp(raw, 0.0, 1.5)

    def _distance_score(self, instance: SentenceInstance) -> float:
        text_len = max(len(instance.text), 1)
        h_center = (instance.h_pos[0] + instance.h_pos[1]) / 2
        t_center = (instance.t_pos[0] + instance.t_pos[1]) / 2
        dist = abs(h_center - t_center) / text_len
        return clamp(1.0 - dist * 1.3)

    def _between_score(self, instance: SentenceInstance) -> float:
        between = instance.between_text()
        if not between:
            return 1.0
        token_count = max(1, len(between.split()))
        # Encourage concise contexts; clamp after ~35 words.
        return clamp(1.0 - (token_count / 35.0))

    def _keyword_score(self, instance: SentenceInstance, hint: Dict[str, Any]) -> float:
        text_lower = instance.text_lower
        between = instance.between_text()
        keywords = hint.get("keywords", [])
        between_keywords = hint.get("between", [])
        score = 0.0
        weight = 0.0
        for kw in keywords:
            weight += 1.0
            if kw in text_lower:
                score += 1.0
        for kw in between_keywords:
            weight += 0.5
            if kw in between:
                score += 0.5
        if weight == 0:
            return 0.0
        return clamp(score / weight)

    def _order_score(self, instance: SentenceInstance, hint: Dict[str, Any]) -> float:
        preferred = hint.get("preferred_order", "any")
        if preferred == "any":
            return 1.0
        h_first = instance.h_pos[0] <= instance.t_pos[0]
        expect_head = preferred == "h-first"
        return 1.0 if h_first == expect_head else 0.5

    def _get_hint(self, relation: str) -> Dict[str, Any]:
        if relation not in self._hint_cache:
            merged = {**RELATION_HINTS["default"]}
            specific = RELATION_HINTS.get(relation)
            if specific:
                for key, value in specific.items():
                    if isinstance(value, list):
                        merged[key] = list(value)
                    elif isinstance(value, dict):
                        merged[key] = dict(value)
                    else:
                        merged[key] = value
            self._hint_cache[relation] = merged
        return self._hint_cache[relation]


class ExternalAttentionProvider(AttentionProvider):
    """
    Optional attention provider that loads per-bag attention scores from disk.

    Expected file format: one JSON object per line with either
        {"bag_id": "<relation>||<h_id>||<t_id>", "scores": [0.2, 0.8, ...]}
    or
        {"bag_id": "...", "attentions": [...]}
    Extra fields are ignored. Bags without entries fall back to the heuristic provider.
    """

    def __init__(self, path: Path, fallback: Optional[AttentionProvider] = None):
        self._scores: Dict[str, List[float]] = {}
        self._fallback = fallback or HeuristicAttentionProvider()
        LOGGER.info("Loading external attention from %s", path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bag_id = payload.get("bag_id")
                scores = payload.get("scores") or payload.get("attentions")
                if not bag_id or not isinstance(scores, list):
                    continue
                self._scores[bag_id] = [float(x) for x in scores]
        LOGGER.info("Loaded attention for %d bags", len(self._scores))

    def get_scores(self, bag_id: str, sentences: Sequence[SentenceInstance]) -> List[float]:
        cached = self._scores.get(bag_id)
        if cached and len(cached) == len(sentences):
            total = sum(cached)
            if total <= 0:
                return self._fallback.get_scores(bag_id, sentences)
            return [score / total for score in cached]
        return self._fallback.get_scores(bag_id, sentences)


class FilterPolicy:
    def __init__(self, min_keep: int = 1):
        self.min_keep = max(1, min_keep)

    def select(self, scores: Sequence[float]) -> List[bool]:
        raise NotImplementedError


class AbsoluteThresholdPolicy(FilterPolicy):
    def __init__(self, threshold: float, min_keep: int = 1):
        super().__init__(min_keep)
        self.threshold = threshold

    def select(self, scores: Sequence[float]) -> List[bool]:
        mask = [score >= self.threshold for score in scores]
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
        return AbsoluteThresholdPolicy(threshold=args.threshold, min_keep=args.min_keep)
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
    input_path = Path(args.input) if args.input else dataset_root / f"nyt10m_{args.split}.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}_{args.policy}_filtered.txt"
    )
    ensure_dir(output_path)
    bags, bag_order = load_dataset(input_path, limit_bags=args.limit_bags)
    attention_provider = build_attention_provider(args)
    policy = build_policy(args)
    stats = StatsTracker(enable_noise_eval=args.evaluate_noise)
    LOGGER.info(
        "Filtering %d bags using policy=%s (min_keep=%d)",
        len(bag_order),
        args.policy,
        args.min_keep,
    )
    with output_path.open("w") as writer:
        for bag_id in bag_order:
            bag = bags.get(bag_id, [])
            if not bag:
                continue
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
    print("\n=== Filter summary ===\n" + summary_text)
    if args.report:
        ensure_dir(Path(args.report))
        with Path(args.report).open("w") as f:
            json.dump(stats.summary(), f, indent=2)


if __name__ == "__main__":
    main()


# 使用例：
# python3 filter/main.py --split train --limit-bags 5 --policy topk --topk 1 --output tmp/nyt10m_train_filtered_sample.txt
# python3 filter/main.py --split train --limit-bags 2 --policy drop --drop-bottom 1 --output tmp/nyt10m_train_filtered_sample.txt
# --policy topk --topk N
# --policy absolute --threshold T
# --policy relative --margin M
# --min-keep
# --policy drop --drop-bottom K
# python3 filter/main.py --input data/arnor_test.txt --policy drop --drop-bottom 10 --limit-bags 10 --evaluate-noise --output tmp/arnor_filtered.txt
# python3 filter/main.py --input data/arnor_test.txt --policy drop_pct --drop-percent 0.5 --limit-bags 10 --evaluate-noise --output tmp/arnor_filtered.txt
# python3 filter/main.py --input data/arnor_test.txt --policy drop_pct --drop-percent 0.1 --evaluate-noise --output tmp/arnor_filtered_full.txt

# TP=40（True Positive）：确实是噪声并被过滤掉的句子，说明过滤器成功删掉 40 条标注为噪声的样本。
# FP=164（False Positive）：原本是干净句子却被误删的 164 条，越少越好。
# FN=219（False Negative）：原本是噪声但被保留下来的 219 条，意味着还有很多噪声没被过滤。
# TN=627（True Negative）：干净句子且被保留下来的 627 条。