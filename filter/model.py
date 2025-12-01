#!/usr/bin/env python3
"""Attention and scoring components for NYT10m filtering."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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

def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))

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
            def find_mention_span(name: str) -> Optional[Tuple[int,int]]:
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

