#!/usr/bin/env python3
"""
Find and list relation vocabularies in the Arnor dataset and NYT10m benchmark,
then report the overlap (common relations).

Usage examples:
  python3 scripts/find_common_relations.py
  python3 scripts/find_common_relations.py --arnor-dir data/arnor_dataset-2.0.0 \
      --nyt-rel2id OpenNRE/benchmark/nyt10m/nyt10m_rel2id.json \
      --save-json data/arnor_benchmark/common_relations.json
"""
import argparse
import json
import os
import sys
from typing import Set


def _collect_from_obj(obj, rels: Set[str]):
    if not isinstance(obj, dict):
        return
    # Arnor format: relationMentions -> list of {label: ...}
    rms = obj.get("relationMentions")
    if isinstance(rms, list):
        for rm in rms:
            if not isinstance(rm, dict):
                continue
            label = rm.get("label") or rm.get("relation") or rm.get("rel")
            if not label:
                continue
            lab = label.strip()
            if lab.lower() in ("none", "null", ""):
                lab = "NA"
            rels.add(lab)

    # fallback: top-level relation field
    if isinstance(obj.get("relation"), str):
        lab = obj["relation"].strip()
        if lab.lower() in ("none", "null", ""):
            lab = "NA"
        rels.add(lab)


def extract_arnor_relations(arnor_dir: str) -> Set[str]:
    rels = set()
    if not os.path.isdir(arnor_dir):
        raise FileNotFoundError(f"Arnor directory not found: {arnor_dir}")

    for root, _, files in os.walk(arnor_dir):
        for fname in files:
            if not fname.lower().endswith((".json", ".jsonl", ".txt")):
                # still try common text/json extensions
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Many Arnor files are JSONL (one JSON object per line)
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # maybe the file is a JSON array - try to parse whole file once
                            try:
                                f.seek(0)
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        _collect_from_obj(item, rels)
                                break
                            except Exception:
                                continue

                        if isinstance(obj, list):
                            for item in obj:
                                _collect_from_obj(item, rels)
                        else:
                            _collect_from_obj(obj, rels)
            except Exception as e:
                print(f"Warning: failed reading {path}: {e}", file=sys.stderr)
    return rels


def extract_nyt_relations(rel2id_path: str, nyt_dir: str) -> Set[str]:
    rels = set()
    # Prefer rel2id mapping if available
    if rel2id_path and os.path.exists(rel2id_path):
        try:
            with open(rel2id_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in data.keys():
                    rels.add(k)
                return rels
        except Exception as e:
            print(f"Warning: failed to read rel2id file {rel2id_path}: {e}", file=sys.stderr)

    # Fallback: parse TXT files under nyt_dir (JSONL expected)
    if nyt_dir and os.path.isdir(nyt_dir):
        for root, _, files in os.walk(nyt_dir):
            for fname in files:
                if not fname.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                rel = obj.get("relation") or obj.get("label")
                                if rel:
                                    lab = rel.strip()
                                    if lab.lower() in ("none", "null", ""):
                                        lab = "NA"
                                    rels.add(lab)
                            except Exception:
                                continue
                except Exception as e:
                    print(f"Warning: failed reading {path}: {e}", file=sys.stderr)
    return rels


def save_common_json(out_path: str, arnor_rels: Set[str], nyt_rels: Set[str], common: Set[str]):
    payload = {
        "arnor_common": sorted(list(arnor_rels)),
        "nyt10m_common": sorted(list(nyt_rels)),
        "common_both": sorted(list(common)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Find relations in Arnor and NYT10m and report overlap")
    parser.add_argument("--arnor-dir", default="data/arnor_dataset-2.0.0",
                        help="Path to the Arnor dataset folder")
    parser.add_argument("--nyt-rel2id", default="OpenNRE/benchmark/nyt10m/nyt10m_rel2id.json",
                        help="Path to nyt10m_rel2id.json (preferred)")
    parser.add_argument("--nyt-dir", default="OpenNRE/benchmark/nyt10m",
                        help="Fallback directory containing nyt10m_*.txt files")
    parser.add_argument("--save-json", default="",
                        help="Optional path to save the JSON summary (arnor/nyt/common)")
    args = parser.parse_args()

    print(f"Scanning Arnor directory: {args.arnor_dir}")
    arnor_rels = extract_arnor_relations(args.arnor_dir)
    print(f"Found {len(arnor_rels)} unique relations in Arnor")

    print(f"Scanning NYT10m (rel2id: {args.nyt_rel2id}, dir: {args.nyt_dir})")
    nyt_rels = extract_nyt_relations(args.nyt_rel2id, args.nyt_dir)
    print(f"Found {len(nyt_rels)} unique relations in NYT10m")

    common = arnor_rels.intersection(nyt_rels)
    print(f"Common relations: {len(common)}\n")

    # Print small summaries
    def _print_sample(title, items):
        print(f"--- {title} ({len(items)}) ---")
        for it in sorted(items):
            print(it)
        print()

    _print_sample("Arnor relations", arnor_rels)
    _print_sample("NYT10m relations", nyt_rels)
    _print_sample("Common relations", common)

    if args.save_json:
        save_common_json(args.save_json, arnor_rels, nyt_rels, common)


if __name__ == "__main__":
    main()
