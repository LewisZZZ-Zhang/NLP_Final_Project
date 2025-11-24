#!/usr/bin/env python3
"""
Convert Arnor dataset to OpenNRE/NYT10m-style JSONL files.

Behavior:
- Runs `scripts/find_common_relations.py --save-json <out>/common_relations.json` to compute common relations.
- For each JSON/JSONL split in `data/arnor_dataset-2.0.0` (e.g., train.json, dev.json, test.json, test_noise.json), expand each relation mention into a separate JSON line with keys:
  - `sentId`, `articleId`, `sentText`, `relation`, `is_noise` (if present), `em1Text`, `em2Text`, `entityMentions` (if present)
- Write two output files for each split into `data/arnor_benchmark` (or `--out-dir`):
  - Common-only: `arnor_<SPLIT>.txt` (contains only entries whose relation is in the intersection with NYT10m, includes `NA` if present)
  - Full: `arnor_<SPLIT>_full.txt` (contains all converted entries)

Usage:
  python3 scripts/convert_arnor_to_nyt10m.py
  python3 scripts/convert_arnor_to_nyt10m.py --arnor-dir data/arnor_dataset-2.0.0 --out-dir data/arnor_benchmark

"""
import argparse
import json
import os
import subprocess
import sys
from typing import List


def run_find_common(out_json: str):
    cmd = [sys.executable, "scripts/find_common_relations.py", "--save-json", out_json]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("find_common_relations.py failed")


def normalize_relation(r: str) -> str:
    if r is None:
        return "na"
    return r.strip().lstrip('/').lower()


def load_json_lines_or_array(path: str) -> List[dict]:
    items = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # try to parse as JSON array first
        text = f.read()
        text_strip = text.lstrip()
        if text_strip.startswith('['):
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        # else fallback to JSONL
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except Exception:
                # ignore malformed lines
                continue
    return items


def convert_split(path: str, common_norm_set: set):
    entries_full = []
    items = load_json_lines_or_array(path)
    for obj in items:
        # If the Arnor object already is per-mention (has 'relation' or 'label') handle that
        rms = obj.get('relationMentions')
        if isinstance(rms, list) and len(rms) > 0:
            for i, rm in enumerate(rms):
                rel = rm.get('label') or rm.get('relation') or rm.get('rel')
                if rel is None or (isinstance(rel, str) and rel.strip().lower() in ('none','null','')):
                    rel = 'NA'
                out = {
                    'sentId': obj.get('sentId') or obj.get('sentId', '') + f"_{i}",
                    'articleId': obj.get('articleId'),
                    'sentText': obj.get('sentText') or obj.get('sentText', obj.get('text')),
                    'relation': rel,
                }
                # preserve is_noise at relation mention level if present
                if 'is_noise' in rm:
                    out['is_noise'] = rm.get('is_noise')
                elif 'is_noise' in obj:
                    out['is_noise'] = obj.get('is_noise')
                # keep em1/em2
                if 'em1Text' in rm:
                    out['em1Text'] = rm.get('em1Text')
                if 'em2Text' in rm:
                    out['em2Text'] = rm.get('em2Text')
                if 'entityMentions' in obj:
                    out['entityMentions'] = obj.get('entityMentions')
                entries_full.append(out)
        else:
            # no relationMentions: look for top-level relation or label
            rel = obj.get('relation') or obj.get('label')
            if rel is None or (isinstance(rel, str) and rel.strip().lower() in ('none','null','')):
                rel = 'NA'
            out = {
                'sentId': obj.get('sentId'),
                'articleId': obj.get('articleId'),
                'sentText': obj.get('sentText') or obj.get('text'),
                'relation': rel,
            }
            if 'is_noise' in obj:
                out['is_noise'] = obj.get('is_noise')
            if 'entityMentions' in obj:
                out['entityMentions'] = obj.get('entityMentions')
            entries_full.append(out)

    return entries_full


def write_jsonl(path: str, entries: List[dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for obj in entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description='Convert Arnor dataset to NYT10m-style JSONL (two versions).')
    parser.add_argument('--arnor-dir', default='data/arnor_dataset-2.0.0', help='Path to Arnor dataset folder')
    parser.add_argument('--out-dir', default='data/arnor_benchmark', help='Where to write converted files')
    parser.add_argument('--run-find-common', action='store_true', help='Force run find_common_relations even if JSON exists')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    common_json = os.path.join(args.out_dir, 'common_relations.json')
    # run find_common_relations to ensure up-to-date mapping
    if args.run_find_common or not os.path.exists(common_json):
        run_find_common(common_json)

    with open(common_json, 'r', encoding='utf-8') as f:
        common_data = json.load(f)

    common_list = common_data.get('common_both', [])
    # build normalized set for tolerant matching
    common_norm_set = set(normalize_relation(r) for r in common_list)

    # find splits in arnor dir
    for fname in os.listdir(args.arnor_dir):
        if not fname.lower().endswith(('.json', '.jsonl', '.txt')):
            continue
        split = os.path.splitext(fname)[0]  # e.g., train, dev, test, test_noise
        in_path = os.path.join(args.arnor_dir, fname)
        print(f"Converting split {split} -> {in_path}")
        entries_full = convert_split(in_path, common_norm_set)

        # full file (suffix)
        out_full = os.path.join(args.out_dir, f'arnor_{split}_full.txt')
        write_jsonl(out_full, entries_full)

        # common-only
        entries_common = []
        for e in entries_full:
            rel = e.get('relation')
            if normalize_relation(rel) in common_norm_set:
                entries_common.append(e)

        # name requested: arnor_TRAIN/TEST/... .txt (we'll use lowercase split but prefix 'arnor_')
        out_common = os.path.join(args.out_dir, f'arnor_{split}.txt')
        write_jsonl(out_common, entries_common)

        print(f"Wrote {len(entries_full)} entries -> {out_full}")
        print(f"Wrote {len(entries_common)} common entries -> {out_common}\n")


if __name__ == '__main__':
    main()
