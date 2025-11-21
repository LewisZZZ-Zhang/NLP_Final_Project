#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Arnor dataset (data/arnor_dataset-2.0.0/*.json) to OpenNRE line-by-line format.

Behavior:
- For each sentence in the Arnor file, emit one OpenNRE-style record per relation mention.
- Preserve the `is_noise` flag from Arnor in the output record as `is_noise`.
- Normalize relation labels: map None/"None"/"none"/'' -> "NA"; otherwise keep label text.
- Build `rel2id` automatically with `"NA": 0` and other relations starting from 1.

Output (by default):
- OpenNRE/benchmark/converted_arnor/arnor_test.txt  (one JSON dict per line)
- OpenNRE/benchmark/converted_arnor/arnor_rel2id.json

Usage:
    python3 OpenNRE/scripts/convert_arnor_to_opennre.py
or with explicit paths:
    python3 OpenNRE/scripts/convert_arnor_to_opennre.py \
        --input data/arnor_dataset-2.0.0/test.json \
        --out OpenNRE/benchmark/converted_arnor/arnor_test.txt \
        --rel2id OpenNRE/benchmark/converted_arnor/arnor_rel2id.json
"""
import json
import argparse
import re
from pathlib import Path
from collections import OrderedDict, defaultdict


def find_span(sentence, entity_text):
    """Locate character span (start, end) of entity_text in sentence.
    Returns (start, end) or None if not found.
    Strategy:
    1) direct find
    2) case-insensitive
    3) whitespace-normalized
    4) token-sequence approximate
    """
    if not entity_text:
        return None
    # direct
    idx = sentence.find(entity_text)
    if idx >= 0:
        return (idx, idx + len(entity_text))
    # case-insensitive
    low_s = sentence.lower()
    low_e = entity_text.lower()
    idx = low_s.find(low_e)
    if idx >= 0:
        return (idx, idx + len(entity_text))
    # collapse whitespace
    s_norm = re.sub(r"\s+", " ", sentence)
    e_norm = re.sub(r"\s+", " ", entity_text)
    idx = s_norm.find(e_norm)
    if idx >= 0:
        # map back approximately by searching the subsequence
        return (idx, idx + len(e_norm))
    # token sequence match
    sent_tokens = sentence.split()
    ent_tokens = entity_text.split()
    if len(ent_tokens) == 0:
        return None
    for i in range(len(sent_tokens) - len(ent_tokens) + 1):
        if [t.strip(".,;:'\"()[]{}") for t in sent_tokens[i:i+len(ent_tokens)]] == [t.strip(".,;:'\"()[]{}") for t in ent_tokens]:
            seq = " ".join(sent_tokens[i:i+len(ent_tokens)])
            idx = sentence.find(seq)
            if idx >= 0:
                return (idx, idx + len(seq))
    return None


def normalize_label(lab):
    if lab is None:
        return "NA"
    s = str(lab).strip()
    if s == "" or s.lower() in ("none", "null"):
        return "NA"
    return s


def ensure_dir(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def convert_file(input_path, out_path, rel2id_path):
    rels = OrderedDict()
    rels['NA'] = None

    out_lines = []
    # noise_counts[relation] = {'true': int, 'false': int}
    noise_counts = defaultdict(lambda: {'true': 0, 'false': 0})
    with open(input_path, 'r', encoding='utf-8') as rf:
        for raw in rf:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                # try to be permissive: skip if can't parse
                continue

            sent = obj.get('sentText') or obj.get('sent') or obj.get('sentText')
            if sent is None:
                # try alternative keys
                sent = obj.get('sentText', '')

            relation_mentions = obj.get('relationMentions', []) or []
            # For each relation mention create one OpenNRE-style record
            for ridx, rm in enumerate(relation_mentions):
                h_text = rm.get('em1Text') or rm.get('em1') or ''
                t_text = rm.get('em2Text') or rm.get('em2') or ''
                label = normalize_label(rm.get('label'))
                is_noise = rm.get('is_noise', False)

                # find spans
                h_span = find_span(sent, h_text)
                t_span = find_span(sent, t_text)

                if h_span is None:
                    # fallback: empty span and warn
                    # keep processing so downstream rel2id contains this label
                    # but user should inspect warnings
                    # Note: printing here is helpful during conversion runs
                    print(f"Warning: head entity '{h_text}' not found in sentence (truncated): {sent[:120]}")
                    h_span = (0, 0)
                if t_span is None:
                    print(f"Warning: tail entity '{t_text}' not found in sentence (truncated): {sent[:120]}")
                    t_span = (0, 0)

                item = {
                    'text': sent,
                    'h': {'pos': [h_span[0], h_span[1]], 'id': f'h_{ridx}', 'name': h_text},
                    't': {'pos': [t_span[0], t_span[1]], 'id': f't_{ridx}', 'name': t_text},
                    'relation': label,
                    'is_noise': bool(is_noise)
                }
                out_lines.append(item)
                # update noise counts for this relation label
                if bool(is_noise):
                    noise_counts[label]['true'] += 1
                else:
                    noise_counts[label]['false'] += 1
                if label not in rels:
                    rels[label] = None

    # build rel2id (NA -> 0)
    rel2id = {}
    cur = 0
    if 'NA' in rels:
        rel2id['NA'] = 0
        cur = 1
    for r in rels:
        if r == 'NA':
            continue
        rel2id[r] = cur
        cur += 1

    # write output
    ensure_dir(Path(out_path))
    with open(out_path, 'w', encoding='utf-8') as wf:
        for it in out_lines:
            wf.write(json.dumps(it, ensure_ascii=False) + '\n')

    ensure_dir(Path(rel2id_path))
    with open(rel2id_path, 'w', encoding='utf-8') as rf:
        json.dump(rel2id, rf, indent=2, ensure_ascii=False)

    # write noise statistics summary
    noise_stats_path = rel2id_path.replace('_rel2id.json', '_noise_stats.json')
    try:
        ensure_dir(Path(noise_stats_path))
        with open(noise_stats_path, 'w', encoding='utf-8') as nsf:
            json.dump(noise_counts, nsf, indent=2, ensure_ascii=False)
        print(f"Wrote noise stats to {noise_stats_path}")
    except Exception:
        # If writing fails, still print summary
        noise_stats_path = None

    # print a concise summary to stdout
    print("Noise counts per relation:")
    for r in sorted(noise_counts.keys()):
        print(f"  {r}: true={noise_counts[r]['true']}, false={noise_counts[r]['false']}")

    print(f"Wrote {len(out_lines)} examples to {out_path}")
    print(f"Wrote {len(rel2id)} relations to {rel2id_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/arnor_dataset-2.0.0/test.json')
    parser.add_argument('--out', default='OpenNRE/benchmark/converted_arnor/arnor_test.txt')
    parser.add_argument('--rel2id', default='OpenNRE/benchmark/converted_arnor/arnor_rel2id.json')
    args = parser.parse_args()

    convert_file(args.input, args.out, args.rel2id)


if __name__ == '__main__':
    main()
