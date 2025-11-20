#!/usr/bin/env python3
import argparse
import json
import sys


def analyze(path):
    total = 0
    na_count = 0
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip lines that look like comments
            if line.startswith("/*") or line.startswith("//"):
                continue
            # try to find a JSON object in the line
            try:
                obj = json.loads(line)
            except Exception:
                # try to salvage if line contains json after some prefix
                # find first '{' and last '}'
                try:
                    i = line.index('{')
                    j = line.rindex('}')
                    json_part = line[i:j+1]
                    obj = json.loads(json_part)
                except Exception:
                    parse_errors += 1
                    continue
            total += 1
            rel = obj.get('relation') if isinstance(obj, dict) else None
            if rel is None:
                # some files use uppercase/lowercase key variants
                rel = obj.get('Relation') if isinstance(obj, dict) else None
            if isinstance(rel, str) and rel.strip().lower() == 'na':
                na_count += 1
    return total, na_count, parse_errors

def main():
    parser = argparse.ArgumentParser(description='Count relation==NA ratios in JSONL-like files')
    parser.add_argument('paths', nargs='*', help='one or more file paths to analyze')
    args = parser.parse_args()

    default_paths = [
        "data/arnor_test.txt",
        "OpenNRE/benchmark/nyt10m/nyt10m_test.txt",
    ]
    targets = args.paths if args.paths else default_paths

    for p in targets:
        try:
            total, na_count, parse_errors = analyze(p)
        except FileNotFoundError:
            print(f"File not found: {p}")
            continue
        ratio = (na_count / total) if total else 0.0
        print(f"File: {p}")
        print(f"  Total parsed lines: {total}")
        print(f"  NA count: {na_count}")
        print(f"  NA ratio: {ratio:.4f} ({ratio*100:.2f}%)")
        if parse_errors:
            print(f"  Parse errors/skipped lines: {parse_errors}")
        print()


if __name__ == '__main__':
    main()
