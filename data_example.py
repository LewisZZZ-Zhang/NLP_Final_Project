#!/usr/bin/env python3
"""washdata.py

Small helper to print example records from the dataset files in `data/`.

Usage: python washdata.py

It will read the following files (if present):
  - data/test.json
  - data/valid.json
  - data/train.json

For each file it prints up to 3 example JSON records with a short summary
of the top-level keys so you can inspect the structure.
"""
import json
from pathlib import Path
import sys


DATA_DIR = Path(__file__).resolve().parent / "data"


def read_jsonl(path, max_examples=3):
	"""Read up to max_examples JSON objects from a JSONL file.

	Returns a list of parsed Python dicts.
	If the file is not found, returns an empty list.
	"""
	if not path.exists():
		return []
	examples = []
	with path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			if i >= max_examples:
				break
			line = line.strip()
			if not line:
				continue
			try:
				examples.append(json.loads(line))
			except json.JSONDecodeError:
				# skip malformed lines but continue
				continue
	return examples


def summarize_record(rec):
	"""Return a short summary string for a record's top-level keys and some sizes."""
	if not isinstance(rec, dict):
		return str(type(rec))
	keys = list(rec.keys())
	parts = [f"keys={keys}"]
	# common fields in this dataset: sentText, relationMentions, entityMentions
	if "sentText" in rec:
		parts.append(f"sentText_len={len(rec.get('sentText') or '')}")
	if "relationMentions" in rec:
		parts.append(f"#relationMentions={len(rec.get('relationMentions') or [])}")
	if "entityMentions" in rec:
		parts.append(f"#entityMentions={len(rec.get('entityMentions') or [])}")
	return ", ".join(parts)


def print_examples_for(file_name, max_examples=3):
	path = DATA_DIR / file_name
	examples = read_jsonl(path, max_examples=max_examples)
	print(f"\n--- {file_name} ({path if path.exists() else 'MISSING'}) ---")
	if not examples:
		print("(no examples — file missing or empty)")
		return
	for i, ex in enumerate(examples, start=1):
		print(f"\nExample #{i} summary: {summarize_record(ex)}")
		# pretty-print the JSON but keep non-ascii characters readable
		try:
			pretty = json.dumps(ex, indent=2, ensure_ascii=False)
		except Exception:
			pretty = str(ex)
		print(pretty)


def main():
	files = ["test.json", "valid.json", "train.json"]
	for fn in files:
		print_examples_for(fn, max_examples=2)


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(1)

