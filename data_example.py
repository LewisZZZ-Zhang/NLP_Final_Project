'''
python3 data_example.py | sed -n '1,200p' > out/example.txt
'''

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


DATA_DIR = Path(__file__).resolve().parent / "data" / "NTYRED"


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
	# prefer showing train first, then valid, then test
	files = ["train.json", "valid.json", "test.json"]

	# Print a single example from the first available file (prefer test.json)
	example_printed = False
	for fn in files:
		path = DATA_DIR / fn
		exs = read_jsonl(path, max_examples=1)
		if exs:
			ex = exs[0]
			print(f"\n--- Example (from {fn}) ---")
			print(f"Summary: {summarize_record(ex)}")
			print(json.dumps(ex, indent=2, ensure_ascii=False))
			example_printed = True
			break
	if not example_printed:
		print("\n(No example found — none of test.json / valid.json / train.json exist or contain records.)")

	# Now print counts for each split
	print("\n--- Dataset counts ---")
	for fn in files:
		path = DATA_DIR / fn
		count = 0
		if path.exists():
			# count non-empty lines (attempt to ignore blank lines)
			with path.open("r", encoding="utf-8") as f:
				for line in f:
					if line.strip():
						count += 1
		print(f"{fn}: {count}")

	# Collect all relation labels across the splits and count frequencies
	rel_counts = {}
	# Also collect per-file relation counts
	rel_counts_per_file = {fn: {} for fn in files}
	for fn in files:
		path = DATA_DIR / fn
		if not path.exists():
			continue
		with path.open("r", encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				try:
					rec = json.loads(line)
				except json.JSONDecodeError:
					continue
				for rm in rec.get("relationMentions", []) or []:
					label = rm.get("label")
					if not label:
						continue
					rel_counts[label] = rel_counts.get(label, 0) + 1
					rel_counts_per_file[fn][label] = rel_counts_per_file[fn].get(label, 0) + 1

	print("\n--- Relation types (combined label: count) ---")
	if not rel_counts:
		print("(no relation labels found)")
	else:
		# sort labels by frequency descending then alphabetically
		labels_sorted = sorted(rel_counts.items(), key=lambda kv: (-kv[1], kv[0]))
		# compute column widths for neat alignment
		max_label = max(len(lbl) for lbl, _ in labels_sorted)
		max_total = max(len(str(cnt)) for _, cnt in labels_sorted)
		# widths for per-file counts
		def max_width_for(fn):
			return max((len(str(rel_counts_per_file.get(fn, {}).get(lbl, 0))) for lbl, _ in labels_sorted), default=1)
		max_a = max_width_for("test.json")
		max_b = max_width_for("valid.json")
		max_c = max_width_for("train.json")

		# show decomposition order once and print compact aligned columns
		# compute widths for numeric columns
		max_cnt_w = max(len(str(cnt)) for _, cnt in labels_sorted)
		max_a_w = max(len(str(rel_counts_per_file.get("test.json", {}).get(lbl, 0))) for lbl, _ in labels_sorted)
		max_b_w = max(len(str(rel_counts_per_file.get("valid.json", {}).get(lbl, 0))) for lbl, _ in labels_sorted)
		max_c_w = max(len(str(rel_counts_per_file.get("train.json", {}).get(lbl, 0))) for lbl, _ in labels_sorted)

		# header row (dynamic based on files order)
		file_stems = [Path(fn).stem for fn in files]
		max_file_ws = [
			max(len(str(rel_counts_per_file.get(fn, {}).get(lbl, 0))) for lbl, _ in labels_sorted)
			for fn in files
		]
		header_parts = [f"{'total':>{max_cnt_w}}"] + [f"{stem:>{w+1}}" for stem, w in zip(file_stems, max_file_ws)] + ["  label"]
		print(" ".join(header_parts))
		for label, cnt in labels_sorted:
			vals = [rel_counts_per_file.get(fn, {}).get(label, 0) for fn in files]
			vals_parts = [f"{cnt:>{max_cnt_w}}"] + [f"{v:>{w+1}}" for v, w in zip(vals, max_file_ws)]
			print(" ".join(vals_parts) + f"  {label}")
		print(f"\nTotal distinct relation labels (combined): {len(rel_counts)}")

	# Print per-split relation counts (aligned, numeric first)
	print("\n--- Per-split relation counts ---")
	for fn in files:
		print(f"\n[{fn}]")
		per = rel_counts_per_file.get(fn, {})
		if not per:
			print("(no relation labels found or file missing)")
			continue
		# sort by frequency desc then label
		per_sorted = sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))
		# compute widths
		max_cnt_w = max(len(str(cnt)) for _, cnt in per_sorted)
		max_label_w = max(len(lbl) for lbl, _ in per_sorted)
		# header
		print(f"{'count':>{max_cnt_w}}  label")
		for label, cnt in per_sorted:
			print(f"{cnt:>{max_cnt_w}}  {label}")
		print(f"Total distinct relation labels in {fn}: {len(per)}")


if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		sys.exit(1)

