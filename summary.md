**Overview**
- Purpose: Perform a "lightweight attention-style" filtering on JSONL datasets in the OpenNRE / NYT10m format — group sentences into bags, compute an attention score per sentence, then filter according to a policy and output the filtered JSONL.
- Input/Output: Read one JSON record per line (OpenNRE format). Output is JSONL (optionally attaching an attention_score field to retained records).

**Main features**
- Baging rule: bag id is composed of relation + head id + tail id (function `make_bag_id`).
- Sentence representation: `SentenceInstance` holds relation, original text, lowercased text, entity names, entity spans, original data, and provides a cached `between_text()` (text between entities).
- Robust span extraction: `safe_extract_span` performs robust extraction — prefer `pos` from the record, otherwise search the document for entity surface forms.
- Heuristic attention: `HeuristicAttentionProvider`
  - Uses relation-specific lexicons and rules (`RELATION_HINTS`) as lightweight cues.
  - Score components (combined in `_score_sentence` with weights):
    - Distance score (closer entity center character distance is better, weight 0.35)
    - Between-length score (shorter text between entities is favored, weight 0.25)
    - Keyword matches (relation-specific keywords / between-keywords, weight 0.3)
    - Entity order preference (`preferred_order`, weight 0.1)
    - Relation bias / boosts (relation-level bias and small boosts)
  - Scores for the NA relation are multiplied by 0.4 (downweighted).
  - Raw scores are converted to normalized attention weights via a temperature softmax (`temperature`).

- External attention import: `ExternalAttentionProvider` supports loading an external JSONL (each line containing `bag_id` + `scores`/`attentions`); if lengths mismatch or no entry exists it falls back to the heuristic provider.

- Filtering policies:
  - `AbsoluteThresholdPolicy`: keep sentences above a threshold (if fewer than min_keep, retain top min_keep).
  - `TopKPolicy`: keep top-K per bag.
  - `RelativeMarginPolicy`: keep sentences whose score is within margin of the top score (ensures min_keep).
  - `DropBottomPolicy`: drop the bottom K sentences (keep at least min_keep).
  - `DropPercentPolicy`: drop the lowest X% (guarantee min_keep).

- Statistics and noise evaluation: `StatsTracker`
  - Tracks number of bags and sentences processed, per-relation keep/drop stats.
  - Optional noise evaluation (`--evaluate-noise`): if records include an `is_noise` field, treats dropped = predicted noise and compares to ground truth to compute TP/FP/FN/TN and precision/recall/f1/accuracy.

- CLI / runtime parameters:
  - Data, split and file paths: `--dataset-root`, `--split`, `--input`, `--output`
  - Policy and hyperparameters: `--policy` (`absolute/topk/relative/drop/drop_pct`), `--threshold`, `--topk`, `--margin`, `--drop-bottom`, `--drop-percent`, `--min-keep`
  - Attention control: `--temperature`, `--attention-file`, `--annotate-attention`
  - Evaluation / reporting: `--evaluate-noise`, `--report`
  - Other: `--limit-bags` (debug: only read first N bags), `--log-level`

**I/O and behavior**
- Iterate bag-by-bag, obtain attention scores, let the selected policy `select` a keep mask, and write retained records to the output file (one JSONL line per record).
- If `--annotate-attention` is set, add an `attention_score` field (4–6 decimal places) to retained records.
- Print and (optionally) save a summary statistics JSON (`--report`) at the end.

**Implementation notes / emphases**
- `RELATION_HINTS` lists many common relations with trigger keywords, between-keywords, preferred order and bias/boost values to support interpretable heuristic scoring.
- Scoring uses manually set weights (0.35 / 0.25 / 0.3 / 0.1) and clamps to sensible ranges.
- Softmax temperature is adjustable to avoid one sentence monopolizing weight or to prevent over-smoothing.
- Malformed JSON lines are tolerated (skipped and logged).
- External attention files are validated for length and normalization; on mismatch the system falls back to the heuristic provider.

**Usage examples (also included at file bottom)**
- Run topk=1 on the first 5 bags and output samples:
  - python3 main.py --split train --limit-bags 5 --policy topk --topk 1 --output nyt10m_train_filtered_sample.txt
- With noise evaluation:
  - python3 main.py --input arnor_test.txt --policy drop_pct --drop-percent 0.5 --limit-bags 10 --evaluate-noise --output arnor_filtered.txt

**Summary assessment**
- This is a lightweight filtering tool for distant supervision datasets that centers on interpretable heuristic signals:
  - Pros: Does not require deep models, is transparent, can incorporate external attention signals, supports multiple filtering policies and noise evaluation.
  - Cons: Heuristic rules may not generalize across relations or genres; weights and keywords need manual tuning or data-driven optimization.
