# Lightweight distant-supervision noise filtering

A corrected, interpretable course-project pipeline with a small supervised extension. Original course project: Zijun Wu, Tianchen Xie, and Tianluo Zhang. Post-course revision prepared by Zijun Wu on 2026-10-14; manuscript is a draft for review.

**Main finding:** aggressive filtering can achieve high noise recall, but discards most clean data. This is an exploratory audit of previously inspected data, not a new independent benchmark or a claim of improved downstream relation extraction.

## What changed

- Stable surface identities replace sentence-local entity counters; source mention provenance is retained.
- Missing or ambiguous entity spans are quarantined. Unknown noise labels are not counted as clean.
- Bag softmax weights are no longer interpreted as global confidence.
- A training-free raw rule score is compared with L2 logistic regression over nine existing features. The latter requires labeled training data.
- Final selection prioritizes noise recall, with no minimum clean retention, no mandatory instance per bag, and no relation coverage guarantee.

## Recorded results

All numbers below concern the **4,059 valid instances of the previously inspected test split**. Noise is the positive class. The three presets were all reported, not chosen by test performance.

| Development recall target | Test noise recall | Clean retained | Retained-set purity | Kept instances |
|---|---:|---:|---:|---:|
| 90% | 95.12% | 14.30% | 90.16% | 488 |
| 95% | 96.74% | 9.23% | 89.87% | 316 |
| 99% | 100.00% | 0.03% | 100.00% | 1 |

The 90% preset keeps 440 clean and 48 noisy instances. On non-NA instances its purity is only 76.19%, versus 75.74% without filtering. The 99% result is essentially total deletion; its perfect metrics are not evidence of a useful corpus. Empty-set purity is undefined. The recall target is distinct from the model score threshold and from the earlier 95% clean-retention requirement.

Read [the revised manuscript](paper/NLP_Overleaf_Revised.pdf), its [editable source](paper/latex/acl_latex.tex), and [the detailed Chinese report](artifacts/recall_frontier_20261014/噪声召回优先最终报告.md). Aggregate results include raw-score baselines, per-relation diagnostics, and duplicate-excluded diagnostics.

## Run inference

Inference uses Python's standard library. Use one of the three exported models explicitly; there is no hidden default “best” preset. Input must first pass the strict converter. The output records **every scored decision** with a `keep` flag, rather than silently deleting rejected rows. Invalid-span records are not scored and remain in the converted source for separate review.

```sh
python3 scripts/benchmark_data.py --inputs /path/to/ARNOR/test.json
python3 filter/logistic_filter.py --model models/model_recall90.json --input data/arnor_benchmark_v2/arnor_test.txt --output decisions.jsonl
```

Conversion and inference refuse to overwrite existing outputs. Never treat quarantined inputs as confirmed noise. Noise annotations are needed for evaluation/training, but not for applying a model to valid instances.

## Reproduce fitting and evaluation

Tested with Python 3.12 and the pinned dependencies. Obtain the original ARNOR-format `dev.json`, `test.json`, and `test_noise.json` from your authorized data copy; see [ARNOR source and data context](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-ARNOR). Put them in `data/arnor_dataset-2.0.0/`. Compare their SHA-256 digests with `data_source_sha256.json`. The large unlabeled train split is not required for logistic fitting. Data are not redistributed here; upstream access and terms apply.

Run from a fresh checkout root (the `results` output directories must not already exist):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-logistic-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/benchmark_data.py
.venv/bin/python scripts/bag_rejection.py --out results/bag_rejection_20261014_run2
.venv/bin/python scripts/logistic_experiment.py
.venv/bin/python scripts/report_logistic.py
.venv/bin/python scripts/recall_frontier.py
.venv/bin/python scripts/report_recall_frontier.py
```

The earlier logistic run generates inner-fold probabilities and an audit of 100 inner fits. The recall-first run reuses these because model fitting is unchanged, reselects C and cutoffs for each target, and refits outer/full-dev models. Inner folds exclude outer validation instances; scaling is fit only on the corresponding training partition. Every final configuration is frozen before diagnostic test evaluation. Scripts use fixed relative result paths, so run them in the stated order in a fresh checkout.

`artifacts/` contains the recorded aggregates; regenerated per-instance decisions, lineage, and audit files go to ignored `results/`. Hardware-dependent timings and environment strings may differ. The historical Chinese report's `.venv-logreg` path is the original local environment name; use `.venv` here.

## Scope, attribution, and terms

Original repository: https://github.com/LewisZZZ-Zhang/NLP_Final_Project . This package isolates the corrected filtering experiment; it does not bundle the original neural checkpoints, corpora, full repository history, or original submitted PDF. It preserves team attribution without asserting that all coauthors approved the post-course extension. It is a course-project revision, not a new peer-reviewed paper.

No new license grant is made by this packaging step. Confirm the original project's applicable terms with its authors before redistribution; upstream datasets and third-party work retain their own terms.

The old comparison of noise detection F1 against neural relation-classification results is withdrawn, as is the unmatched filtering-versus-training speed comparison. This release makes no claim to outperform BERT/ARNOR or improve downstream relation extraction. A fresh independent evaluation and matched downstream retraining would be needed for stronger claims.
