# Lightweight Noise Filtering for Distant Supervision

An interpretable NLP course project by **Zijun Wu, Tianchen Xie, and Tianluo Zhang**. The pipeline compares a training-free rule scorer with L2-regularized logistic regression over nine lexical and positional features. Configurable thresholds prioritize noise removal and expose the cost in retained clean data.

## Results

The following results use 4,059 valid relation instances from a previously inspected test split. Noise is the positive class; these results do not establish independent generalization or downstream relation-extraction gains.

| Development noise-recall target | Test noise recall | Clean retained | Retained-set purity | Instances kept |
|---|---:|---:|---:|---:|
| 90% | 95.12% | 14.30% | 90.16% | 488 |
| 95% | 96.74% | 9.23% | 89.87% | 316 |
| 99% | 100.00% | 0.03% | 100.00% | 1 |

The 90% preset retains 440 clean and 48 noisy instances. Its non-NA purity is 76.19%, compared with 75.74% without filtering. The 99% preset retains only one instance, so its perfect scores do not imply a useful training corpus. High noise recall therefore comes with substantial clean-data loss. A recall target is distinct from the classifier's numeric decision threshold.

## Code and documentation

- [Implementation, inference, and reproduction guide](revisions/20261014/README.md)
- [Paper PDF](revisions/20261014/paper/NLP_Overleaf_Revised.pdf)
- [Paper LaTeX source](revisions/20261014/paper/latex/acl_latex.tex)
- [Detailed experimental report (Chinese)](revisions/20261014/artifacts/recall_frontier_20261014/噪声召回优先最终报告.md)
- [Machine-readable results](revisions/20261014/artifacts/recall_frontier_20261014/results.json)
- [Exported model presets](revisions/20261014/models)

Run the documented pipeline from `revisions/20261014/`. It includes strict span validation, explicit unknown-label handling, sentence-grouped nested evaluation, and 32 regression tests. Model inference uses Python's standard library; training dependencies are pinned.
