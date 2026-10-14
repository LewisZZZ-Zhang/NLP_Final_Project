# Corrected evaluation and recall-first revision

A post-course revision is available in [revisions/20261014](revisions/20261014/README.md), including corrected conversion, nine-feature L2 logistic filtering, reproduced experimental aggregates, and the [revised paper](revisions/20261014/paper/NLP_Overleaf_Revised.pdf). Original team attribution is preserved.

The historical guide below describes the original submission. Its reported noise F1, global bag-softmax cutoff results, cross-task neural comparisons, and speedup claims are **superseded by the corrected audit** and should not be used as current performance evidence. The new high-recall results also discard most clean data; see the revision for the full trade-off and non-NA limitations.

---

## Historical reproduction guide (original submission)

Reproduction Guide 

**1. Clone the repo**
```bash
git clone https://github.com/LewisZZZ-Zhang/NLP_Final_Project.git
```

**2. Fetch the raw data**

Follow the ARNOR repository instructions and place the extracted dataset under `data/` so the project can find it.

Repository (ARNOR): https://github.com/PaddlePaddle/Research/blob/master/NLP/ACL2019-ARNOR/README.md\
or: http://baidu-nlp.bj.bcebos.com/arnor_dataset-2.0.0.tar.gz

Expected directory layout (example):

```text
data/
└── arnor-dataset-2.0.0/
  ├── dev.json
  ├── test_noise.json
  ├── test.json
  └── train.json
```

Place the extracted `arnor-dataset-2.0.0` folder directly under `data/`. Adjust the folder name if the upstream archive uses a different name.

**3. Convert original ANNOR/ARNOR dataset to OpenNRE-style files**

```bash
python scripts/original_convert_script.py
```

Expected console output (example from a run):

```
Wrote 4567 examples to data/arnor_converted/arnor_dev.txt
Wrote 12 relations to data/arnor_converted/arnor_dev_rel2id.json
Wrote 4484 examples to data/arnor_converted/arnor_test.txt
Wrote 12 relations to data/arnor_converted/arnor_test_rel2id.json
Wrote 533 examples to data/arnor_converted/arnor_test_noise.txt
Wrote 15 relations to data/arnor_converted/arnor_test_noise_rel2id.json
```

The above commands produce the converted dataset under the `data/arnor_converted/` directory (dev, test, and noise splits and their relation mapping files).

**4. Filter noisy/low-confidence sentences**

After conversion, a filtering step was applied using the script `filter/main.py`. The filtering operation used the `absolute` policy with a threshold of `0.0025`, and produced both a filtered dataset and evaluation/plots for noise splitting.

Run the filtering command (example):

```bash
python3 filter/main.py \
  --input data/arnor_converted/arnor_dev.txt \
  --policy absolute --threshold 0.0025 --evaluate-noise \
  --output tmp/arnor_filtered_full.txt
```

```bash
python3 filter/main.py \
  --input data/arnor_converted/arnor_test.txt \
  --policy absolute --threshold 0.0025 --evaluate-noise \
  --output tmp/arnor_filtered_full.txt
```

Example output from this filtering run:

```
=== Filter summary for data/arnor_converted/arnor_dev.txt ===
Bags processed: 198
Sentences kept: 2958 / 4567 (64.77%)
Avg sentences per bag: 23.07 -> 14.94
Per-relation retention (top counts):
  NA: 1989/3592 kept (55.4%)
  /location/location/contains: 472/478 kept (98.7%)
       :
       :
Noise filtering evaluation (predict drop = noise):
  precision=0.4369 recall=0.6194 f1=0.5124 accuracy=0.7070
  counts: TP=703 FP=906 FN=432 TN=2526 total=4567
```

This step writes the filtered output to `tmp/arnor_filtered_full.txt` and also saves the score-distribution plots in the `tmp/` directory.

**Files and artifacts generated**
- Converted data: `data/arnor_converted/arnor_dev.txt`, `data/arnor_converted/arnor_test.txt`, `data/arnor_converted/arnor_test_noise.txt`
- Relation maps: `data/arnor_converted/arnor_dev_rel2id.json`, `data/arnor_converted/arnor_test_rel2id.json`, `data/arnor_converted/arnor_test_noise_rel2id.json`
- Filtered split: `tmp/arnor_filtered_full.txt`
- Distribution Plots: `tmp/arnor_filtered_full_score_distribution_noise_split.png`, `tmp/arnor_filtered_full_score_distribution_noise_split_0-0.10.png`