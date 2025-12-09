Reproduction Guide 

**Environment**
- **Python:** 
- **Shell:** 

**Environment setup**

Follow these steps to recreate the development environment used for this project. The code was tested with Python 3.10.18 in a conda environment named `nlp`.

1. Install Miniconda or Anaconda if you don't already have it: https://docs.conda.io/en/latest/miniconda.html

2. Create and activate the conda environment (uses Python 3.10):

```bash
conda create -n nlp python=3.10 -y
conda activate nlp
```

If you specifically need Python 3.10.18 and your conda channels provide it, you can request `python=3.10.18` instead of `python=3.10`.

3. Install required packages (recommended: use `conda` for scientific packages):

Option A — conda (recommended on macOS):

```bash
conda install -n nlp -c conda-forge numpy matplotlib pandas seaborn -y
```

Option B — pip (inside the activated env):

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib pandas seaborn
```

4. (Optional) If you want an isolated, reproducible manifest, create a `requirements.txt` or `environment.yml` (example `requirements.txt` shown below).

**Requirements (minimal for plotting/analysis)**

Create a file named `requirements.txt` with the following contents for pip installs:

```
numpy>=1.22
matplotlib>=3.4
pandas>=1.3
seaborn>=0.11
```

Then install with:

```bash
python -m pip install -r requirements.txt
```

Notes:
- The project's core conversion and filtering scripts (`scripts/original_convert_script.py`, `filter/main.py`, `filter/model.py`) rely only on the Python standard library and will run with Python 3.10 without third-party packages.
- The plotting and analysis scripts (`filter/plot_scores.py`, `scripts/plot_sentence_stats.py`, `scripts/data_analysis/analyze_train.py`) require the packages above.
- If you use `conda` and need to pin exact versions (for reproducibility), I can generate an `environment.yml` for you.

**1. Clone the repo**
```bash
git clone https://github.com/LewisZZZ-Zhang/NLP_Final_Project.git
```

**2. Fetch the raw data**
```bash
to be filled
```

**1. Convert original ANNOR/ARNOR dataset to OpenNRE-style files**

```bash
/opt/anaconda3/envs/nlp/bin/python "/Users/Lewis/SchoolWorks/05_Junior_Fall/NLP/final_project/nlp_final_project/scripts/original_convert_script.py"
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

**2. Filter noisy/low-confidence sentences**

After conversion, a filtering step was applied using the script `filter/main.py`. The filtering operation used the `absolute` policy with a threshold of `0.0025`, and produced both a filtered dataset and evaluation/plots for noise splitting.

Run the filtering command (example):

```bash
python3 filter/main.py \
  --input OpenNRE/benchmark/arnor_converted/arnor_dev.txt \
  --policy absolute --threshold 0.0025 --evaluate-noise \
  --output tmp/arnor_filtered_full.txt
```

Example output from this filtering run:

```
INFO Loading OpenNRE/benchmark/arnor_converted/arnor_dev.txt
INFO Loaded 198 bags (4567 sentences)
INFO Filtering 198 bags from OpenNRE/benchmark/arnor_converted/arnor_dev.txt using policy=absolute (min_keep=1)
INFO Wrote filtered split to tmp/arnor_filtered_full.txt

=== Filter summary for OpenNRE/benchmark/arnor_converted/arnor_dev.txt ===
Bags processed: 198
Sentences kept: 2958 / 4567 (64.77%)
Avg sentences per bag: 23.07 -> 14.94
Per-relation retention (top counts):
  NA: 1989/3592 kept (55.4%)
  /location/location/contains: 472/478 kept (98.7%)
  /people/person/place_lived: 198/198 kept (100.0%)
  /people/person/nationality: 117/117 kept (100.0%)
  /business/person/company: 113/113 kept (100.0%)
  /location/country/capital: 15/15 kept (100.0%)
  /people/person/place_of_birth: 15/15 kept (100.0%)
  /people/deceased_person/place_of_death: 14/14 kept (100.0%)
Noise filtering evaluation (predict drop = noise):
  precision=0.437 recall=0.619 f1=0.512 accuracy=0.707
  counts: TP=703 FP=906 FN=432 TN=2526 total=4567
Saved clean/noise split plot to tmp/arnor_filtered_full_score_distribution_noise_split.png
Saved zoomed clean/noise split plot to tmp/arnor_filtered_full_score_distribution_noise_split_0-0.10.png

=== Unknown-relations summary (per file) ===

File: OpenNRE/benchmark/arnor_converted/arnor_dev.txt
  (All relations have defined rules)
```

This step writes the filtered output to `tmp/arnor_filtered_full.txt` and also saves the score-distribution plots in the `tmp/` directory.

**Files and artifacts generated**
- Converted data: `data/arnor_converted/arnor_dev.txt`, `data/arnor_converted/arnor_test.txt`, `data/arnor_converted/arnor_test_noise.txt`
- Relation maps: `data/arnor_converted/arnor_dev_rel2id.json`, `data/arnor_converted/arnor_test_rel2id.json`, `data/arnor_converted/arnor_test_noise_rel2id.json`
- Filtered split: `tmp/arnor_filtered_full.txt`
- Plots: `tmp/arnor_filtered_full_score_distribution_noise_split.png`, `tmp/arnor_filtered_full_score_distribution_noise_split_0-0.10.png`

**Notes & tips**
- If you use a conda environment, make sure to activate it before running the conversion script, or call the environment's python executable explicitly as shown above.
- Paths in the commands above assume you are running from the repository root (`nlp_final_project`). Adjust absolute paths if you run commands from a different location.
- The filtering `--threshold` and `--policy` options can be tuned to control how aggressive the noise removal is.

If you'd like, I can also:
- Add a small shell script to reproduce these steps automatically.
- Commit this `README1.md` to the repository and create a brief `Makefile` target for repro.

---
Generated by the project contributor on macOS; adapt paths to your system as needed.
