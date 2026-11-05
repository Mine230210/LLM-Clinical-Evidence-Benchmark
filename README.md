# Benchmarking the Reliability of Large Language Models Across the Clinical Evidence Ecosystem

This repository contains the statistical analysis and figure-generation code accompanying the study *Benchmarking the Reliability of Large Language Models Across the Clinical Evidence Ecosystem*.

The study evaluates three large language models (ChatGPT, Gemini, and DeepSeek) across three independent benchmarks:

- **Evidence completeness:** Identification of clinical trial safety signals and supporting evidence sources.
- **Evidence consensus:** Assessment of agreement and disagreement in clinical evidence.
- **Evidence applicability:** Evaluation of whether randomized controlled trial findings can be applied to specific clinical scenarios.

## Code

The `analysis/` directory contains the scoring and statistical analysis scripts for the three benchmarks. These scripts calculate model performance, bootstrap confidence intervals, permutation tests, and repeated-response agreement.

The `figures/` directory contains scripts for generating Figure 2 and Supplementary Figure 1 from the analysis outputs.

## Data

The benchmark datasets and recorded model responses are provided separately on Figshare. Each benchmark includes a Gold standard file and a corresponding model response file.

Task 1, Task 2, and Task 3 applicability-verdict analyses can be reproduced using the public data. Analyses involving Task 3 key-reason accuracy and reason-quality agreement require additional human evaluation ratings, which are not included in the public dataset.

## Usage

Python 3.10 or later is required. Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

Download the six benchmark CSV files from Figshare and place them in a `data/` directory.

Run the analyses from the repository root:

```bash
python analysis/task1_analysis.py --gold data/Task1_Gold.csv --responses data/Task1_Responses.csv --output-dir outputs/task1

python analysis/task2_analysis.py --gold data/Task2_Gold.csv --responses data/Task2_Responses.csv --output-dir outputs/task2

python analysis/task3_analysis.py --gold data/Task3_Gold.csv --responses data/Task3_Responses.csv --output-dir outputs/task3
```

Supplementary Figure 1 can be generated from the resulting analysis outputs:

```bash
python figures/supplementary_figure1.py --task1-results outputs/task1 --task2-results outputs/task2 --task3-results outputs/task3 --output-dir outputs/figures
```

Public data do not contain the human ratings needed for the complete Figure 2. The public-mode command exports the reproducible figure source tables and a missing-input status file without drawing an incomplete figure:

```bash
python figures/figure2.py --task1-results outputs/task1 --task2-results outputs/task2 --task3-results outputs/task3 --mode public --output-dir outputs/figures
```

For an authorized local full analysis, supply a CSV containing `item_id`, `model`, `repeat_id`, and `human_key_reason_rating`:

```bash
python analysis/task3_analysis.py --gold data/Task3_Gold.csv --responses data/Task3_Responses.csv --reason-ratings private/Task3_Key_Reason_Ratings.csv --output-dir outputs/task3_full

python figures/figure2.py --task1-results outputs/task1 --task2-results outputs/task2 --task3-results outputs/task3_full --mode full --output-dir outputs/figures_full
```

For detailed information on benchmark construction, evaluation procedures, and statistical methods, please refer to the accompanying manuscript and Supplementary Information.
