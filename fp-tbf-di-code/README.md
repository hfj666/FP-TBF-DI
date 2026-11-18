# FP-TBF-DI

Core implementation for **基于局部一致规则的乳腺超声病灶逐批分类方法**
(Sequential Classification of Breast Ultrasound Lesions Using Locally Consistent Rules).
FP-TBF-DI maintains an FP-tree, fits three scoring branches on each revealed
training prefix, and combines the three scores using fixed weights selected
with continuous ant colony optimization (ACOR):

| Branch | Weight |
| --- | ---: |
| A | 0.38633621372064075 |
| B | 0.3109653252978891 |
| C | 0.30269846098146996 |

The weighted score is classified as malignant at zero or above. The weights
come from the pooled internal development predictions and are reused for
every batch. The pooled development accuracy is an in-sample result for the
fusion weights. The runner records the weights in its output manifest.

## Search fusion weights with ACOR

From the `fp-tbf-di-code` directory, pass a local development CSV containing
columns `A,B,C,label`. The first three are branch scores and `label` is
`-1` or `+1`. Use only scores designated for weight fitting. If using
runner output, select the development `stream` rows before passing the CSV.

```sh
python search_acor_fusion_weights.py development_scores.csv --seed 20261117
```

The search starts with 96 candidates and runs 100 rounds of 96 proposals.
Each round samples around candidates in the ranked archive, projects proposals
onto nonnegative weights that sum to one, and retains the best 96.
It maximizes the number of correct classifications at threshold zero, with ties
favoring weights closer to equal weighting. The script prints the weights
and their in-sample development accuracy. The published weights were fitted to
pooled internal development scores, which are distributed separately.

## Run the synthetic example

Use Python 3.12 and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python demo.py
```

The example generates all inputs and checks that fitting and prediction run.
It does not report a performance estimate for the study cohort.

## Run a saved study split

Use the original score table and split file only if you have permission to
access them. This repository contains the method implementation and a runner
for one study configuration. The clinical score table, split indices,
per-record study predictions, and manuscript files are distributed separately
subject to their applicable permissions.

The whitespace-delimited score table has 1,062 rows. Column one contains
labels (-1 benign, +1 malignant), followed by the 18 integer scores in original
column order. The split JSON is an array of objects with seed, train_stream
(849 ordered zero-based row indices), and holdout (213 disjoint row indices).

```sh
python run_fp_tbf_simplified.py --data /path/to/scores.txt --splits /path/to/splits.json --seed 17 --batch-size 50 --out run-output/seed17-b50
```

Seeds 17, 29 and 43 and batch sizes 50 and 100 select the six saved configurations.
The runner writes per-record scores, fit details, a metric summary and input
hashes to the chosen local output directory. It evaluates the training stream
and final holdout. The manuscript's comparison and ablation archives are
outside the scope of this repository.
