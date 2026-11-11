# FP-TBF-DI

Core implementation for **基于局部一致规则的乳腺超声病灶逐批分类方法**
(Sequential Classification of Breast Ultrasound Lesions Using Locally Consistent Rules).
FP-TBF-DI maintains an FP-tree, fits three scoring branches on each revealed
training prefix, and averages the scores to classify the next batch.

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
