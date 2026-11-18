# Paper experiment code

Run commands from this directory with Python 3.12. This tree preserves the
paper scripts' relative paths. The separate `fp-tbf-di-code` directory is the
standalone prediction implementation.

```sh
python -m pip install -r requirements.txt
python generate_splits.py data/dataset_1062.txt
python run_fp_tbf_simplified.py --seed 17 --batch-size 50 --fusion equal --out run-output/equal-17-50
```

Place an authorized copy of the original 1,062-row score table at
`data/dataset_1062.txt`. Keep its original row order. Each whitespace-delimited
row contains the label (-1 benign or +1 malignant) and 18 integer scores.
The split generator reproduces the ordered training and holdout indices for
seeds 17, 29 and 43. For each split, it leaves 213 rows out and shuffles the
849-row training stream. The first 300 rows initialize the model. Batches
of 50 or 100 are predicted before their labels enter training.

Repeat the command for all three seeds and both batch sizes, using a new
output directory each time. The manuscript's main experiment uses
`--fusion equal` (3,175 correct predictions out of 3,294 stream events).
The later fixed-weight variant uses `--fusion acor` (3,179/3,294 on the
pooled development events used to fit the weights). Both give 1,236/1,278
correct final holdout predictions when pooled. Each count sums events across
the six overlapping configurations.

## Manuscript experiment entry points

| Analysis | Run with `python` | Inputs beyond the score table and split indices |
| --- | --- | --- |
| Main stream and final holdout | `run_fp_tbf_simplified.py` | None |
| Historical three-branch holdout replay | `run_three_way_retrospective_holdout.py` | Historical rescue-interaction stream predictions |
| Training checkpoints | `run_paper_incremental_checkpoints.py` | Historical stream and holdout replay |
| Matched rule budgets and original C ablations | `run_paper_budget_c_ablation.py` | Historical stream and holdout replay |
| Joint P/E/U removal and simplified-model checkpoints | `run_paper_c_joint_reduction.py` | Historical stream, holdout replay and checkpoint metrics |
| Conventional baselines, branch replacement and grouped five folds | `run_paper_evidence_extension.py` | Joint-reduction predictions and fit cache |
| Frozen model comparison | `run_paper_frozen_update.py` | Evidence-extension predictions |
| Two-branch comparison and tree maintenance timing | `run_paper_parsimony_cost.py` | Joint-reduction and grouped results |
| A+LR timing | `run_paper_a_lr_cost.py` | Parsimony matched predictions |
| Matched fusion budgets | `run_paper_matched_fusion_budget.py` | Rule-budget and joint-reduction predictions |
| Repeated grouped evaluation | `run_paper_grouped_stability.py` | Evidence-extension grouped splits and predictions |
| Rule selection by fusion structure | `run_paper_selection_cross.py` | Rule-budget, joint-reduction and repeated grouped results |
| Literature-inspired baseline alignment | `run_paper_huang_alignment.py` | Direct-comparison and evidence-extension predictions |
| PSO, GA, ABC and ACOR comparison | `run_paper_fusion_optimizer_comparison.py` | Joint-reduction predictions |

The `verify_paper_*.py` and `summarize_paper_*.py` scripts check and summarize
locally generated outputs. Run the verifier before its corresponding
summarizer. The joint-reduction verifier also reads an equal-weight reference
run for seed 17, batch size 100 at `results/fp_tbf_di_entry_check`.
Generate it with `python run_fp_tbf_simplified.py --seed 17 --batch-size 100
--fusion equal --out results/fp_tbf_di_entry_check`. The verifier's manuscript
format checks also require the local `paper/FP_TBF_DI_中文论文精简修订稿.md`.
Timing results depend on machine, numerical libraries and thread settings.
For historical timing runs, set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
OMP_NUM_THREADS to 1 in the shell before launching Python.

## Historical inputs and rendering

The archived stream scores expected by the checkpoint and C-branch experiments
are at `parallel_work/robust_learning/experiment_rescue_interactions/predictions.csv`.
The corresponding holdout file is
`results/three_way_retrospective_holdout/predictions.csv`.
The source tree includes upstream research scripts under `parallel_work/` and
supporting runners such as `run_online_study.py` and
`run_direct_comparison.py`. Historical runs check upstream manifests and
predictions. For an exact replay of those runs, obtain the authorized archives
and preserve their filenames. The generated splits suffice for the main
model, while older comparisons also use their original `choices.json` and,
in some cases, the original split-file bytes for provenance.
The direct-comparison runner hashes the local `reference.txt` literature
excerpt as part of its source manifest.

The table renderer reads the locally supplied manuscript at
`paper/FP_TBF_DI_中文论文_完整投稿稿.md` and uses the Windows font directory.
Figure scripts are in `paper/revision_evidence/`. Keep manuscript copies,
clinical input files, case-level outputs, and historical experiment archives
local. This repository distributes source and protocol notes.
