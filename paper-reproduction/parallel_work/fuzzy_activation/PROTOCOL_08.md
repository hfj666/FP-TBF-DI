# Round 08 — diminishing returns for repeated training coverage

Protocol fixed before candidate fitting. Rebuilt original FP pools and reproduced all saved rule selections on the diagnostic prefixes. Each of the 16 persistent low-activation events has 225–359 eligible exact positive antecedents, of which only 1–4 were selected. Test the selector rather than changing activation again.

Change only the novelty term in CoveragePool. A rule covering a same-class training row currently covered c times earns 1/(1+c) for that row instead of 1[c=0]. Divide by the same class training count. This is the marginal gain of harmonic coverage, applied symmetrically. Keep quality, max Jaccard overlap penalty, alpha .5, stable tie order, 32 rules per class, support .1, max length 3, purity >.65, minimum 5. Selection uses only the current training prefix. Keep original HVDM table/distance, Gaussian firing, individual widths, zero-order BCE/L2, frozen scale, initialization and 500-iteration optimizer. Threshold 0. No search of coverage schedules.

Use the same three seeds and two batch sizes, 54 fits and 3294 predictions. Compare BA/AUC and paired FN against original HVDM and cumulative logistic, track the 16 low-activation and six Cauchy-added events. Prior diagnostics guided the selector change, so results are exploratory. Split-specific 213 holdout rows remain excluded.
