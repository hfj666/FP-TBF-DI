# Round 07 — Cauchy firing on HVDM distance

Fixed after replaying the 22 diagnostic events and before candidate predictions. In all 16 original low-activation events the shared-width model still has a minority positive activation share. All 16 also remain negative if only the widths are swapped. Among the six new misses, the width-only swap reproduces five and the head-only swap reproduces the other. Exact rule sets stay fixed in this experiment. Test whether keeping more activation on approximate matches can help compensate for sparse exact positive matches.

Relative to original HVDM, change only the radial firing function from exp(-a) to 1/(1+a), where a = distance/(2 sigma^2). This is the same curvature-matched function already implemented for numeric distance in round 03, now evaluated on the unchanged HVDM distance. Keep rule confidence, sum normalization, individual widths, unconstrained zero-order consequents, BCE/L2, bounds, cold initialization, and 500-iteration budget. There is no new shape parameter.

Use 57 verified original HVDM states, 54 unique prefix fits, 549 predictions for each of the six configurations. Keep threshold 0. Compare BA/AUC and new/corrected FN with original HVDM and cumulative logistic. Report the two diagnostic groups. The 213 holdout rows per split remain outside the experiment. The design was informed by seen errors and is exploratory.
