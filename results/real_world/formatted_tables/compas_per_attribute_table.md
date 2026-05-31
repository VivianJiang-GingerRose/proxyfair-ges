# COMPAS Per-Attribute Results

- dataset name: compas
- outcome variable: two_year_recid
- protected attributes: race_cat, sex
- fairness metrics note: lower is better for |TE|, |PSE|, |NDE|, SPD, and EO
- uncertainty note: mean +/- std denotes variation over CPDAG-consistent DAG extensions (or pruned DAGs), not resampling uncertainty

| protected_attribute | method | |TE| | |PSE| | |NDE| | SPD | EO | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| race_cat | GES | 0.0335 +/- 0.0297 | 0.0000 +/- 0.0000 | 0.0335 +/- 0.0297 | 0.0701 +/- 0.0329 | 0.1991 +/- 0.0245 | 0.6190 +/- 0.0531 |
| race_cat | GES+Phase2 | 0.0197 | 0.0000 | 0.0197 | 0.1830 | 0.2258 | 0.6089 |
| race_cat | GES+Hard | 0.0299 | 0.0045 | 0.0133 | 0.1213 | 0.1892 | 0.6772 |
| race_cat | PostHocPruning | 0.0128 +/- 0.0098 | 0.0000 +/- 0.0000 | 0.0128 +/- 0.0098 | 0.2152 +/- 0.1106 | 0.3160 +/- 0.1512 | 0.6319 +/- 0.0473 |
| race_cat | PF-Soft | 0.0173 | 0.0069 | 0.0104 | 0.0379 | 0.0835 | 0.6133 |
| race_cat | PF-Hard+Soft | 0.0076 | 0.0041 | 0.0090 | 0.1667 | 0.1771 | 0.6167 |
| sex | GES | 0.0118 +/- 0.0163 | 0.0267 +/- 0.0157 | 0.0200 +/- 0.0157 | 0.0980 +/- 0.0361 | 0.2190 +/- 0.0296 | 0.6190 +/- 0.0531 |
| sex | GES+Phase2 | 0.1056 | 0.0230 | 0.0727 | 0.1221 | 0.1978 | 0.6089 |
| sex | GES+Hard | 0.1353 | 0.0173 | 0.1159 | 0.2273 | 0.2971 | 0.6772 |
| sex | PostHocPruning | 0.0632 +/- 0.0148 | 0.0564 +/- 0.0102 | 0.0108 +/- 0.0032 | 0.1698 +/- 0.0594 | 0.2706 +/- 0.0817 | 0.6319 +/- 0.0473 |
| sex | PF-Soft | 0.0149 | 0.0370 | 0.0221 | 0.0502 | 0.2115 | 0.6133 |
| sex | PF-Hard+Soft | 0.0331 | 0.0175 | 0.0225 | 0.1762 | 0.3585 | 0.6167 |
| overall_avg | GES | 0.0227 +/- 0.0230 | 0.0133 +/- 0.0079 | 0.0268 +/- 0.0227 | 0.0840 +/- 0.0345 | 0.2091 +/- 0.0271 | 0.6190 +/- 0.0531 |
| overall_avg | GES+Phase2 | 0.0627 | 0.0115 | 0.0462 | 0.1525 | 0.2118 | 0.6089 |
| overall_avg | GES+Hard | 0.0826 | 0.0109 | 0.0646 | 0.1743 | 0.2431 | 0.6772 |
| overall_avg | PostHocPruning | 0.0380 +/- 0.0123 | 0.0282 +/- 0.0051 | 0.0118 +/- 0.0065 | 0.1925 +/- 0.0850 | 0.2933 +/- 0.1165 | 0.6319 +/- 0.0473 |
| overall_avg | PF-Soft | 0.0161 | 0.0219 | 0.0162 | 0.0441 | 0.1475 | 0.6133 |
| overall_avg | PF-Hard+Soft | 0.0203 | 0.0108 | 0.0157 | 0.1714 | 0.2678 | 0.6167 |
