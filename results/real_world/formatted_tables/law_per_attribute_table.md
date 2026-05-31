# LAW Per-Attribute Results

- dataset name: law
- outcome variable: pass_bar
- protected attributes: race_cat, male_cat
- fairness metrics note: lower is better for |TE|, |PSE|, |NDE|, SPD, and EO
- uncertainty note: mean +/- std denotes variation over CPDAG-consistent DAG extensions (or pruned DAGs), not resampling uncertainty

| protected_attribute | method | |TE| | |PSE| | |NDE| | SPD | EO | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| race_cat | GES | 0.1288 +/- 0.0479 | 0.0101 +/- 0.0042 | 0.1065 +/- 0.0383 | 0.1583 +/- 0.0554 | 0.3505 +/- 0.0143 | 0.6129 +/- 0.0056 |
| race_cat | GES+Phase2 | 0.0725 | 0.0140 | 0.0589 | 0.1863 | 0.2346 | 0.5998 |
| race_cat | GES+Hard | 0.0121 | 0.0073 | 0.0048 | 0.0348 | 0.0504 | 0.4879 |
| race_cat | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0109 +/- 0.0036 | N/A | 0.0111 +/- 0.0092 | 0.0216 +/- 0.0119 | 0.1147 +/- 0.0282 | 0.5131 +/- 0.0109 |
| race_cat | PF-Soft | 0.0184 | 0.0062 | 0.0122 | 0.0298 | 0.1037 | 0.5860 |
| race_cat | PF-Hard+Soft | 0.0097 | 0.0058 | 0.0104 | 0.0187 | 0.0415 | 0.5063 |
| male_cat | GES | 0.0047 +/- 0.0044 | 0.0032 +/- 0.0018 | 0.0060 +/- 0.0040 | 0.0289 +/- 0.0121 | 0.1235 +/- 0.0648 | 0.6129 +/- 0.0056 |
| male_cat | GES+Phase2 | 0.0018 | 0.0082 | 0.0001 | 0.0518 | 0.2522 | 0.5998 |
| male_cat | GES+Hard | 0.0219 | 0.0007 | 0.0226 | 0.0106 | 0.0327 | 0.4879 |
| male_cat | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0028 +/- 0.0008 | N/A | 0.0047 +/- 0.0004 | 0.0152 +/- 0.0036 | 0.0665 +/- 0.0224 | 0.5131 +/- 0.0109 |
| male_cat | PF-Soft | 0.0045 | 0.0048 | 0.0003 | 0.0197 | 0.0842 | 0.5860 |
| male_cat | PF-Hard+Soft | 0.0025 | 0.0033 | 0.0022 | 0.0104 | 0.0571 | 0.5063 |
| overall_avg | GES | 0.0667 +/- 0.0261 | 0.0066 +/- 0.0030 | 0.0563 +/- 0.0211 | 0.0936 +/- 0.0338 | 0.2370 +/- 0.0395 | 0.6129 +/- 0.0056 |
| overall_avg | GES+Phase2 | 0.0371 | 0.0111 | 0.0295 | 0.1190 | 0.2434 | 0.5998 |
| overall_avg | GES+Hard | 0.0170 | 0.0040 | 0.0137 | 0.0227 | 0.0416 | 0.4879 |
| overall_avg | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0068 +/- 0.0022 | N/A | 0.0079 +/- 0.0048 | 0.0184 +/- 0.0077 | 0.0906 +/- 0.0253 | 0.5131 +/- 0.0109 |
| overall_avg | PF-Soft | 0.0115 | 0.0055 | 0.0062 | 0.0248 | 0.0940 | 0.5860 |
| overall_avg | PF-Hard+Soft | 0.0061 | 0.0045 | 0.0063 | 0.0146 | 0.0493 | 0.5063 |
