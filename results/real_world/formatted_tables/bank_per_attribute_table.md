# BANK Per-Attribute Results

- dataset name: bank
- outcome variable: y
- protected attributes: marital, age
- fairness metrics note: lower is better for |TE|, |PSE|, |NDE|, SPD, and EO
- uncertainty note: mean +/- std denotes variation over CPDAG-consistent DAG extensions (or pruned DAGs), not resampling uncertainty

| protected_attribute | method | |TE| | |PSE| | |NDE| | SPD | EO | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| marital | GES | 0.0243 +/- 0.0001 | 0.0044 +/- 0.0016 | 0.0199 +/- 0.0017 | 0.0385 +/- 0.0079 | 0.1240 +/- 0.0461 | 0.6417 +/- 0.0230 |
| marital | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0203 | N/A | 0.0130 | 0.0670 | 0.2538 | 0.6527 |
| marital | GES+Hard | 0.0065 | 0.0010 | 0.0065 | 0.0312 | 0.1136 | 0.6091 |
| marital | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0254 +/- 0.0016 | N/A | 0.0198 +/- 0.0009 | 0.0494 +/- 0.0007 | 0.0684 +/- 0.0126 | 0.6683 +/- 0.0132 |
| marital | PF-Soft | 0.0019 | 0.0029 | 0.0048 | 0.0092 | 0.0630 | 0.6148 |
| marital | PF-Hard+Soft | 0.0001 | 0.0010 | 0.0012 | 0.0140 | 0.0245 | 0.5194 |
| age | GES | 0.0104 +/- 0.0077 | 0.0000 +/- 0.0000 | 0.0104 +/- 0.0077 | 0.0264 +/- 0.0001 | 0.0743 +/- 0.0324 | 0.6417 +/- 0.0230 |
| age | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0190 | N/A | 0.0190 | 0.0525 | 0.1194 | 0.6527 |
| age | GES+Hard | 0.0073 | 0.0162 | 0.0036 | 0.0198 | 0.0703 | 0.6091 |
| age | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0175 +/- 0.0004 | N/A | 0.0175 +/- 0.0004 | 0.0358 +/- 0.0023 | 0.0713 +/- 0.0167 | 0.6683 +/- 0.0132 |
| age | PF-Soft | 0.0070 | 0.0095 | 0.0165 | 0.0215 | 0.0826 | 0.6148 |
| age | PF-Hard+Soft | 0.0060 | 0.0014 | 0.0036 | 0.0122 | 0.0307 | 0.5194 |
| overall_avg | GES | 0.0174 +/- 0.0039 | 0.0022 +/- 0.0008 | 0.0152 +/- 0.0047 | 0.0324 +/- 0.0040 | 0.0992 +/- 0.0392 | 0.6417 +/- 0.0230 |
| overall_avg | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0197 | N/A | 0.0160 | 0.0597 | 0.1866 | 0.6527 |
| overall_avg | GES+Hard | 0.0069 | 0.0086 | 0.0051 | 0.0255 | 0.0919 | 0.6091 |
| overall_avg | PostHocPruning (|PSE|: N/A - no valid proxy-mediated paths remain after pruning (paper footnote ddagger)) | 0.0215 +/- 0.0010 | N/A | 0.0187 +/- 0.0007 | 0.0426 +/- 0.0015 | 0.0699 +/- 0.0147 | 0.6683 +/- 0.0132 |
| overall_avg | PF-Soft | 0.0045 | 0.0062 | 0.0107 | 0.0153 | 0.0728 | 0.6148 |
| overall_avg | PF-Hard+Soft | 0.0031 | 0.0012 | 0.0024 | 0.0131 | 0.0276 | 0.5194 |
