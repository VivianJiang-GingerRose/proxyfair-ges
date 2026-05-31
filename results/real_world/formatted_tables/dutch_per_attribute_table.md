# DUTCH Per-Attribute Results

- dataset name: dutch
- outcome variable: occupation
- protected attributes: sex_enc, country_birth, citizenship
- fairness metrics note: lower is better for |TE|, |PSE|, |NDE|, SPD, and EO
- uncertainty note: mean +/- std denotes variation over CPDAG-consistent DAG extensions (or pruned DAGs), not resampling uncertainty

| protected_attribute | method | |TE| | |PSE| | |NDE| | SPD | EO | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sex_enc | GES | 0.1743 +/- 0.0127 | 0.0000 +/- 0.0000 | 0.1743 +/- 0.0127 | 0.3178 +/- 0.0610 | 0.1859 +/- 0.0471 | 0.8338 +/- 0.0200 |
| sex_enc | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.1836 | N/A | 0.1836 | 0.3498 | 0.2098 | 0.8408 |
| sex_enc | GES+Hard | 0.0083 | 0.0029 | 0.0095 | 0.1005 | 0.1338 | 0.7131 |
| sex_enc | PostHocPruning (|PSE|: Reported over valid subset of proxy-mediated PSE paths (paper footnote ddagger)) | 0.1794 +/- 0.0088 | 0.0000 +/- 0.0000 | 0.1794 +/- 0.0088 | 0.3512 +/- 0.0469 | 0.2004 +/- 0.0559 | 0.8483 +/- 0.0164 |
| sex_enc | PF-Soft | 0.0300 | 0.0024 | 0.0237 | 0.0127 | 0.0354 | 0.8108 |
| sex_enc | PF-Hard+Soft | 0.0085 | 0.0049 | 0.0090 | 0.0277 | 0.0556 | 0.7151 |
| country_birth | GES | 0.0688 +/- 0.0387 | 0.0335 +/- 0.0236 | 0.0504 +/- 0.0081 | 0.1106 +/- 0.0651 | 0.1092 +/- 0.0509 | 0.8338 +/- 0.0200 |
| country_birth | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0233 | N/A | 0.0533 | 0.2363 | 0.1387 | 0.8408 |
| country_birth | GES+Hard | 0.0101 | 0.0240 | 0.0027 | 0.0578 | 0.1378 | 0.7131 |
| country_birth | PostHocPruning (|PSE|: Reported over valid subset of proxy-mediated PSE paths (paper footnote ddagger)) | 0.0232 +/- 0.0194 | 0.0304 +/- 0.0651 | 0.0217 +/- 0.0172 | 0.1114 +/- 0.0670 | 0.1152 +/- 0.0454 | 0.8483 +/- 0.0164 |
| country_birth | PF-Soft | 0.0214 | 0.0017 | 0.0197 | 0.0486 | 0.0307 | 0.8108 |
| country_birth | PF-Hard+Soft | 0.0050 | 0.0070 | 0.0076 | 0.1858 | 0.2697 | 0.7151 |
| citizenship | GES | 0.0108 +/- 0.0102 | 0.0013 +/- 0.0008 | 0.0107 +/- 0.0106 | 0.1689 +/- 0.0943 | 0.2551 +/- 0.1391 | 0.8338 +/- 0.0200 |
| citizenship | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0030 | N/A | 0.0041 | 0.1300 | 0.2511 | 0.8408 |
| citizenship | GES+Hard | 0.0233 | 0.0099 | 0.0172 | 0.1757 | 0.2609 | 0.7131 |
| citizenship | PostHocPruning (|PSE|: Reported over valid subset of proxy-mediated PSE paths (paper footnote ddagger)) | 0.0099 +/- 0.0051 | 0.0009 +/- 0.0006 | 0.0095 +/- 0.0054 | 0.1746 +/- 0.1048 | 0.2209 +/- 0.1581 | 0.8483 +/- 0.0164 |
| citizenship | PF-Soft | 0.0030 | 0.0041 | 0.0012 | 0.1810 | 0.3252 | 0.8108 |
| citizenship | PF-Hard+Soft | 0.0270 | 0.0117 | 0.0107 | 0.1440 | 0.3121 | 0.7151 |
| overall_avg | GES | 0.0846 +/- 0.0205 | 0.0116 +/- 0.0081 | 0.0784 +/- 0.0105 | 0.1991 +/- 0.0735 | 0.1834 +/- 0.0790 | 0.8338 +/- 0.0200 |
| overall_avg | GES+Phase2 (|PSE|: N/A - invalid protected-attribute orientations (paper footnote dagger)) | 0.0700 | N/A | 0.0804 | 0.2387 | 0.1999 | 0.8408 |
| overall_avg | GES+Hard | 0.0139 | 0.0123 | 0.0098 | 0.1113 | 0.1775 | 0.7131 |
| overall_avg | PostHocPruning (|PSE|: Reported over valid subset of proxy-mediated PSE paths (paper footnote ddagger)) | 0.0708 +/- 0.0111 | 0.0104 +/- 0.0219 | 0.0702 +/- 0.0105 | 0.2124 +/- 0.0729 | 0.1788 +/- 0.0865 | 0.8483 +/- 0.0164 |
| overall_avg | PF-Soft | 0.0181 | 0.0027 | 0.0149 | 0.0808 | 0.1304 | 0.8108 |
| overall_avg | PF-Hard+Soft | 0.0135 | 0.0079 | 0.0091 | 0.1192 | 0.2125 | 0.7151 |
