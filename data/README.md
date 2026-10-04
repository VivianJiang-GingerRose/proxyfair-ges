# Data provenance and redistribution

The repository's root MIT license applies to the ProxyFair source code. The
datasets in this directory originate from third parties and remain subject to
their respective source terms. The committed files contain categorical,
processed analysis variables and no names or direct identifiers.

The real-world experiment runner uses the committed `*_preprocessed.csv` files
by default. Raw source data are not required to reproduce the experiments from
these processed inputs. To audit preprocessing from an independently obtained
raw file, use `--reprocess-data --raw-data-path PATH`; this processes the file
in memory and does not overwrite the committed artifact.

## Dataset inventory

| Dataset | Committed analysis file | Source and citation | Redistribution status | Transformation code |
|---|---|---|---|---|
| Bank Marketing | `data_bank_marketing/bank_preprocessed.csv` | S. Moro, P. Rita, and P. Cortez, *Bank Marketing*, UCI Machine Learning Repository, DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306) | UCI publishes the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve attribution when redistributing this processed derivative. | `src/faircausal/data/data_loader_bank.py` |
| COMPAS | `data_compas/compas_preprocessed.csv` | ProPublica, [COMPAS Analysis](https://github.com/propublica/compas-analysis) | The source repository does not state data redistribution terms in this project. Confirm the applicable source terms before public release. | `src/faircausal/data/data_loader_compas.py` |
| Law School (LSAC) | `data_law_school/law_school_preprocessed.csv` | Exact source URL and citation must be copied from the camera-ready paper's source records. | **Pending author verification. Do not publish this file until its redistribution terms are recorded here.** | `src/faircausal/data/data_loader_law.py` |
| Dutch Census | `data_dutch_census/dutch_preprocessed.csv` | Exact source URL and citation must be copied from the camera-ready paper's source records. | **Pending author verification. Do not publish this file until its redistribution terms are recorded here.** | `src/faircausal/data/data_loader_dutch.py` |

## Processed schemas

- Bank Marketing: 17 categorical columns; target `y`; protected attributes
  `age` and `marital`.
- COMPAS: 12 categorical columns; target `two_year_recid`; protected attributes
  `race_cat` and `sex`.
- Law School: 12 categorical columns; target `pass_bar`; protected attributes
  `race_cat` and `male_cat`.
- Dutch Census: 12 categorical columns; target `occupation`; protected
  attributes `sex_enc`, `country_birth`, and `citizenship`.

Before changing repository visibility to public, replace both pending entries
with the exact source URLs, citations, and redistribution terms used to obtain
the corresponding raw data.
