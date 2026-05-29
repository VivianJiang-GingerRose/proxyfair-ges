# Fair Causal Discovery ICDM: ProxyFair with Equivalence-Class Search

This repository contains the implementation used for the ProxyFair paper.

ProxyFair is a fairness-aware extension of Greedy Equivalence Search (GES) that combines:
- hard normative constraints (forbidden/required edges), and
- a continuous edge-level proxy penalty based on proxy-excess fraction.

The design follows a two-phase procedure to preserve score equivalence:
- Phase 1: CPDAG search with a symmetric fairness penalty.
- Phase 2: Fair DAG extraction inside the learned equivalence class.

## Core Algorithm Mapping

- Phase 1 scoring primitives: `src/faircausal/core/fairness_scoring.py`
  - `compute_edge_proxy_fractions(...)` computes edge-level proxy-excess fractions $\pi^\Sigma$.
  - `compute_outcome_relevance(...)` computes static outcome relevance $\rho_{adj}$.
  - `compute_symmetric_penalty_from_pi(...)` computes symmetric edge penalty $\Psi_{sym}$.
- Constrained GES orchestration: `src/faircausal/core/ges_runner_causal_learn.py`
- Phase 2 DAG selection and fairness analysis: `src/faircausal/core/counterfactual_fairness_runner.py`

## Repository Layout

- `src/faircausal/core/`: ProxyFair algorithm, GES orchestration, fairness analysis
- `src/faircausal/causal_learn/`: local causal-learn fork used by the project
- `src/experiments/`: experiment runners, plotting, and table compilation
- `src/faircausal/data/`: dataset loaders and synthetic generators
- `tests/synthetic/`: synthetic and integration tests
- `results/`: generated experiment outputs

## Reproducing Main Paper Artifacts

### Table 1 (Synthetic single-attribute)

```bash
poetry run python src/experiments/run_synthetic_main_grid.py --mode result1 --n-jobs -1
```

### Table 2 (Synthetic displacement)

```bash
poetry run python src/experiments/run_synthetic_main_grid.py --mode displacement --n-jobs -1
```

### Real-world runs (Law, COMPAS, Dutch, Bank)

```bash
pwsh -File scripts/run_all_real_datasets.ps1
```

### Compile causal-fairness tables from completed runs

```bash
poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset law
poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset compas
poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset dutch
poetry run python src/experiments/compile_causal_fairness_analysis_table.py --dataset bank
```

### Pareto plots (COMPAS)

```bash
pwsh -File scripts/run_compas_pareto_grid.ps1
poetry run python src/experiments/plot_pareto_front.py --dataset compas
```

## Quick Demo

Run a lightweight synthetic comparison between vanilla GES and ProxyFair:

```bash
poetry run python examples/demo_proxyfair.py
```

The demo prints TE/PSE-style structural diagnostics for a small run so ICDM readers can quickly verify behavior.

## Setup

### Prerequisites

- Python 3.11
- Poetry

### Install

```bash
git clone <repository-url>
cd fair-causal-discovery-icdm
poetry install
poetry run python --version
```

The repository directory in this workspace may appear as `FairCausalDiscovery_ICDM`; use your cloned folder name if it differs.

### Optional: Run Tests

```bash
poetry run pytest tests/synthetic -q
```

## Notes on LLM Constraint Elicitation

This repository also includes optional LLM tooling under `src/faircausal/llm/` used for role elicitation and constraint generation in some experiments. The ProxyFair algorithm itself does not require LLM inference at runtime when constraints are already provided.

Example LLM analysis command:

```bash
poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset bank
```
