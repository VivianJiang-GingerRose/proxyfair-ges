# Modifications to causal-learn Package

## Upstream Package
- Package: causal-learn
- Original repository: https://github.com/py-why/causal-learn
- License: MIT
- Local fork location: `src/faircausal/causal_learn/causallearn`

## Project-Relevant Modifications

### 1) Fairness-aware GES entrypoint
- File: `causallearn/search/ScoreBased/GES_fair.py`
- Purpose: Add fairness-regularized scoring hooks used by ProxyFair Phase 1.
- Notes:
  - Accepts precomputed symmetric fairness penalties.
  - Applies fairness-weighted score adjustments during edge operations.
  - Keeps penalty evaluation orientation-invariant during CPDAG search.

### 2) Constraint handling integration
- File: `causallearn/search/ScoreBased/GES.py` and/or project-side wrapper usage
- Purpose: Ensure forbidden-edge constraints are respected during search.
- Notes:
  - Hard constraints are passed from project code (`prior_knowledge_processor.py`, `ges_runner_causal_learn.py`).
  - Blacklisted edges are blocked in candidate operations.

## How to Re-verify
1. Run baseline GES and fairness-enabled GES on the same dataset.
2. Confirm blacklisted edges do not appear in final learned graph.
3. Confirm fairness penalty path is active only when fairness caches are supplied.
