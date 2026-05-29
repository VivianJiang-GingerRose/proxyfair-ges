"""
Analyze constraint stability within and across normative frameworks.
Computes Jaccard similarity, Krippendorff's alpha, and agreement rates.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from itertools import combinations

from .ConstraintParser import ConstraintSet


class StabilityAnalyzer:
    """
    Compute stability metrics for constraint sets.
    """
    
    def __init__(self, constraint_sets: List[ConstraintSet]):
        """
        Args:
            constraint_sets: List of parsed constraint sets to analyze
        """
        self.constraint_sets = constraint_sets
        self.datasets = set(cs.dataset for cs in constraint_sets)
        self.frameworks = set(cs.framework for cs in constraint_sets)
    
    def within_framework_stability(
        self,
        framework: str,
        dataset: str,
        constraint_type: str = 'fairness'  # 'fairness', 'all', 'type_a'
    ) -> Dict[str, float]:
        """
        Compute stability metrics for a single framework across multiple runs.
        
        Args:
            framework: Framework name
            dataset: Dataset name
            constraint_type: Which constraints to analyze
            
        Returns:
            Dict with metrics: {
                'agreement_rate': float,
                'krippendorff_alpha': float,
                'mean_jaccard': float,
                'std_jaccard': float,
                'n_runs': int,
                'n_stable_edges': int  # edges appearing in >80% of runs
            }
        """
        # Filter to relevant constraint sets
        sets = [
            cs for cs in self.constraint_sets
            if cs.framework == framework and cs.dataset == dataset
        ]
        
        if len(sets) < 2:
            raise ValueError(f"Need at least 2 runs for {framework}/{dataset}, found {len(sets)}")
        
        # Extract constraint sets based on type
        constraint_lists = [self._get_constraints(cs, constraint_type) for cs in sets]
        
        # Compute metrics
        metrics = {}
        
        # 1. Agreement rate (edge-level)
        metrics['agreement_rate'] = self._compute_agreement_rate(constraint_lists)
        
        # 2. Krippendorff's alpha (treating each edge as a binary rating task)
        metrics['krippendorff_alpha'] = self._compute_krippendorff_alpha(constraint_lists)
        
        # 3. Pairwise Jaccard similarity
        jaccard_scores = self._compute_pairwise_jaccard(constraint_lists)
        metrics['mean_jaccard'] = np.mean(jaccard_scores) if jaccard_scores else 1.0
        metrics['std_jaccard'] = np.std(jaccard_scores) if jaccard_scores else 0.0
        
        # 4. Stable edges (consensus)
        metrics['n_stable_edges'] = self._count_stable_edges(constraint_lists, threshold=0.8)
        metrics['n_runs'] = len(sets)
        
        return metrics
    
    def between_framework_divergence(
        self,
        framework_a: str,
        framework_b: str,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> Dict[str, any]:
        """
        Measure divergence between two frameworks.
        
        Returns:
            Dict with metrics: {
                'jaccard_similarity': float,
                'unique_to_a': Set[Tuple],
                'unique_to_b': Set[Tuple],
                'shared': Set[Tuple],
                'divergence_rate': float  # 1 - jaccard
            }
        """
        # Get consensus constraints for each framework
        sets_a = [cs for cs in self.constraint_sets 
                  if cs.framework == framework_a and cs.dataset == dataset]
        sets_b = [cs for cs in self.constraint_sets 
                  if cs.framework == framework_b and cs.dataset == dataset]
        
        # Take consensus (edges appearing in >50% of runs)
        consensus_a = self._get_consensus_constraints(sets_a, constraint_type, threshold=0.5)
        consensus_b = self._get_consensus_constraints(sets_b, constraint_type, threshold=0.5)
        
        # Compute set differences
        shared = consensus_a & consensus_b
        unique_a = consensus_a - consensus_b
        unique_b = consensus_b - consensus_a
        
        # Jaccard similarity
        union = consensus_a | consensus_b
        jaccard = len(shared) / len(union) if union else 1.0
        
        return {
            'jaccard_similarity': jaccard,
            'divergence_rate': 1 - jaccard,
            'unique_to_a': unique_a,
            'unique_to_b': unique_b,
            'shared': shared,
            'n_consensus_a': len(consensus_a),
            'n_consensus_b': len(consensus_b)
        }
    
    def all_frameworks_comparison(
        self,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> pd.DataFrame:
        """
        Create pairwise comparison matrix for all frameworks.
        
        Returns:
            DataFrame with Jaccard similarities between all framework pairs
        """
        frameworks = sorted(self.frameworks)
        n = len(frameworks)
        
        # Initialize matrix
        jaccard_matrix = np.zeros((n, n))
        
        for i, fw_a in enumerate(frameworks):
            for j, fw_b in enumerate(frameworks):
                if i == j:
                    jaccard_matrix[i, j] = 1.0
                elif i < j:
                    try:
                        metrics = self.between_framework_divergence(
                            fw_a, fw_b, dataset, constraint_type
                        )
                        jaccard_matrix[i, j] = metrics['jaccard_similarity']
                        jaccard_matrix[j, i] = jaccard_matrix[i, j]  # Symmetric
                    except Exception as e:
                        print(f"Warning: Could not compare {fw_a} and {fw_b}: {e}")
                        jaccard_matrix[i, j] = np.nan
                        jaccard_matrix[j, i] = np.nan
        
        return pd.DataFrame(
            jaccard_matrix,
            index=frameworks,
            columns=frameworks
        )
    
    def _get_constraints(
        self,
        constraint_set: ConstraintSet,
        constraint_type: str
    ) -> Set[Tuple[str, str]]:
        """Extract specific constraint type from ConstraintSet."""
        if constraint_type == 'fairness':
            return constraint_set.fairness_constraints_only
        elif constraint_type == 'all':
            return constraint_set.all_blacklisted
        elif constraint_type == 'type_a':
            return constraint_set.blacklisted_type_a
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
    
    def _compute_agreement_rate(
        self,
        constraint_lists: List[Set[Tuple[str, str]]]
    ) -> float:
        """
        Compute simple agreement rate: fraction of edge decisions that are unanimous.
        """
        # Get all edges that appear in at least one run
        all_edges = set().union(*constraint_lists) if constraint_lists else set()
        
        if not all_edges:
            return 1.0  # Vacuous agreement if no edges
        
        # Count unanimous decisions
        unanimous_count = 0
        for edge in all_edges:
            appearances = sum(1 for clist in constraint_lists if edge in clist)
            if appearances == 0 or appearances == len(constraint_lists):
                unanimous_count += 1
        
        return unanimous_count / len(all_edges)
    
    def _compute_krippendorff_alpha(
        self,
        constraint_lists: List[Set[Tuple[str, str]]]
    ) -> float:
        """
        Compute Krippendorff's alpha for inter-rater reliability.
        
        Treats each edge as an item, each run as a rater, and blocked/allowed as binary rating.
        """
        try:
            import krippendorff
        except ImportError:
            print("Warning: krippendorff package not installed. Skipping alpha calculation.")
            return np.nan
        
        # Get all edges
        all_edges = sorted(set().union(*constraint_lists)) if constraint_lists else []
        
        if not all_edges:
            return 1.0
        
        # Create reliability data matrix
        # Rows: edges, Cols: runs, Values: 1=blocked, 0=allowed
        n_edges = len(all_edges)
        n_runs = len(constraint_lists)
        
        data = np.zeros((n_edges, n_runs))
        for i, edge in enumerate(all_edges):
            for j, clist in enumerate(constraint_lists):
                data[i, j] = 1 if edge in clist else 0
        
        # Krippendorff's alpha (binary data)
        try:
            alpha = krippendorff.alpha(
                reliability_data=data.T,  # Transpose: rows=raters, cols=items
                level_of_measurement='nominal'
            )
            return alpha
        except Exception as e:
            print(f"Warning: Krippendorff's alpha calculation failed: {e}")
            return np.nan
    
    def _compute_pairwise_jaccard(
        self,
        constraint_lists: List[Set[Tuple[str, str]]]
    ) -> List[float]:
        """Compute Jaccard similarity for all pairs of constraint lists."""
        jaccard_scores = []
        
        for set_a, set_b in combinations(constraint_lists, 2):
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 1.0
            jaccard_scores.append(jaccard)
        
        return jaccard_scores
    
    def _count_stable_edges(
        self,
        constraint_lists: List[Set[Tuple[str, str]]],
        threshold: float = 0.8
    ) -> int:
        """Count edges that appear in at least `threshold` fraction of runs."""
        all_edges = set().union(*constraint_lists) if constraint_lists else set()
        n_runs = len(constraint_lists)
        
        if n_runs == 0:
            return 0
        
        stable_count = 0
        for edge in all_edges:
            appearances = sum(1 for clist in constraint_lists if edge in clist)
            if appearances / n_runs >= threshold:
                stable_count += 1
        
        return stable_count
    
    def _get_consensus_constraints(
        self,
        constraint_sets: List[ConstraintSet],
        constraint_type: str,
        threshold: float = 0.5
    ) -> Set[Tuple[str, str]]:
        """
        Get consensus constraints (edges appearing in ≥threshold fraction of runs).
        """
        if not constraint_sets:
            return set()
        
        all_constraints = [self._get_constraints(cs, constraint_type) 
                          for cs in constraint_sets]
        
        all_edges = set().union(*all_constraints) if all_constraints else set()
        n_runs = len(all_constraints)
        
        if n_runs == 0:
            return set()
        
        consensus = set()
        for edge in all_edges:
            appearances = sum(1 for clist in all_constraints if edge in clist)
            if appearances / n_runs >= threshold:
                consensus.add(edge)
        
        return consensus


# Example usage
if __name__ == "__main__":
    from .ConstraintParser import ConstraintParser
    from pathlib import Path
    
    # Parse constraint sets
    parser = ConstraintParser()
    results_dir = Path("src/faircausal/llm/llm_responses/bank/")
    
    all_constraints = []
    for framework in ['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic']:
        try:
            constraints = parser.parse_batch(results_dir, framework, 'bank')
            all_constraints.extend(constraints)
            print(f"Loaded {len(constraints)} constraint sets for {framework}")
        except Exception as e:
            print(f"Could not load {framework}: {e}")
    
    if all_constraints:
        # Analyze stability
        analyzer = StabilityAnalyzer(all_constraints)
        
        # Within-framework stability
        try:
            metrics = analyzer.within_framework_stability('Statistical', 'bank')
            print(f"\nStatistical stability: {metrics}")
        except Exception as e:
            print(f"Could not compute stability: {e}")
