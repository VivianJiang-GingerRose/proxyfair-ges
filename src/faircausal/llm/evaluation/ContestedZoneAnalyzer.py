"""
Identify and analyze the "contested zone" - edges where normative frameworks disagree.
"""

import pandas as pd
from typing import List, Dict, Set, Tuple
from collections import defaultdict

try:
    from .ConstraintParser import ConstraintSet
except ImportError:
    from ConstraintParser import ConstraintSet


class ContestedZoneAnalyzer:
    """
    Identify edges where frameworks disagree and categorize by consensus level.
    
    Uses two-level majority voting:
    1. Within-framework: An edge is "blocked by framework X" if ≥2 out of 3 LLM providers block it
    2. Cross-framework: Categorize edges based on how many frameworks (using internal majority) block them
    
    With 4 frameworks:
    - Consensus Prohibited: ≥3 frameworks block (75-100%)
    - Low Blocking: ≤1 framework blocks (0-25%) - not enough consensus to block
    - Contested: 2 frameworks block (50%)
    """
    
    CONSENSUS_THRESHOLDS = {
        'prohibited': 0.75,   # ≥75% of frameworks block → consensus prohibited (≥3/4)
        'low_blocking': 0.25,    # ≤25% of frameworks block → low blocking / no consensus (≤1/4)
        # Between 25-75% (exactly 2/4 frameworks) → contested
    }
    
    def __init__(self, constraint_sets: List[ConstraintSet]):
        self.constraint_sets = constraint_sets
        self.frameworks = sorted(set(cs.framework for cs in constraint_sets))
        self.datasets = set(cs.dataset for cs in constraint_sets)
    
    def categorize_edges(
        self,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """
        Categorize all edges into consensus prohibited, consensus permitted, or contested.
        
        Uses two-level majority voting:
        1. Within each framework: edge is blocked if \u22652/3 LLM providers block it (majority)
        2. Count how many frameworks block the edge (using their internal majority)
        3. With 4 frameworks:
           - Consensus Prohibited: \u22653 frameworks block (\u226575%)
           - Consensus Permitted: \u22641 framework blocks (\u226425%)
           - Contested: 2 frameworks block (50%)
        
        Args:
            dataset: Dataset name to analyze
            constraint_type: Type of constraints ('fairness' for Type B only, 'all', 'type_a')
        
        Returns:
            Dict with keys: 'consensus_prohibited', 'rarely_blocked', 'contested'
        """
        # Get all edges across all frameworks
        all_edges = self._get_all_edges(dataset, constraint_type)
        
        # Compute blocking probability for each edge
        edge_probs = {}
        for edge in all_edges:
            p_blocked = self._compute_blocking_probability(
                edge, dataset, constraint_type
            )
            edge_probs[edge] = p_blocked
        
        # Categorize
        categorized = {
            'consensus_prohibited': set(),
            'rarely_blocked': set(),
            'contested': set()
        }
        
        for edge, p_blocked in edge_probs.items():
            if p_blocked >= self.CONSENSUS_THRESHOLDS['prohibited']:
                categorized['consensus_prohibited'].add(edge)
            elif p_blocked <= self.CONSENSUS_THRESHOLDS['low_blocking']:
                categorized['rarely_blocked'].add(edge)
            else:
                categorized['contested'].add(edge)
        
        return categorized
    
    def get_framework_breakdown(
        self,
        edge: Tuple[str, str],
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> Dict[str, bool]:
        """
        For a specific edge, show which frameworks block it.
        
        A framework blocks an edge if ANY provider (≥1 out of 3) blocks it.
        
        Returns:
            Dict mapping framework → blocked (bool)
        """
        breakdown = {}
        
        for framework in self.frameworks:
            # Get all runs for this framework
            sets = [cs for cs in self.constraint_sets 
                   if cs.framework == framework and cs.dataset == dataset]
            
            if not sets:
                breakdown[framework] = None
                continue
            
            # Count how many runs block this edge
            constraints_lists = [self._get_constraints(cs, constraint_type) 
                                for cs in sets]
            whitelisted_lists = [cs.whitelisted_edges for cs in sets]
            
            n_blocking = sum(
                1 for clist, wlist in zip(constraints_lists, whitelisted_lists)
                if edge in clist and edge not in wlist
            )
            
            # Framework blocks if ANY provider (≥1 out of 3) blocks it
            breakdown[framework] = n_blocking >= 1
        
        return breakdown
    
    def get_framework_llm_breakdown(
        self,
        edge: Tuple[str, str],
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> Dict[str, List[str]]:
        """
        For a specific edge, show which LLM providers within each framework block it.
        
        Returns:
            Dict mapping framework → list of provider names that blocked the edge
        """
        breakdown = {}
        
        for framework in self.frameworks:
            # Get all runs for this framework
            sets = [cs for cs in self.constraint_sets 
                   if cs.framework == framework and cs.dataset == dataset]
            
            if not sets:
                breakdown[framework] = []
                continue
            
            # Collect which providers block this edge
            blocking_providers = []
            for cs in sets:
                constraints = self._get_constraints(cs, constraint_type)
                # Check if this provider blocks the edge (and it's not whitelisted)
                if edge in constraints and edge not in cs.whitelisted_edges:
                    blocking_providers.append(cs.provider)
            
            breakdown[framework] = blocking_providers
        
        return breakdown
    
    def create_contested_zone_report(
        self,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> pd.DataFrame:
        """
        Create detailed report on edges that appear in blacklisted_type_b constraints.
        
        ONLY includes edges from blacklisted_type_b lists (no whitelisted edges).
        
        Uses two-level majority voting:
        1. Within each framework: edge is blocked if ≥2/3 LLM providers block it
        2. p_blocked = proportion of frameworks (using their internal majority) that block the edge
        3. Categorize: consensus_prohibited (≥0.75), rarely_blocked (≤0.25), contested (0.5)
        
        Returns:
            DataFrame with columns: edge, source, target, p_blocked, category, blocked_by_{framework},
            llms_blocking_{framework} (comma-separated list of providers), n_providers_blocking 
        """
        all_edges = self._get_all_edges(dataset, constraint_type)
        
        rows = []
        for edge in sorted(all_edges):
            p_blocked = self._compute_blocking_probability(edge, dataset, constraint_type)
            
            # Categorize
            if p_blocked >= self.CONSENSUS_THRESHOLDS['prohibited']:
                category = 'consensus_prohibited'
            elif p_blocked <= self.CONSENSUS_THRESHOLDS['low_blocking']:
                category = 'rarely_blocked'
            else:
                category = 'contested'
            
            # Get framework breakdown (boolean)
            breakdown = self.get_framework_breakdown(edge, dataset, constraint_type)
            
            # Get LLM provider breakdown (which specific LLMs within each framework)
            llm_breakdown = self.get_framework_llm_breakdown(edge, dataset, constraint_type)
            
            # Count total providers that block this edge (across all frameworks)
            n_providers_blocking = self._count_total_providers_blocking(edge, dataset, constraint_type)
            
            row_data = {
                'edge': f"{edge[0]} → {edge[1]}",
                'source': edge[0],
                'target': edge[1],
                'p_blocked': p_blocked,
                'category': category,
                'n_providers_blocking': n_providers_blocking,
            }
            
            # Add blocked_by columns (boolean)
            for fw, blocked in breakdown.items():
                row_data[f'blocked_by_{fw}'] = blocked
            
            # Add llms_blocking columns (comma-separated provider names)
            for fw, providers in llm_breakdown.items():
                row_data[f'llms_blocking_{fw}'] = ', '.join(sorted(providers)) if providers else ''
            
            rows.append(row_data)
        
        return pd.DataFrame(rows)
    
    def _count_total_providers_blocking(
        self,
        edge: Tuple[str, str],
        dataset: str,
        constraint_type: str
    ) -> int:
        """
        Count total number of LLM providers (across all frameworks) that block this edge.
        Maximum possible is 12 (4 frameworks × 3 providers each).
        """
        total_blocking = 0
        
        for cs in self.constraint_sets:
            if cs.dataset == dataset:
                constraints = self._get_constraints(cs, constraint_type)
                # Check if this provider blocks the edge (and it's not whitelisted)
                if edge in constraints and edge not in cs.whitelisted_edges:
                    total_blocking += 1
        
        return total_blocking
    
    def get_framework_specific_edges(
        self,
        dataset: str,
        constraint_type: str = 'fairness',
        consensus_threshold: float = 0.8
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """
        Identify edges that are uniquely blocked by each framework (with high consensus).
        
        Returns:
            Dict mapping framework → set of edges blocked only by that framework
        """
        framework_edges = {}
        
        # Get consensus constraints for each framework
        for framework in self.frameworks:
            sets = [cs for cs in self.constraint_sets 
                   if cs.framework == framework and cs.dataset == dataset]
            
            if not sets:
                framework_edges[framework] = set()
                continue
            
            # Get edges with high consensus in this framework
            all_constraints = [self._get_constraints(cs, constraint_type) for cs in sets]
            all_edges = set().union(*all_constraints) if all_constraints else set()
            
            consensus_edges = set()
            for edge in all_edges:
                appearances = sum(1 for clist in all_constraints if edge in clist)
                if appearances / len(sets) >= consensus_threshold:
                    consensus_edges.add(edge)
            
            framework_edges[framework] = consensus_edges
        
        # Find unique edges for each framework
        unique_edges = {}
        for framework in self.frameworks:
            other_frameworks = [fw for fw in self.frameworks if fw != framework]
            other_edges = set().union(*[framework_edges.get(fw, set()) 
                                       for fw in other_frameworks])
            unique_edges[framework] = framework_edges[framework] - other_edges
        
        return unique_edges
    
    def create_within_framework_agreement_report(
        self,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> pd.DataFrame:
        """
        Analyze within-framework LLM agreement/disagreement for each edge.
        
        For each edge and framework, count how many of the 3 LLM providers block it.
        This shows the internal consensus within each fairness framework.
        
        Returns:
            DataFrame with columns: edge, source, target, framework, n_blocking, 
            agreement_level (unanimous/majority/minority/none)
        """
        all_edges = self._get_all_edges(dataset, constraint_type)
        
        rows = []
        for edge in sorted(all_edges):
            for framework in self.frameworks:
                # Get all constraint sets for this framework and dataset
                sets = [cs for cs in self.constraint_sets 
                       if cs.framework == framework and cs.dataset == dataset]
                
                if not sets:
                    continue
                
                # Count how many LLM providers block this edge in this framework
                constraints_lists = [self._get_constraints(cs, constraint_type) for cs in sets]
                whitelisted_lists = [cs.whitelisted_edges for cs in sets]
                
                n_blocking = sum(
                    1 for clist, wlist in zip(constraints_lists, whitelisted_lists)
                    if edge in clist and edge not in wlist
                )
                
                n_total = len(sets)
                
                # Determine agreement level
                if n_blocking == n_total:
                    agreement = 'unanimous'  # All 3 LLMs agree to block
                elif n_blocking >= 2:
                    agreement = 'majority'   # 2/3 LLMs agree to block
                elif n_blocking == 1:
                    agreement = 'minority'   # Only 1/3 LLMs block
                else:
                    agreement = 'none'       # No LLMs block
                
                rows.append({
                    'edge': f"{edge[0]} → {edge[1]}",
                    'source': edge[0],
                    'target': edge[1],
                    'framework': framework,
                    'n_blocking': n_blocking,
                    'n_total': n_total,
                    'agreement_level': agreement
                })
        
        return pd.DataFrame(rows)
    
    def _get_all_edges(
        self,
        dataset: str,
        constraint_type: str
    ) -> Set[Tuple[str, str]]:
        """
        Get all edges that appear in blacklisted constraints for this dataset.
        ONLY includes edges from blacklisted_type_b (fairness constraints).
        Does NOT include whitelisted edges.
        """
        all_edges = set()
        
        for cs in self.constraint_sets:
            if cs.dataset == dataset:
                # Get edges based on constraint type (only blocked edges)
                edges = self._get_constraints(cs, constraint_type)
                all_edges.update(edges)
                
                # DO NOT include whitelisted edges - we only want blocked edges
        
        return all_edges
    
    def _compute_blocking_probability(
        self,
        edge: Tuple[str, str],
        dataset: str,
        constraint_type: str
    ) -> float:
        """
        Compute proportion of frameworks that block this edge.
        
        A framework is considered to "block" an edge if ANY provider (≥1 out of 3) blocks it.
        This captures all edges that appear in any blacklisted_type_b list.
        
        Returns:
            Float between 0.0 and 1.0 representing the proportion of frameworks blocking the edge
            (e.g., 0.75 means 3 out of 4 frameworks block it)
        """
        n_frameworks_blocking = 0
        n_frameworks_total = 0
        
        for framework in self.frameworks:
            # Get all runs for this framework and dataset
            sets = [cs for cs in self.constraint_sets 
                   if cs.framework == framework and cs.dataset == dataset]
            
            if not sets:
                continue
            
            n_frameworks_total += 1
            
            # Count how many LLM providers (runs) block this edge within the framework
            constraints_lists = [self._get_constraints(cs, constraint_type) for cs in sets]
            whitelisted_lists = [cs.whitelisted_edges for cs in sets]
            
            n_blocking_in_framework = sum(
                1 for clist, wlist in zip(constraints_lists, whitelisted_lists)
                if edge in clist and edge not in wlist
            )
            
            # Framework blocks the edge if ANY provider (≥1 out of 3) blocks it
            if n_blocking_in_framework >= 1:
                n_frameworks_blocking += 1
        
        if n_frameworks_total == 0:
            return 0.0
        
        return n_frameworks_blocking / n_frameworks_total
    
    def _get_constraints(
        self,
        constraint_set: ConstraintSet,
        constraint_type: str
    ) -> Set[Tuple[str, str]]:
        """Extract specific constraint type."""
        if constraint_type == 'fairness':
            return constraint_set.fairness_constraints_only
        elif constraint_type == 'all':
            return constraint_set.all_blacklisted
        elif constraint_type == 'type_a':
            return constraint_set.blacklisted_type_a
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")
    
    def analyze_within_framework_agreement(
        self,
        dataset: str,
        constraint_type: str = 'fairness'
    ) -> pd.DataFrame:
        """
        Analyze LLM provider agreement within each framework.
        
        For each framework, calculate:
        - Total number of unique edges blocked by ANY provider
        - For each edge, count how many of the 3 LLM providers blocked it
        - Agreement distribution: how many edges have 3/3, 2/3, 1/3 agreement
        
        Returns:
            DataFrame with columns: framework, edge, source, target, n_providers_blocking, agreement_level
        """
        rows = []
        
        for framework in self.frameworks:
            # Get all constraint sets for this framework and dataset
            sets = [cs for cs in self.constraint_sets 
                   if cs.framework == framework and cs.dataset == dataset]
            
            if not sets:
                continue
            
            # Get all edges that ANY provider blocked
            all_edges = set()
            for cs in sets:
                edges = self._get_constraints(cs, constraint_type)
                all_edges.update(edges)
            
            # For each edge, count provider agreement
            for edge in sorted(all_edges):
                n_providers_blocking = 0
                for cs in sets:
                    constraints = self._get_constraints(cs, constraint_type)
                    if edge in constraints and edge not in cs.whitelisted_edges:
                        n_providers_blocking += 1
                
                # Determine agreement level
                if n_providers_blocking == 3:
                    agreement_level = 'unanimous'
                elif n_providers_blocking == 2:
                    agreement_level = 'majority'
                else:
                    agreement_level = 'minority'
                
                rows.append({
                    'framework': framework,
                    'edge': f"{edge[0]} \u2192 {edge[1]}",
                    'source': edge[0],
                    'target': edge[1],
                    'n_providers_blocking': n_providers_blocking,
                    'agreement_level': agreement_level
                })
        
        return pd.DataFrame(rows)


# Example usage
if __name__ == "__main__":
    from .ConstraintParser import ConstraintParser
    from pathlib import Path
    
    # Parse constraints
    parser = ConstraintParser()
    all_constraints = []
    
    results_dir = Path("src/faircausal/llm/llm_responses/bank/")
    for framework in ['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic']:
        try:
            constraints = parser.parse_batch(results_dir, framework, 'bank')
            all_constraints.extend(constraints)
            print(f"Loaded {len(constraints)} constraint sets for {framework}")
        except Exception as e:
            print(f"Could not load {framework}: {e}")
    
    if all_constraints:
        # Analyze contested zone
        analyzer = ContestedZoneAnalyzer(all_constraints)
        
        # Categorize edges
        categories = analyzer.categorize_edges('bank')
        print(f"\nConsensus Prohibited: {len(categories['consensus_prohibited'])}")
        print(f"Rarely Blocked: {len(categories['rarely_blocked'])}")
        print(f"Contested: {len(categories['contested'])}")
        
        # Detailed report
        report = analyzer.create_contested_zone_report('bank')
        print("\nContested Zone Report:")
        print(report[report['category'] == 'contested'])
