from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from enum import Enum

class ConstraintType(Enum):
    HARD = "Hard Constraint"
    SOFT = "Soft Constraint"

class PriorKnowledgeProcessor:
    """
    A class to process prior knowledge constraints for causal discovery algorithms.
    Handles both temporal ordering and explicit edge constraints.
    """
    def __init__(self, node_names: List[str]):
        """
        Initialize the processor with node names.
        
        Args:
            node_names: List of variable names in the dataset
        """
        self.node_names = node_names
        self.node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        
    def process_temporal_constraints(self, 
                                  temporal_order: Dict[int, Tuple[List[str], str]]) -> np.ndarray:
        """
        Process temporal ordering constraints into a blacklist matrix.
        
        Args:
            temporal_order: Dictionary mapping tier numbers to (list of nodes, constraint type)
                Example: {
                    1: (["SEX", "AGE"], "Hard Constraint"),
                    2: (["MARRIAGE", "EDUCATION"], "Soft Constraint")
                }
                
        Returns:
            np.ndarray: Boolean matrix where True indicates a blacklisted edge
        """
        n_nodes = len(self.node_names)
        blacklist_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        
        # Convert temporal order to tier information
        tiers = {
            tier_num: (
                [self.node_to_idx[node] for node in nodes],
                ConstraintType.HARD if "Hard" in constraint else ConstraintType.SOFT
            )
            for tier_num, (nodes, constraint) in temporal_order.items()
        }
        
        # Process each tier
        sorted_tiers = sorted(tiers.keys())
        for i, current_tier in enumerate(sorted_tiers):
            current_indices, constraint_type = tiers[current_tier]
            
            # For hard constraints, block edges within the tier
            if constraint_type == ConstraintType.HARD:
                for idx1 in current_indices:
                    for idx2 in current_indices:
                        if idx1 != idx2:
                            blacklist_matrix[idx1, idx2] = True
            
            # Block edges from later tiers to earlier tiers
            for later_tier in sorted_tiers[i+1:]:
                later_indices, _ = tiers[later_tier]
                for later_idx in later_indices:
                    for current_idx in current_indices:
                        blacklist_matrix[later_idx, current_idx] = True
        
        return blacklist_matrix
    
    def process_blacklist(self, 
                         blacklist: List[Tuple[str, str]], 
                         existing_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process explicit blacklist constraints and combine with existing constraints.
        
        Args:
            blacklist: List of tuples (source_node, target_node) representing forbidden edges
            existing_matrix: Optional existing blacklist matrix to combine with
            
        Returns:
            np.ndarray: Combined boolean matrix where True indicates a blacklisted edge
        """
        n_nodes = len(self.node_names)
        if existing_matrix is None:
            blacklist_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        else:
            blacklist_matrix = existing_matrix.copy()
            
        # Process each blacklisted edge
        for source, target in blacklist:
            try:
                source_idx = self.node_to_idx[source]
                target_idx = self.node_to_idx[target]
                blacklist_matrix[source_idx, target_idx] = True
            except KeyError as e:
                print(f"Warning: Node {e} not found in node_names")
                continue
                
        return blacklist_matrix
    
    def process_whitelist(self, whitelist: List[Tuple[str, str]]) -> np.ndarray:
        """
        Process list of whitelisted edges into a boolean matrix.
        
        Args:
            whitelist: List of tuples (source, target) representing required edges
            
        Returns:
            Boolean matrix where True indicates a required edge
        """
        n_nodes = len(self.node_names)
        whitelist_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        
        # Create node name to index mapping
        node_indices = {name: i for i, name in enumerate(self.node_names)}
        
        # Process each whitelisted edge
        for src, dst in whitelist:
            if src in node_indices and dst in node_indices:
                i, j = node_indices[src], node_indices[dst]
                whitelist_matrix[i, j] = True
            else:
                print(f"Warning: Whitelisted edge ({src} -> {dst}) references unknown node(s)")
        
        return whitelist_matrix
    
    def process_all_constraints(self,
                            temporal_order: Optional[Dict[int, Tuple[List[str], str]]] = None,
                            blacklist: Optional[List[Tuple[str, str]]] = None,
                            whitelist: Optional[List[Tuple[str, str]]] = None) -> Dict:
        """
        Process temporal, blacklist, and whitelist constraints.
        
        Args:
            temporal_order: Dictionary of temporal ordering constraints
            blacklist: List of explicitly blacklisted edges
            whitelist: List of explicitly whitelisted edges
            
        Returns:
            Dict containing:
                - combined_blacklist: Final blacklist matrix
                - temporal_matrix: Matrix from temporal constraints only
                - explicit_matrix: Matrix from explicit blacklist only
                - whitelist: Matrix of whitelisted edges
        """
        n_nodes = len(self.node_names)
        temporal_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        explicit_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        whitelist_matrix = np.zeros((n_nodes, n_nodes), dtype=bool)
        
        # Process temporal constraints if provided
        if temporal_order:
            temporal_matrix = self.process_temporal_constraints(temporal_order)
        
        # Process explicit blacklist if provided
        if blacklist:
            explicit_matrix = self.process_blacklist(blacklist)
        
        # Process whitelist if provided
        if whitelist:
            whitelist_matrix = self.process_whitelist(whitelist)
            
            # Check for conflicts between whitelist and blacklist
            conflicts = np.where(whitelist_matrix & (temporal_matrix | explicit_matrix))
            if len(conflicts[0]) > 0:
                conflict_edges = []
                for i, j in zip(conflicts[0], conflicts[1]):
                    conflict_edges.append((self.node_names[i], self.node_names[j]))
                warnings.warn(f"Conflicts detected between whitelist and blacklist edges: {conflict_edges}")
        
        # Combine constraints
        combined_blacklist = temporal_matrix | explicit_matrix
        
        return {
            'combined_blacklist': combined_blacklist,
            'temporal_matrix': temporal_matrix,
            'explicit_matrix': explicit_matrix,
            'whitelist': whitelist_matrix
        }

def validate_constraints(node_names: List[str],
                       temporal_order: Optional[Dict[int, Tuple[List[str], str]]] = None,
                       blacklist: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Validate the format and content of constraint specifications.
    
    Args:
        node_names: List of all variable names
        temporal_order: Dictionary of temporal ordering constraints
        blacklist: List of explicitly blacklisted edges
        
    Raises:
        ValueError: If constraints are invalid
    """
    node_set = set(node_names)
    
    # Validate temporal ordering
    if temporal_order:
        for tier, (nodes, constraint_type) in temporal_order.items():
            # Check that all nodes exist
            unknown_nodes = set(nodes) - node_set
            if unknown_nodes:
                raise ValueError(f"Nodes in tier {tier} not found in data: {unknown_nodes}")
            
            # Validate constraint type
            if constraint_type not in [ConstraintType.HARD.value, ConstraintType.SOFT.value]:
                raise ValueError(
                    f"Invalid constraint type in tier {tier}: {constraint_type}. "
                    "Must be either 'Hard Constraint' or 'Soft Constraint'"
                )
    
    # Validate blacklist
    if blacklist:
        for source, target in blacklist:
            if source not in node_set:
                raise ValueError(f"Source node in blacklist not found: {source}")
            if target not in node_set:
                raise ValueError(f"Target node in blacklist not found: {target}")