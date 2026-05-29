# Standard library imports
from typing import Optional, Dict, Any, List, Tuple, Union, Set, Callable
from enum import Enum
from math import log
from dataclasses import dataclass

# Third-party library imports
import numpy as np

# Local application/library imports
from src.faircausal.causal_learn.causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph
from src.faircausal.causal_learn.causallearn.graph.GraphNode import GraphNode
from src.faircausal.causal_learn.causallearn.graph.Dag import Dag
from src.faircausal.causal_learn.causallearn.utils.DAG2CPDAG_fair import dag2cpdag_with_blacklist
from src.faircausal.causal_learn.causallearn.utils.GESUtils_fair import *
from src.faircausal.causal_learn.causallearn.utils.PDAG2DAG_fair import pdag2dag
from src.faircausal.core.fairness_scoring import (
    compute_edge_proxy_fractions,
    compute_outcome_relevance,
    compute_symmetric_penalty_from_pi,
    compute_fairness_penalty,
    fairness_adjusted_delta,
)
import warnings
from copy import deepcopy

@dataclass
class ScoreCache:
    """Cache for storing precomputed values"""
    covariance_matrix: np.ndarray
    sample_size: int
    node_variances: np.ndarray

class OptimizedBICScore:
    def __init__(self, data: np.ndarray):
        """
        Initialize scoring mechanism for discrete/categorical data.
        
        Args:
            data: Input data matrix where all variables are categorical
        """
        self.data = data
        self.sample_size = data.shape[0]
        
        # Check if data appears to be categorical
        self._validate_categorical_data()
        
        # Store unique values and their counts for each variable
        self.unique_values = [np.unique(data[:, i]) for i in range(data.shape[1])]
        self.n_categories = [len(vals) for vals in self.unique_values]
        
        # Create mappings from original values to 0-based indices for each variable
        self.value_to_index = [
            {val: idx for idx, val in enumerate(vals)} 
            for vals in self.unique_values
        ]
    
    def _validate_categorical_data(self):
        """Check if data appears to be categorical by examining unique value counts"""
        for col in range(self.data.shape[1]):
            unique_vals = np.unique(self.data[:, col])
            # If there are too many unique values relative to sample size
            # it's probably not categorical data
            if len(unique_vals) > min(100, self.sample_size / 10):
                warnings.warn(
                    f"Column {col} has {len(unique_vals)} unique values, which may indicate "
                    f"it's not categorical. BIC score assumes categorical data."
                )
    
    def score(self, i: int, PAi: Union[list, np.ndarray]) -> float:
        """
        Compute local score for categorical node i given parents PAi.
        
        Args:
            i: Target variable index
            PAi: List of parent indices
            
        Returns:
            float: BIC score for the relationship
        """
        PAi = list(PAi) if isinstance(PAi, np.ndarray) else PAi
        
        # Validate inputs
        if not (0 <= i < self.data.shape[1]):
            raise ValueError(f"Target index {i} is out of range for data with {self.data.shape[1]} columns")
        
        for p in PAi:
            if not (0 <= p < self.data.shape[1]):
                raise ValueError(f"Parent index {p} is out of range for data with {self.data.shape[1]} columns")
        
        return self._categorical_score(i, PAi)

    def _categorical_score(self, i: int, PAi: list) -> float:
        """
        Compute score for categorical variable using contingency tables.
        
        Args:
            i: Target variable index
            PAi: List of parent indices
            
        Returns:
            float: BIC score for the categorical variable
        """
        target_var = self.data[:, i]
        n_categories = self.n_categories[i]
        value_to_index = self.value_to_index[i]
        
        if not PAi:
            # No parents: Use multinomial likelihood
            counts = np.zeros(n_categories)
            for val in target_var:
                counts[value_to_index[val]] += 1
            
            non_zero_counts = counts[counts > 0]
            log_likelihood = np.sum(non_zero_counts * np.log(non_zero_counts)) - self.sample_size * np.log(self.sample_size)
            penalty = -0.5 * (n_categories - 1) * np.log(self.sample_size)
            return log_likelihood + penalty
        
        # Initialize variables for counting
        log_likelihood = 0
        
        # Create contingency table for all parent configurations
        from collections import defaultdict
        config_counts = defaultdict(lambda: np.zeros(n_categories))
        
        for row_idx in range(self.sample_size):
            # Create parent configuration tuple
            parent_values = tuple(self.data[row_idx, dp] for dp in PAi)
            target_value = target_var[row_idx]
            
            # Map the target value to its index
            target_idx = value_to_index[target_value]
            
            # Update counts
            config_counts[parent_values][target_idx] += 1
        
        # Calculate likelihood for each observed parent configuration
        n_parent_configs = len(config_counts)
        for counts in config_counts.values():
            config_sum = np.sum(counts)
            if config_sum > 0:
                non_zero_counts = counts[counts > 0]
                log_likelihood += np.sum(non_zero_counts * np.log(non_zero_counts)) - config_sum * np.log(config_sum)
        
        # Calculate BIC penalty
        n_params = (n_categories - 1) * n_parent_configs
        penalty = -0.5 * n_params * np.log(self.sample_size)

        # Return negative BIC to match original ges implementation in causal learn
        bic_score = log_likelihood + penalty
        
        return -bic_score

def local_score_BIC_optimized(data: np.ndarray, i: int, PAi: list, parameters: Dict) -> float:
    """
    Wrapper function that matches the interface expected by GES algorithm
    for discrete data only
    """
    score_computer = parameters.get('score_computer')
    if score_computer is None:
        score_computer = OptimizedBICScore(data)
        parameters['score_computer'] = score_computer
    
    # Convert PAi to list if it's not already
    PAi = list(PAi) if isinstance(PAi, np.ndarray) else PAi
    
    # Ensure i and PAi elements are valid indices
    assert 0 <= i < data.shape[1], f"Invalid target index {i} for data with {data.shape[1]} columns"
    for p in PAi:
        assert 0 <= p < data.shape[1], f"Invalid parent index {p} for data with {data.shape[1]} columns"
    
    # Call score
    return -score_computer.score(i, PAi)

def ges(
    X: ndarray,
    score_func: str = "local_score_BIC",
    maxP: Optional[float] = None,
    parameters: Optional[Dict[str, Any]] = None,
    node_names: Union[List[str], None] = None,
    blacklist_matrix: Optional[np.ndarray] = None,
    whitelist_matrix: Optional[np.ndarray] = None,
    debug: bool = True,
    log_file_handle: Optional[Any] = None,
    fairness_lambda: Optional[float] = None,
    fairness_tau_c: float = 5.0,
    protected_attr_indices: Optional[List[int]] = None,
    outcome_index: Optional[int] = None,
    path_effect_method: str = "distance",
) -> Dict[str, Any]:
    """
    Perform greedy equivalence search (GES) algorithm for discrete data.
    Handles blacklisted edges during structure learning.

    Parameters
    ----------
    X : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of samples and n_features is the number of features.
    score_func : the string name of score function. (default: "local_score_BIC")
    maxP : allowed maximum number of parents when searching the graph
    parameters : Optional parameters for the scoring function
    node_names : Optional[List[str]]
        Names of the variables
    blacklist_matrix : Optional[np.ndarray]
        Boolean matrix where True indicates a forbidden edge
    debug : bool
        Whether to print debugging information
    log_file_handle : file object, optional
        File object to write log messages to
    
    Returns
    -------
    Record['G']: learned causal graph, where Record['G'].graph[j,i]=1 and Record['G'].graph[i,j]=-1 indicates  i --> j ,
                    Record['G'].graph[i,j] = Record['G'].graph[j,i] = -1 indicates i --- j.
    Record['update1']: each update (Insert operator) in the forward step
    Record['update2']: each update (Delete operator) in the backward step
    Record['G_step1']: learned graph at each step in the forward step
    Record['G_step2']: learned graph at each step in the backward step
    Record['score']: the score of the learned graph
    """
    if X.shape[0] < X.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    # Get number of variables/features
    N = X.shape[1]

    def log_message(msg):
        """Helper function to handle both console and file logging"""
        if debug:
            print(msg)
        if log_file_handle:
            print(msg, file=log_file_handle)
            log_file_handle.flush()

    log_message("=" * 60)
    log_message("STARTING GES ALGORITHM WITH DEBUGGING")
    log_message(f"Number of variables: {N}")
    if blacklist_matrix is not None:
        n_blacklisted = np.sum(blacklist_matrix)
        log_message(f"Blacklist provided with {n_blacklisted} forbidden edges")
    log_message("=" * 60)

    # Initialize parameters as empty dict if None
    if parameters is None:
        parameters = {}
    
    if score_func == "local_score_BIC":
        # Create score computer only once for discrete data
        score_computer = OptimizedBICScore(X)
        parameters['score_computer'] = score_computer

        # Set default maxP for BIC score if not provided
        if maxP is None:
            maxP = X.shape[1] / 2  # maximum number of parents is half the number of variables
        
        # Create local score class with our optimized function
        localScoreClass = LocalScoreClass(
            data=X,
            local_score_fun=local_score_BIC_optimized,
            parameters=parameters
        )
    elif score_func == "local_score_BDeu":  # Greedy equivalence search with BDeu score
        if maxP is None:
            maxP = X.shape[1] / 2
        N = X.shape[1]  # number of variables
        localScoreClass = LocalScoreClass(
            data=X, local_score_fun=local_score_BDeu, parameters=None
        )
    else:
        raise Exception("Only BIC and BDeu scores are supported for discrete data!")
    
    score_func = localScoreClass

    # Initialize graph
    if node_names is None:
        node_names = [f"X{i+1}" for i in range(N)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)

    # Add whitelisted edges to initial graph
    if whitelist_matrix is not None:
        log_message("Adding whitelisted edges to initial graph...")
        for i in range(N):
            for j in range(N):
                if whitelist_matrix[i, j] and i != j:
                    # Add directed edge from i to j
                    G.graph[i, j] = Endpoint.TAIL.value
                    G.graph[j, i] = Endpoint.ARROW.value
                    log_message(f"  Added whitelisted edge from {node_names[i]} to {node_names[j]}")


    # Initialize score
    score = score_g(X, G, score_func, parameters)
    
    log_message(f"Initial graph score: {score:.4f}\n")

    # Precompute Phase 1 symmetric penalty (constant throughout search)
    _pi_sigma = None
    _psi_sym = None
    _fairness_edge_logs: List[Dict[str, Any]] = []
    _protected_names: List[str] = []
    _fairness_enabled = (
        fairness_lambda is not None and fairness_lambda > 0
        and protected_attr_indices is not None
        and outcome_index is not None
    )
    if _fairness_enabled:
        _protected_names = [node_names[idx] for idx in protected_attr_indices]
        _pi_sigma = compute_edge_proxy_fractions(
            X,
            protected_attr_indices=protected_attr_indices,
            tau_c=fairness_tau_c,
            min_group_size=10,
            var_names=node_names,
            logger=log_message,
        )
        _rho_adj = compute_outcome_relevance(X, outcome_index)
        _psi_sym = compute_symmetric_penalty_from_pi(_pi_sigma, _rho_adj)
        log_message(
            f"Fairness regularization enabled: lambda={fairness_lambda}, "
            f"tau_c={fairness_tau_c}, lambda*N={fairness_lambda * X.shape[0]:.4f}"
        )
        log_message(f"Protected attributes: {_protected_names}")
        log_message(f"Outcome node: {node_names[outcome_index]}")

    # Convert initial graph
    if blacklist_matrix is not None:
        G = pdag2dag(G, blacklist_matrix)
    else:
        G = pdag2dag(G)
    G = dag2cpdag_with_blacklist(G, blacklist_matrix)

    ## --------------------------------------------------------------------
    ## forward greedy search
    log_message("\n" + "=" * 60 + "\n")
    log_message("STARTING FORWARD PHASE (EDGE INSERTION)\n")
    log_message("=" * 60 + "\n")
        
    record_local_score = [[] for i in range(N)]  # record the local score calculated each time
    score_new = score
    count1 = 0
    update1 = []
    G_step1 = []
    score_record1 = []
    graph_record1 = []

    while True:
        count1 = count1 + 1
        score = score_new
        score_record1.append(score)
        graph_record1.append(G)
        min_chscore = 1e7
        min_desc = []

        log_message(f"\nForward iteration {count1}\n")
        log_message(f"Current score: {score:.4f}\n")

        for i in range(N):
            for j in range(N):
                # Check if edge is blacklisted or already exists
                if (blacklist_matrix is None or not blacklist_matrix[i, j]) and \
                   G.graph[i, j] == Endpoint.NULL.value and \
                   G.graph[j, i] == Endpoint.NULL.value and \
                   i != j and \
                   len(np.where(G.graph[j, :] == Endpoint.ARROW.value)[0]) <= maxP:
                    
                    # Always log which edge we're considering
                    log_message(f"  Considering edge between nodes {node_names[i]} and {node_names[j]}\n")
                    
                    # Find neighbors for validity tests
                    Tj = np.intersect1d(
                        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
                    )  # neighbors of Xj
                    Ti = np.union1d(
                        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                    )  # adjacent to Xi
                    NTi = np.setdiff1d(np.arange(N), Ti)
                    T0 = np.intersect1d(Tj, NTi)  # neighbors of Xj that are not adjacent to Xi

                    # Check subsets for validity
                    sub = Combinatorial(T0.tolist())  # find all the subsets for T0
                    S = np.zeros(len(sub))  # S indicates whether we need to check subset(k)
                    
                    edge_valid = False  # Flag to track if any valid configuration was found
                    best_subset_score = 1e7
                    best_subset = []

                    for k in range(len(sub)):
                        if S[k] < 2:  # S indicate whether we need to check subset(k)
                            V1 = insert_validity_test1(G, i, j, sub[k])  # Test condition 1
                            if V1:
                                if not S[k]:
                                    V2 = insert_validity_test2(G, i, j, sub[k])  # Test condition 2
                                else:
                                    V2 = 1
                                if V2:
                                    Idx = find_subset_include(sub[k], sub)  # Find subsets that include sub(k)
                                    S[np.where(Idx == 1)] = 1
                                    chscore, desc, record_local_score = insert_changed_score(
                                        X, G, i, j, sub[k], record_local_score, score_func, parameters
                                    )
                                    
                                    if _fairness_enabled:
                                        # Phase 1: use precomputed symmetric penalty Ψ_sym(i, j).
                                        # Whitelisted edges are exempted: they cannot be removed
                                        # regardless of score, so penalising them would distort
                                        # the relative attractiveness of other edges.
                                        _is_whitelisted = (
                                            whitelist_matrix is not None
                                            and (whitelist_matrix[i, j] or whitelist_matrix[j, i])
                                        )
                                        if not _is_whitelisted:
                                            _psi = _psi_sym.get((i, j), 0.0)
                                            _bic_only = chscore
                                            chscore = fairness_adjusted_delta(
                                                chscore, _psi, X.shape[0], fairness_lambda
                                            )
                                            if _psi > 0:
                                                _penalty_term = fairness_lambda * X.shape[0] * _psi
                                                log_message(f"      [Fairness] BIC_delta={_bic_only:.4f}, psi_sym={_psi:.4f}, penalty={_penalty_term:.4f}, adjusted={chscore:.4f}\n")
                                            _fairness_edge_logs.append(
                                                {
                                                    "phase": "forward",
                                                    "edge_i": int(i),
                                                    "edge_j": int(j),
                                                    "edge_i_name": node_names[i],
                                                    "edge_j_name": node_names[j],
                                                    "subset": "|".join(node_names[idx] for idx in sub[k]),
                                                    "bic_delta": float(_bic_only),
                                                    "psi_sym": float(_psi),
                                                    "penalty_term": float(fairness_lambda * X.shape[0] * _psi),
                                                    "adjusted_delta": float(chscore),
                                                }
                                            )
                                    
                                    edge_valid = True
                                    subset_str = [node_names[idx] for idx in sub[k]]
                                    log_message(f"    Subset {subset_str} score change: {chscore:.4f}\n")
                                    
                                    if chscore < best_subset_score:
                                        best_subset_score = chscore
                                        best_subset = subset_str
                                        
                                    if chscore < min_chscore:
                                        min_chscore = chscore
                                        min_desc = desc
                            else:
                                Idx = find_subset_include(sub[k], sub)  # Find subsets that include sub(k)
                                S[np.where(Idx == 1)] = 2
                    
                    # After evaluating all subsets, log the best one for this edge
                    if edge_valid:
                        log_message(f"    Best subset for this edge: {best_subset} with score change: {best_subset_score:.4f}\n")
                    else:
                        log_message(f"    No valid configuration found for this edge\n")

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new  <= 1e-8:
                log_message("  No significant score improvement. Ending forward phase.\n")
                break
                
            # Apply the operator
            src_node = node_names[min_desc[0]]
            dst_node = node_names[min_desc[1]]
            subset_nodes = [node_names[idx] for idx in min_desc[2]]
            log_message(f"  INSERTING edge from {src_node} to {dst_node} with subset: {subset_nodes}\n")
            log_message(f"  Score change: {min_chscore:.4f}, New score: {score_new:.4f}\n")
                
            G = insert(G, min_desc[0], min_desc[1], min_desc[2])
            update1.append([min_desc[0], min_desc[1], min_desc[2]])

            # Store original graph state
            G_orig = deepcopy(G)

            # Perform graph conversions with blacklist handling
            try:
                if blacklist_matrix is not None:
                    G_dag = pdag2dag(G, blacklist_matrix)
                else:
                    G_dag = pdag2dag(G)
                G = dag2cpdag_with_blacklist(G_dag, blacklist_matrix)
            except ValueError as e:
                # If conversion fails, revert to original state
                G = G_orig
                log_message(f"  WARNING: Graph conversion failed: {str(e)}\n")
                log_message("  Reverting to previous valid graph. Ending forward phase.\n")
                break
                
            G_step1.append(deepcopy(G))
        else:
            score_new = score
            log_message("  No valid insertions found. Ending forward phase.\n")
            break

    log_message("\nForward phase summary:\n")
    log_message(f"Number of iterations: {count1}\n")
    log_message(f"Number of edge insertions: {len(update1)}\n")
    log_message(f"Final score after forward phase: {score_new:.4f}\n")
        
    # Display edge insertions
    if update1:
        log_message("\nEdges inserted:\n")
        for idx, upd in enumerate(update1):
            src_node = node_names[upd[0]]
            dst_node = node_names[upd[1]]
            subset_nodes = [node_names[idx] for idx in upd[2]]
            log_message(f"  {idx+1}. {src_node} --> {dst_node} with subset: {subset_nodes}\n")

    ## --------------------------------------------------------------------
    ## backward greedy search
    log_message("\n" + "=" * 60 + "\n")
    log_message("STARTING BACKWARD PHASE (EDGE DELETION)\n")
    log_message("=" * 60 + "\n")
    
    count2 = 0
    score_new = score
    update2 = []
    G_step2 = []
    score_record2 = []
    graph_record2 = []

    while True:
        count2 = count2 + 1
        score = score_new
        score_record2.append(score)
        graph_record2.append(G)
        min_chscore = 1e7
        min_desc = []
        
        log_message(f"\nBackward iteration {count2}\n")
        log_message(f"Current score: {score:.4f}\n")
        
        for i in range(N):
            for j in range(N):
                # Check that edge isn't whitelisted before considering deletion
                is_whitelisted = (whitelist_matrix is not None and 
                                (whitelist_matrix[i, j] or whitelist_matrix[j, i]))
                
                if not is_whitelisted and (blacklist_matrix is None or not blacklist_matrix[i, j]) and \
                (((G.graph[j, i] == Endpoint.TAIL.value and \
                    G.graph[i, j] == Endpoint.TAIL.value) or \
                    G.graph[j, i] == Endpoint.ARROW.value) and i!=j):

                    # Always log which edge we're considering
                    log_message(f"  Considering deletion of edge between nodes {node_names[i]} and {node_names[j]}\n")
                        
                    # Find neighbors for validity tests
                    Hj = np.intersect1d(
                        np.where(G.graph[:, j] == Endpoint.TAIL.value)[0],
                        np.where(G.graph[j, :] == Endpoint.TAIL.value)[0],
                    )  # neighbors of Xj
                    Hi = np.union1d(
                        np.where(G.graph[i, :] != Endpoint.NULL.value)[0],
                        np.where(G.graph[:, i] != Endpoint.NULL.value)[0],
                    )  # adjacent to Xi
                    H0 = np.intersect1d(Hj, Hi)  # neighbours of Xj that are adjacent to Xi

                    # Check subsets for validity
                    sub = Combinatorial(H0.tolist())  # Find all subsets for H0
                    S = np.ones(len(sub))  # S indicates whether we need to check subset(k)
                    
                    edge_valid = False  # Flag to track if any valid configuration was found
                    best_subset_score = 1e7
                    best_subset = []

                    for k in range(len(sub)):
                        if S[k] == 1:
                            V = delete_validity_test(G, i, j, sub[k])  # Delete validation test
                            if V:
                                # Find subsets that include sub(k)
                                Idx = find_subset_include(sub[k], sub)
                                S[np.where(Idx == 1)] = 2  # Set their S to 2
                        else:
                            V = 1

                        if V:
                            chscore, desc, record_local_score = delete_changed_score(
                                X, G, i, j, sub[k], record_local_score, score_func, parameters
                            )
                            
                            if _fairness_enabled:
                                # Backward phase uses the same symmetric penalty as forward
                                # to preserve score equivalence during Stage-1 search.
                                # Negate the penalty to reward removing fairness-concerning edges.
                                _psi = _psi_sym.get((i, j), 0.0)
                                _bic_only = chscore
                                chscore = fairness_adjusted_delta(
                                    chscore, -_psi, X.shape[0], fairness_lambda
                                )
                                if _psi > 0:
                                    _penalty_term = fairness_lambda * X.shape[0] * _psi
                                    log_message(f"      [Fairness] BIC_delta={_bic_only:.4f}, psi_sym={_psi:.4f}, reward={_penalty_term:.4f}, adjusted={chscore:.4f}\n")
                                _fairness_edge_logs.append(
                                    {
                                        "phase": "backward",
                                        "edge_i": int(i),
                                        "edge_j": int(j),
                                        "edge_i_name": node_names[i],
                                        "edge_j_name": node_names[j],
                                        "subset": "|".join(node_names[idx] for idx in sub[k]),
                                        "bic_delta": float(_bic_only),
                                        "psi_sym": float(_psi),
                                        "penalty_term": float(fairness_lambda * X.shape[0] * _psi),
                                        "adjusted_delta": float(chscore),
                                    }
                                )
                            
                            edge_valid = True
                            subset_str = [node_names[idx] for idx in sub[k]]
                            log_message(f"    Subset {subset_str} score change: {chscore:.4f}\n")
                            
                            if chscore < best_subset_score:
                                best_subset_score = chscore
                                best_subset = subset_str
                                
                            if chscore < min_chscore:
                                min_chscore = chscore
                                min_desc = desc
                    
                    # After evaluating all subsets, log the best one for this edge
                    if edge_valid:
                        log_message(f"    Best subset for this edge: {best_subset} with score change: {best_subset_score:.4f}\n")
                    else:
                        log_message(f"    No valid configuration found for this edge\n")

        if len(min_desc) != 0:
            score_new = score + min_chscore
            if score - score_new <= 1e-8:
                log_message("  No significant score improvement. Ending backward phase.\n")
                break

            # Apply the operator
            src_node = node_names[min_desc[0]]
            dst_node = node_names[min_desc[1]]
            subset_nodes = [node_names[idx] for idx in min_desc[2]]
            log_message(f"  DELETING edge between {src_node} and {dst_node} with subset: {subset_nodes}\n")
            log_message(f"  Score change: {min_chscore:.4f}, New score: {score_new:.4f}\n")
                
            G = delete(G, min_desc[0], min_desc[1], min_desc[2])
            update2.append([min_desc[0], min_desc[1], min_desc[2]])

            # Store original graph state
            G_orig = deepcopy(G)

            # Perform graph conversions with blacklist handling
            try:
                if blacklist_matrix is not None:
                    G_dag = pdag2dag(G, blacklist_matrix)
                else:
                    G_dag = pdag2dag(G)
                G = dag2cpdag_with_blacklist(G_dag, blacklist_matrix)
            except ValueError as e:
                # If conversion fails, revert to original state
                G = G_orig
                log_message(f"  WARNING: Graph conversion failed: {str(e)}\n")
                log_message("  Reverting to previous valid graph. Ending backward phase.\n")
                break
            
            G_step2.append(G)
        else:
            score_new = score
            log_message("  No valid deletions found. Ending backward phase.\n")
            break

    log_message("\nBackward phase summary:\n")
    log_message(f"Number of iterations: {count2}\n")
    log_message(f"Number of edge deletions: {len(update2)}\n")
    log_message(f"Final score after backward phase: {score_new:.4f}\n")
        
    # Display edge deletions
    if update2:
        log_message("\nEdges deleted:\n")
        for idx, upd in enumerate(update2):
            src_node = node_names[upd[0]]
            dst_node = node_names[upd[1]]
            subset_nodes = [node_names[idx] for idx in upd[2]]
            log_message(f"  {idx+1}. {src_node} -- {dst_node} with subset: {subset_nodes}\n")
    
    # Final results summary
    log_message("\n" + "=" * 60 + "\n")
    log_message("FINAL RESULTS\n")
    log_message("=" * 60 + "\n")
    log_message(f"Initial score: {score_record1[0]:.4f}\n")
    log_message(f"Score after forward phase: {score_record2[0]:.4f}\n")
    log_message(f"Final score: {score_new:.4f}\n")
    
    # Print summary of blacklisted edges if applicable
    if blacklist_matrix is not None:
        blacklist_count = 0
        for i in range(N):
            for j in range(N):
                if blacklist_matrix[i, j] and i != j:
                    blacklist_count += 1
        log_message(f"\nBlacklisted edges prevented: {blacklist_count}\n")
    
    log_message("\nGES algorithm completed.\n")

    Record = {
        "update1": update1,
        "update2": update2,
        "G_step1": G_step1,
        "G_step2": G_step2,
        "G": G,
        "score": score_new,
        "pi_sigma": _pi_sigma,
        "rho_adj": _rho_adj if _fairness_enabled else None,
        "fairness_edge_logs": _fairness_edge_logs,
    }
    return Record
