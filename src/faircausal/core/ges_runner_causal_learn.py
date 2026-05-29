import sys
import os
import datetime
import numpy as np
import csv
from contextlib import redirect_stdout

# ProxyFair algorithm map for ICDM:
# - Phase 1 (score-equivalent CPDAG search with symmetric fairness penalty): run_ges(...)
# - Phase 2 (directed fair DAG extraction inside CPDAG): implemented in
#   src/faircausal/core/counterfactual_fairness_runner.py (select_fairest_dag flow)

# Add the causal_learn directory to path using relative path
current_dir = os.path.dirname(os.path.abspath(__file__))
causal_learn_path = os.path.join(current_dir, "..", "causal_learn")
sys.path.append(causal_learn_path)

# Import both GES versions for conditional use
from src.faircausal.causal_learn.causallearn.search.ScoreBased.GES_fair import ges as ges_fair
from src.faircausal.causal_learn.causallearn.search.ScoreBased.GES import ges as ges_standard
from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint

def validate_blacklist_compliance(graph, blacklist_matrix, node_names, log_file=None):
    """
    Validate that the graph complies with blacklist constraints
    """
    def log_message(msg):
        if log_file:
            print(msg, file=log_file)
            log_file.flush()
        print(msg)
    
    violations = []
    
    if blacklist_matrix is None:
        log_message("No blacklist matrix provided - skipping validation")
        return True, violations
    
    log_message("\n" + "="*50)
    log_message("BLACKLIST COMPLIANCE VALIDATION")
    log_message("="*50)
    
    # Get all edges from the graph
    edges = graph.get_graph_edges()
    
    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        
        node1_idx = None
        node2_idx = None
        
        # Find node indices
        for i, node in enumerate(graph.get_nodes()):
            if node.get_name() == node1.get_name():
                node1_idx = i
            if node.get_name() == node2.get_name():
                node2_idx = i
        
        if node1_idx is None or node2_idx is None:
            continue
            
        name1 = node_names[node1_idx] if node1_idx < len(node_names) else node1.get_name()
        name2 = node_names[node2_idx] if node2_idx < len(node_names) else node2.get_name()
        
        endpoint1_to_2 = graph.graph[graph.node_map[node1], graph.node_map[node2]]
        endpoint2_to_1 = graph.graph[graph.node_map[node2], graph.node_map[node1]]
        
        # Check for directed edges and validate against blacklist
        if (endpoint1_to_2 == Endpoint.TAIL.value and 
            endpoint2_to_1 == Endpoint.ARROW.value):
            # Directed edge from node1 to node2
            if blacklist_matrix[node1_idx, node2_idx]:
                violations.append((name1, name2, "directed"))
                log_message(f"VIOLATION: Blacklisted directed edge {name1} -> {name2}")
                
        elif (endpoint1_to_2 == Endpoint.ARROW.value and 
              endpoint2_to_1 == Endpoint.TAIL.value):
            # Directed edge from node2 to node1
            if blacklist_matrix[node2_idx, node1_idx]:
                violations.append((name2, name1, "directed"))
                log_message(f"VIOLATION: Blacklisted directed edge {name2} -> {name1}")
                
        else:
            # Undirected edge - check both directions
            if (blacklist_matrix[node1_idx, node2_idx] or 
                blacklist_matrix[node2_idx, node1_idx]):
                violations.append((name1, name2, "undirected"))
                log_message(f"VIOLATION: Blacklisted undirected edge {name1} -- {name2}")
    
    if violations:
        log_message(f"\nTOTAL VIOLATIONS: {len(violations)}")
        is_valid = False
    else:
        log_message("[OK] All edges comply with blacklist constraints")
        is_valid = True
    
    log_message("="*50)
    return is_valid, violations

def debug_blacklist_matrix(blacklist_matrix, node_names, file=None):
    """
    Debug function to print all blacklisted edges in a readable format
    """
    def log_message(msg):
        if file:
            print(msg, file=file)
            file.flush()
        print(msg)
    
    if blacklist_matrix is None:
        log_message("No blacklist matrix provided")
        return
    
    log_message("\n" + "="*50)
    log_message("BLACKLIST MATRIX ANALYSIS")
    log_message("="*50)
    
    forbidden_edges = np.where(blacklist_matrix)
    log_message(f"Total forbidden edges: {len(forbidden_edges[0])}")
    
    # Group by source node for better readability
    edge_dict = {}
    for i, j in zip(forbidden_edges[0], forbidden_edges[1]):
        src = node_names[i] if i < len(node_names) else f"Node_{i}"
        dst = node_names[j] if j < len(node_names) else f"Node_{j}"
        
        if src not in edge_dict:
            edge_dict[src] = []
        edge_dict[src].append(dst)
    
    log_message("\nForbidden edges by source node:")
    for src in sorted(edge_dict.keys()):
        destinations = sorted(edge_dict[src])
        log_message(f"  {src} -> {', '.join(destinations)}")
    
    log_message("="*50)

def print_graph_edges(graph, node_names=None, file=None):
    """
    Enhanced version that prints detailed graph information and validates blacklist
    """
    # Handle file parameter
    file_obj = None
    close_file = False
    dowhy_edges = []
    
    try:
        if isinstance(file, str):
            file_obj = open(file, 'a', encoding='utf-8')
            close_file = True
        else:
            file_obj = file
            
        def log_message(msg):
            if file_obj:
                print(msg, file=file_obj)
                file_obj.flush()
            print(msg)
        
        log_message(f"\n{'='*60}")
        log_message("FINAL GES RESULT - DETAILED EDGE ANALYSIS")
        log_message(f"{'='*60}")
        
        edges = graph.get_graph_edges()
        log_message(f"Total number of edges: {len(edges)}")
        
        # Create a mapping of internal names to display names
        node_name_map = {}
        if node_names is not None:
            for i, node in enumerate(graph.get_nodes()):
                if i < len(node_names):
                    node_name_map[node.get_name()] = node_names[i]
        
        directed_edges = []
        undirected_edges = []
        
        for idx, edge in enumerate(edges, start=1):
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            
            name1 = node_name_map.get(node1.get_name(), node1.get_name())
            name2 = node_name_map.get(node2.get_name(), node2.get_name())
            
            endpoint1_to_2 = graph.graph[graph.node_map[node1], graph.node_map[node2]]
            endpoint2_to_1 = graph.graph[graph.node_map[node2], graph.node_map[node1]]
            
            # For directed edges, add to DoWhy format
            if (endpoint1_to_2 == Endpoint.TAIL.value and 
                endpoint2_to_1 == Endpoint.ARROW.value):
                directed_edges.append((name1, name2))
                dowhy_edges.append((name1, name2))
            elif (endpoint1_to_2 == Endpoint.ARROW.value and 
                  endpoint2_to_1 == Endpoint.TAIL.value):
                directed_edges.append((name2, name1))
                dowhy_edges.append((name2, name1))
            else:
                # For undirected edges
                undirected_edges.append((name1, name2))
        
        # Print directed edges
        if directed_edges:
            log_message(f"\nDirected Edges ({len(directed_edges)}):")
            for idx, (src, dst) in enumerate(directed_edges, 1):
                log_message(f"  {idx}. {src} --> {dst}")
        
        # Print undirected edges
        if undirected_edges:
            log_message(f"\nUndirected Edges ({len(undirected_edges)}):")
            for idx, (n1, n2) in enumerate(undirected_edges, 1):
                log_message(f"  {idx}. {n1} -- {n2}")
        
        # Print DoWhy format for directed edges only
        if directed_edges:
            log_message(f"\nDoWhy DAG format (directed edges only):")
            log_message("dag = [")
            for src, dst in directed_edges:
                log_message(f'    ("{src}", "{dst}"),')
            log_message("]")
        
        return dowhy_edges, undirected_edges
    
    finally:
        if close_file and file_obj:
            file_obj.close()

#------------------
# Causal graph with GES
#------------------
def run_ges(data, 
            score_func='local_score_BIC', 
            maxP=None, 
            parameters=None, 
            node_names=None,
            blacklist_matrix=None,
            whitelist_matrix=None,
            debug=False,
            log_dir=None,
            log_file=None,
            run_dir=None,
            experiment_type='baseline',
            fairness_lambda=None,
            fairness_tau_c=5.0,
            protected_attr_indices=None,
            outcome_index=None,
            path_effect_method='distance'):
    """
    Run GES (Greedy Equivalence Search) algorithm on the provided data
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        The input data for causal discovery
    score_func : str, optional
        Scoring function to use ('local_score_BIC' or 'local_score_BDeu')
    maxP : int, optional
        Maximum number of parents
    parameters : dict, optional
        Additional parameters for the algorithm
    node_names : list of str, optional
        Column names or variable names for the nodes in the graph
    blacklist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a forbidden edge
    whitelist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a must have edge
    debug : bool, optional
        Whether to print debug information during execution (default: True)
    log_dir : str, optional
        Directory to save log files
        log_file : str, optional
        Path to existing log file. If provided, will append to this file instead
        of creating a new one.
    run_dir : str, optional
        Main run directory where all outputs for this run are stored
    experiment_type : str, optional
        Type of experiment ('baseline', 'temporal_constraints', 'domain_knowledge', 'soft_fairness').
        For 'baseline', uses standard GES without constraints.
        For other types, uses GES_fair with constraint support.
        
    Returns:
    --------
    dict
        Contains the results including:
        - G: The resulting causal graph
        - steps: The steps taken by the algorithm
        - score: The final score
        - blacklist_matrix: The forbidden edges matrix used in the search
        - log_file: Path to the log file used
    """
    # Set up logging file
    log_file_handle = None
    if log_file:
        # Use provided log file in append mode
        log_file_handle = open(log_file, 'a', encoding='utf-8')
    elif run_dir:
        # Use the run directory if provided
        log_file = os.path.join(run_dir, f"ges_detailed_log.txt")
        log_file_handle = open(log_file, 'w', encoding='utf-8')
    elif log_dir:
        # Fallback to log_dir if run_dir not provided
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"ges_log_{timestamp}.txt")
        log_file_handle = open(log_file, 'w', encoding='utf-8')
    
    try:
        print("\nInitializing GES algorithm...")
        
        # Convert DataFrame to numpy array if necessary
        if hasattr(data, 'values'):
            data = data.values
            print("Converted DataFrame to numpy array")

        # Log constraint information for debugging
        if experiment_type == 'baseline':
            print("="*60)
            print("BASELINE EXPERIMENT - USING STANDARD GES")
            print("="*60)
            print("• No constraints will be applied")
            print("• Using standard GES implementation")
            print("• Blacklist and whitelist matrices will be ignored")
            
            if blacklist_matrix is not None or whitelist_matrix is not None:
                print("[WARN]  WARNING: Constraints provided but will be ignored for baseline experiment")
                
        else:
            print("="*60)
            print(f"{experiment_type.upper()} EXPERIMENT - USING GES_FAIR WITH CONSTRAINTS")
            print("="*60)
            
            # Log blacklist matrix information if provided
            if blacklist_matrix is not None:
                print("[OK] Blacklist matrix provided:")
                n_forbidden = np.sum(blacklist_matrix)
                print(f"  Number of forbidden edges: {n_forbidden}")
                
                if debug and node_names:
                    # Log specific forbidden edges for debugging
                    forbidden_edges = np.where(blacklist_matrix)
                    print("  Forbidden edges:")
                    for i, j in zip(forbidden_edges[0], forbidden_edges[1]):
                        print(f"    {node_names[i]} -> {node_names[j]}")

            # Log whitelist matrix information if provided
            if whitelist_matrix is not None:
                print("[OK] Whitelist matrix provided:")
                n_required = np.sum(whitelist_matrix)
                print(f"  Number of required edges: {n_required}")
                
                if debug and node_names:
                    # Log specific required edges for debugging
                    required_edges = np.where(whitelist_matrix)
                    print("  Required edges:")
                    for i, j in zip(required_edges[0], required_edges[1]):
                        print(f"    {node_names[i]} -> {node_names[j]}")
                
        print("\nRunning GES algorithm...")
        
        # Choose appropriate GES version based on experiment type
        if experiment_type == 'baseline':
            print("Using standard GES (no constraints) for baseline experiment")
            # Use standard GES without constraints
            result = ges_standard(
                X=data,
                score_func=score_func,
                maxP=maxP,
                parameters=parameters,
                node_names=node_names
            )
        else:
            print(f"Using GES_fair with constraints for {experiment_type} experiment")
            # Use GES_fair with constraint support
            result = ges_fair(
                X=data,
                score_func=score_func,
                maxP=maxP,
                parameters=parameters,
                node_names=node_names,
                blacklist_matrix=blacklist_matrix,
                whitelist_matrix=whitelist_matrix,
                debug=debug,
                log_file_handle=log_file_handle,
                fairness_lambda=fairness_lambda,
                fairness_tau_c=fairness_tau_c,
                protected_attr_indices=protected_attr_indices,
                outcome_index=outcome_index,
                path_effect_method=path_effect_method,
            )
        
        print("\nGES algorithm completed")
        
        # ENHANCED: Print detailed final graph information
        final_graph = result['G']
        print(f"\nFinal graph has {final_graph.get_num_edges()} edges")
        
        # Print detailed edge information
        dowhy_edges, undirected_edges = print_graph_edges(
            final_graph, 
            node_names, 
            file=log_file_handle
        )
        
        # CRITICAL: Validate blacklist compliance (only for non-baseline experiments)
        if blacklist_matrix is not None and experiment_type != 'baseline':
            is_valid, violations = validate_blacklist_compliance(
                final_graph, 
                blacklist_matrix, 
                node_names, 
                log_file_handle
            )
            
            if not is_valid:
                print(f"\n[ERROR] CRITICAL ERROR: {len(violations)} blacklist violations detected!")
                print("This indicates a problem with the GES implementation or graph conversion.")
                
                # Log all violations for debugging
                print("\nViolation details:")
                for src, dst, edge_type in violations:
                    print(f"  • {src} -> {dst} ({edge_type})")
            else:
                print("\n[OK] Graph successfully complies with all blacklist constraints")
        elif experiment_type == 'baseline':
            print("\n[OK] Baseline experiment: no constraint validation needed")
            is_valid = True
            violations = []
        else:
            print("\n[OK] No blacklist constraints to validate")
            is_valid = True
            violations = []

        # Persist Phase-1 fairness diagnostics when available.
        if run_dir and experiment_type in ('soft_fairness', 'soft_fairness_only'):
            pi_sigma = result.get('pi_sigma', None)
            if pi_sigma is not None:
                pi_sigma_path = os.path.join(run_dir, "pi_sigma.csv")
                np.savetxt(pi_sigma_path, pi_sigma, delimiter=",", fmt="%.8f")

                # Best-effort heatmap export.
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(pi_sigma, cmap='viridis', aspect='auto')
                    ax.set_title('pi_sigma (edge-level proxy fractions)')
                    if node_names is not None:
                        ticks = np.arange(len(node_names))
                        ax.set_xticks(ticks)
                        ax.set_yticks(ticks)
                        ax.set_xticklabels(node_names, rotation=90, fontsize=7)
                        ax.set_yticklabels(node_names, fontsize=7)
                    fig.colorbar(im, ax=ax)
                    fig.tight_layout()
                    fig.savefig(os.path.join(run_dir, "pi_sigma_heatmap.png"), dpi=150)
                    plt.close(fig)
                except Exception as heatmap_exc:
                    print(f"[WARN] Could not save pi_sigma_heatmap.png: {heatmap_exc}")

            fairness_edge_logs = result.get('fairness_edge_logs', []) or []
            if fairness_edge_logs:
                edge_log_path = os.path.join(run_dir, "fairness_edge_penalties.csv")
                fieldnames = [
                    "phase",
                    "edge_i",
                    "edge_j",
                    "edge_i_name",
                    "edge_j_name",
                    "subset",
                    "bic_delta",
                    "psi_sym",
                    "penalty_term",
                    "adjusted_delta",
                ]
                with open(edge_log_path, 'w', encoding='utf-8', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in fairness_edge_logs:
                        writer.writerow({k: row.get(k, "") for k in fieldnames})
        
        # Add comprehensive results to return dictionary
        result['dowhy_edges'] = dowhy_edges
        result['undirected_edges'] = undirected_edges
        result['log_file'] = log_file
        result['blacklist_matrix'] = blacklist_matrix
        result['blacklist_compliant'] = is_valid
        result['blacklist_violations'] = violations
        result['experiment_type'] = experiment_type

        return result
        
    finally:
        if log_file_handle:
            log_file_handle.close()
            
