# faircausal/core/ges_runner_pgmpy.py
import sys
import os
import datetime
import numpy as np
import pandas as pd
from contextlib import redirect_stdout

from pgmpy.estimators import GES, ExpertKnowledge
from pgmpy.estimators.StructureScore import get_scoring_method

from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint

class CausalLearnCompatibleGraph:
    """
    Wrapper class to make pgmpy DAG compatible with causal-learn style interface
    """
    def __init__(self, pgmpy_dag, node_names=None):
        self.pgmpy_dag = pgmpy_dag
        self.node_names = node_names if node_names is not None else list(pgmpy_dag.nodes())
        
        # Create a mapping from node names to indices for compatibility
        self.node_map = {}
        self._nodes = []
        
        for idx, name in enumerate(self.node_names):
            node_obj = CausalLearnNode(name)
            self._nodes.append(node_obj)
            self.node_map[node_obj] = idx
        
        # Create a simple graph representation for compatibility (adjacency matrix)
        n_nodes = len(self.node_names)
        self.graph = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # Fill the adjacency matrix based on pgmpy DAG edges
        # For a DAG, all edges are directed (Tail -> Arrow)
        for source, target in self.pgmpy_dag.edges():
            if source in self.node_names and target in self.node_names:
                source_idx = self.node_names.index(source)
                target_idx = self.node_names.index(target)
                # Set endpoints: source has TAIL, target has ARROW (directed edge)
                self.graph[source_idx, target_idx] = Endpoint.ARROW.value
                self.graph[target_idx, source_idx] = Endpoint.TAIL.value
        
    def get_graph_edges(self):
        """Return edge objects compatible with causal-learn"""
        edges = []
        
        for source, target in self.pgmpy_dag.edges():
            if source in self.node_names and target in self.node_names:
                source_idx = self.node_names.index(source)
                target_idx = self.node_names.index(target)
                
                source_node = self._nodes[source_idx]
                target_node = self._nodes[target_idx]
                
                edge = CausalLearnEdge(source_node, target_node)
                edges.append(edge)
        
        return edges
    
    def get_nodes(self):
        """Return node objects compatible with causal-learn"""
        return self._nodes
    
    def edges(self):
        """Return edges (pgmpy style)"""
        return self.pgmpy_dag.edges()
    
    def nodes(self):
        """Return nodes (pgmpy style)"""
        return self.pgmpy_dag.nodes()
    
    def number_of_nodes(self):
        """Return number of nodes"""
        return self.pgmpy_dag.number_of_nodes()
    
    def number_of_edges(self):
        """Return number of edges"""
        return self.pgmpy_dag.number_of_edges()
    
    def get_parents(self, node):
        """Get parents of a node"""
        return self.pgmpy_dag.get_parents(node)
    
    def get_children(self, node):
        """Get children of a node"""
        return self.pgmpy_dag.get_children(node)
    
    def has_edge(self, u, v):
        """Check if edge exists"""
        return self.pgmpy_dag.has_edge(u, v)
    
    def add_edge(self, u, v):
        """Add edge"""
        return self.pgmpy_dag.add_edge(u, v)
    
    def remove_edge(self, u, v):
        """Remove edge"""
        return self.pgmpy_dag.remove_edge(u, v)
    
    def copy(self):
        """Return a copy of the graph"""
        return CausalLearnCompatibleGraph(self.pgmpy_dag.copy(), self.node_names)


class CausalLearnNode:
    """
    Simple node class compatible with causal-learn interface
    """
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def __str__(self):
        return str(self.name)
    
    def __repr__(self):
        return f"CausalLearnNode({self.name})"


class CausalLearnEdge:
    """
    Simple edge class compatible with causal-learn interface
    """
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
    
    def get_node1(self):
        return self.node1
    
    def get_node2(self):
        return self.node2
    
    def __str__(self):
        return f"{self.node1} -> {self.node2}"
    
    def __repr__(self):
        return f"CausalLearnEdge({self.node1}, {self.node2})"

def print_graph_edges(graph, node_names=None, file=None):
    """
    Print the edges of a pgmpy graph and return them in DoWhy format.
    
    Parameters:
    -----------
    graph : pgmpy.base.DAG or pgmpy.models.BayesianNetwork
        The graph object to print edges from
    node_names : list of str, optional
        Names of the nodes. If provided and different from graph nodes,
        creates a mapping. If None, uses graph node names.
    file : file object or str, optional
        File object or path to write output to (if None, prints to stdout)
        
    Returns:
    --------
    list
        List of tuples representing edges in DoWhy format [(source, target), ...]
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
            
        edges = list(graph.edges())
        print("\nGraph Edges:", file=file_obj)
        
        # Create a mapping of graph names to display names if needed
        graph_nodes = list(graph.nodes())
        node_name_map = {}
        if node_names is not None and len(node_names) == len(graph_nodes):
            for graph_node, display_name in zip(sorted(graph_nodes), node_names):
                node_name_map[graph_node] = display_name
        
        for idx, (source, target) in enumerate(edges, start=1):
            # Use mapped names if available, otherwise use original names
            display_source = node_name_map.get(source, source)
            display_target = node_name_map.get(target, target)
            
            print(f"{idx}. {display_source} --> {display_target}", file=file_obj)
            dowhy_edges.append((display_source, display_target))
        
        # Print DoWhy format
        print("\nDoWhy DAG format:", file=file_obj)
        print("dag = [", file=file_obj)
        for edge in dowhy_edges:
            print(f'    ("{edge[0]}", "{edge[1]}"),', file=file_obj)
        print("]", file=file_obj)
        
        return dowhy_edges
    
    finally:
        if close_file and file_obj:
            file_obj.close()

def create_expert_knowledge(node_names, blacklist_matrix=None, whitelist_matrix=None):
    """
    Convert blacklist/whitelist matrices to pgmpy ExpertKnowledge format.
    
    Parameters:
    -----------
    node_names : list of str
        Names of the nodes/variables
    blacklist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a forbidden edge (i -> j)
    whitelist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a required edge (i -> j)
        
    Returns:
    --------
    ExpertKnowledge
        pgmpy ExpertKnowledge object with constraints
    """
    expert_knowledge = ExpertKnowledge()
    
    # Initialize the attributes as empty sets if they don't exist
    if not hasattr(expert_knowledge, 'forbidden_edges'):
        expert_knowledge.forbidden_edges = set()
    if not hasattr(expert_knowledge, 'required_edges'):
        expert_knowledge.required_edges = set()
    
    if blacklist_matrix is not None:
        forbidden_edges = np.where(blacklist_matrix)
        for i, j in zip(forbidden_edges[0], forbidden_edges[1]):
            expert_knowledge.forbidden_edges.add((node_names[i], node_names[j]))
    
    if whitelist_matrix is not None:
        required_edges = np.where(whitelist_matrix)
        for i, j in zip(required_edges[0], required_edges[1]):
            expert_knowledge.required_edges.add((node_names[i], node_names[j]))
    
    return expert_knowledge

#------------------
# Causal graph with GES (pgmpy version)
#------------------
def run_ges(data, 
            score_func='BIC', 
            maxP=None, 
            parameters=None, 
            node_names=None,
            blacklist_matrix=None,
            whitelist_matrix=None,
            debug=False,
            log_dir=None,
            log_file=None,
            run_dir=None):
    """
    Run GES (Greedy Equivalence Search) algorithm using pgmpy on the provided data
    
    Parameters:
    -----------
    data : pandas.DataFrame or numpy.ndarray
        The input data for causal discovery
    score_func : str, optional
        Scoring function to use ('BIC'/'local_score_BIC' or 'BDeu'/'local_score_BDeu')
        Maps to pgmpy scoring methods: 'bic-d' and 'bdeu'
    maxP : int, optional
        Maximum number of parents (WARNING: Not supported in pgmpy GES, will be ignored)
    parameters : dict, optional
        Additional parameters for the algorithm (currently unused in pgmpy GES)
    node_names : list of str, optional
        Column names or variable names for the nodes in the graph
    blacklist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a forbidden edge
    whitelist_matrix : numpy.ndarray, optional
        Boolean matrix where True indicates a must have edge
    debug : bool, optional
        Whether to print debug information during execution (default: False)
    log_dir : str, optional
        Directory to save log files
    log_file : str, optional
        Path to existing log file. If provided, will append to this file instead
        of creating a new one.
    run_dir : str, optional
        Main run directory where all outputs for this run are stored
        
    Returns:
    --------
    dict
        Contains the results including:
        - G: The resulting causal graph (pgmpy PDAG)
        - score: The final score
        - expert_knowledge: The ExpertKnowledge object used
        - dowhy_edges: Edges in DoWhy format
        - log_file: Path to the log file used
        - blacklist_matrix: The blacklist matrix (for compatibility)
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
    
    def log_message(msg):
        """Helper function to handle both console and file logging"""
        if debug:
            print(msg)
        if log_file_handle:
            print(msg, file=log_file_handle)
            log_file_handle.flush()

    try:
        log_message("\nInitializing pgmpy GES algorithm...")
        
        # Handle data conversion to maintain compatibility with causal-learn style calls
        if isinstance(data, np.ndarray):
            # This is the expected format from causal-learn style calls (df.values)
            if node_names is None:
                node_names = [f"X{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=node_names)
            log_message("Converted numpy array to pandas DataFrame")
        elif hasattr(data, 'values'):
            # It's a DataFrame, extract node names and convert to ensure compatibility
            if node_names is None:
                node_names = list(data.columns)
            # Convert to numpy then back to DataFrame to ensure consistent handling
            data_array = data.values
            data = pd.DataFrame(data_array, columns=node_names)
            log_message("Processed DataFrame for compatibility")
        else:
            # Handle other data types
            if node_names is None:
                node_names = [f"X{i}" for i in range(data.shape[1] if hasattr(data, 'shape') else len(data[0]))]
            data = pd.DataFrame(data, columns=node_names)
            log_message("Converted data to pandas DataFrame")
        
        log_message(f"Data shape: {data.shape}")
        log_message(f"Node names: {node_names}")

        # Create expert knowledge from matrices
        expert_knowledge = None
        if blacklist_matrix is not None or whitelist_matrix is not None:
            expert_knowledge = create_expert_knowledge(
                node_names, blacklist_matrix, whitelist_matrix
            )
            
            # Log expert knowledge information
            if blacklist_matrix is not None:
                n_forbidden = np.sum(blacklist_matrix)
                log_message(f"\nNumber of forbidden edges: {n_forbidden}")
                
                if debug and n_forbidden > 0:
                    forbidden_edges = np.where(blacklist_matrix)
                    log_message("Forbidden edges:")
                    for i, j in zip(forbidden_edges[0], forbidden_edges[1]):
                        log_message(f"  {node_names[i]} -> {node_names[j]}")

            if whitelist_matrix is not None:
                n_required = np.sum(whitelist_matrix)
                log_message(f"Number of required edges: {n_required}")
                
                if debug and n_required > 0:
                    required_edges = np.where(whitelist_matrix)
                    log_message("Required edges:")
                    for i, j in zip(required_edges[0], required_edges[1]):
                        log_message(f"  {node_names[i]} -> {node_names[j]}")
        
        # Set up scoring function - handle both causal-learn and pgmpy naming conventions
        score_func_upper = score_func.upper()
        if score_func_upper in ['BIC', 'LOCAL_SCORE_BIC']:
            scoring_method = 'bic-d'  # pgmpy uses 'bic-d' for discrete BIC
        elif score_func_upper in ['BDEU', 'LOCAL_SCORE_BDEU']:
            scoring_method = 'bdeu'   # pgmpy uses 'bdeu'
        else:
            log_message(f"Warning: Unknown score function {score_func}, using BIC")
            scoring_method = 'bic-d'
        
        log_message(f"Using scoring method: {scoring_method}")
        
        # Initialize GES with DataFrame
        ges_estimator = GES(data)
        
        log_message("\nRunning GES algorithm...")
        
        # Prepare arguments for estimate method
        estimate_args = {
            'scoring_method': scoring_method,
            'debug': debug
        }
        
        if expert_knowledge is not None:
            estimate_args['expert_knowledge'] = expert_knowledge
        
        # Note: pgmpy GES doesn't have max_indegree parameter like HillClimbSearch
        # The maxP parameter is not directly supported in pgmpy GES
        if maxP is not None:
            log_message(f"Warning: maxP (max_indegree) parameter is not supported in pgmpy GES")
            log_message(f"Requested maxP: {maxP} - this constraint will be ignored")
        
        # Run GES
        best_model = ges_estimator.estimate(**estimate_args)
        
        log_message("\nGES algorithm completed")
        log_message(f"Number of nodes: {best_model.number_of_nodes()}")
        log_message(f"Number of edges: {best_model.number_of_edges()}")
        
        # Calculate final score using the same method as GES internally
        try:
            _, score_cache = get_scoring_method(scoring_method, data, use_cache=True)
            score_fn = score_cache.local_score
            
            # Calculate total score by summing local scores for each node
            final_score = 0
            for node in best_model.nodes():
                parents = best_model.get_parents(node)
                final_score += score_fn(node, parents)
            
            log_message(f"Final score: {final_score}")
            
        except Exception as score_error:
            log_message(f"Warning: Could not calculate final score: {score_error}")
            final_score = None
        
        # Get edges in DoWhy format
        dowhy_edges = print_graph_edges(best_model, node_names, file=log_file_handle)
        
        # Wrap the pgmpy DAG in our compatibility wrapper
        compatible_graph = CausalLearnCompatibleGraph(best_model, node_names)
        
        # Prepare result dictionary to match causal-learn format
        result = {
            'G': compatible_graph,  # Use the wrapped graph
            'score': final_score,
            'dowhy_edges': dowhy_edges,
            'undirected_edges': [],  # pgmpy always returns a full DAG
            'log_file': log_file,
            'blacklist_matrix': blacklist_matrix,  # Include for compatibility
            'expert_knowledge': expert_knowledge,
            'scoring_method': scoring_method,
            'pgmpy_dag': best_model  # Keep the original pgmpy DAG for reference
        }
        
        log_message(f"\nResults saved. Log file: {log_file}")
        
        return result
        
    except Exception as e:
        error_msg = f"Error during GES execution: {str(e)}"
        if log_file_handle:
            print(error_msg, file=log_file_handle)
            log_file_handle.flush()
        if debug:
            print(error_msg)
        raise
        
    finally:
        if log_file_handle:
            log_file_handle.close()

def simple_dag_converter(graph_wrapper, node_names=None):
    """
    Convert a simple DAG (from pgmpy) to the format expected by analyze_all_dags_fairness.
    Since pgmpy GES returns a DAG (not CPDAG), we just need to convert it to the right format.
    
    Parameters:
    -----------
    graph_wrapper : CausalLearnCompatibleGraph
        The wrapped pgmpy DAG
    node_names : list, optional
        List of node names
        
    Returns:
    --------
    list
        A list containing a single DAG represented as a list of (source, target) tuples
    """
    if node_names is None:
        node_names = graph_wrapper.node_names
    
    # Since this is already a DAG, just convert edges to the expected format
    dag_edges = []
    for source, target in graph_wrapper.get_graph_edges():
        dag_edges.append((source, target))
    
    # Return as a list containing one DAG (since there are no undirected edges to resolve)
    return [dag_edges]

# Additional utility functions for compatibility

def get_adjacency_matrix(graph, node_names=None):
    """
    Get adjacency matrix from pgmpy graph or compatible wrapper
    
    Parameters:
    -----------
    graph : pgmpy DAG or CausalLearnCompatibleGraph
        The graph to convert
    node_names : list, optional
        Ordered list of node names for matrix rows/columns
        
    Returns:
    --------
    numpy.ndarray
        Adjacency matrix where entry (i,j) is 1 if there's an edge i->j
    """
    # Handle both wrapped and unwrapped graphs
    if hasattr(graph, 'pgmpy_dag'):
        actual_graph = graph.pgmpy_dag
        if node_names is None:
            node_names = graph.node_names
    else:
        actual_graph = graph
        if node_names is None:
            node_names = sorted(list(actual_graph.nodes()))
    
    n_nodes = len(node_names)
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    
    node_to_idx = {node: idx for idx, node in enumerate(node_names)}
    
    for source, target in actual_graph.edges():
        i = node_to_idx[source]
        j = node_to_idx[target]
        adj_matrix[i, j] = 1
    
    return adj_matrix

def print_model_info(result, detailed=True):
    """
    Print detailed information about the GES result
    
    Parameters:
    -----------
    result : dict
        Result dictionary from run_ges function
    detailed : bool
        Whether to print detailed information
    """
    graph = result['G']
    
    print(f"\n=== GES Results ===")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Final score: {result['score']:.4f}" if result['score'] is not None else "Final score: Not available")
    
    if detailed:
        print(f"\nNodes: {list(graph.nodes())}")
        print(f"Edges: {list(graph.edges())}")
        
        if result.get('expert_knowledge'):
            ek = result['expert_knowledge']
            print(f"\nExpert Knowledge Constraints:")
            if hasattr(ek, 'forbidden_edges') and ek.forbidden_edges:
                print(f"  Forbidden edges: {len(ek.forbidden_edges)}")
            if hasattr(ek, 'required_edges') and ek.required_edges:
                print(f"  Required edges: {len(ek.required_edges)}")
