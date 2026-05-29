from copy import deepcopy
import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph

def pdag2dag(G: GeneralGraph, blacklist_matrix: np.ndarray = None) -> GeneralGraph:
    """
    Convert a PDAG to its corresponding DAG while strictly enforcing blacklisted edges

    Parameters
    ----------
    G : GeneralGraph
        Partially Direct Acyclic Graph
    blacklist_matrix : numpy.ndarray
        Boolean matrix where blacklist_matrix[i,j] = True indicates 
        edge i->j is forbidden.

    Returns
    -------
    Gd : GeneralGraph
        Direct Acyclic Graph
    """
    nodes = G.get_nodes()
    n_vars = G.num_vars

    # Initialize blacklist matrix if not provided
    if blacklist_matrix is None:
        blacklist_matrix = np.zeros((n_vars, n_vars), dtype=bool)

    # Initialize output graph with directed edges only
    Gd = deepcopy(G)
    edges = Gd.get_graph_edges()

    # Process edges in batch using numpy operations
    edge_matrix = np.zeros((n_vars, n_vars), dtype=bool)
    for edge in edges:
        node1_idx = nodes.index(edge.node1)
        node2_idx = nodes.index(edge.node2)
        
        # Check if edge is directed
        is_directed = ((edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL) or
                      (edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW))
        
        if is_directed:
            # For directed edges, check blacklist in correct direction
            if edge.endpoint1 == Endpoint.ARROW:  # node2 -> node1
                is_blacklisted = blacklist_matrix[node2_idx, node1_idx]
            else:  # node1 -> node2
                is_blacklisted = blacklist_matrix[node1_idx, node2_idx]
        else:
            is_blacklisted = False  # Undirected edges handled later
            
        if not is_directed or is_blacklisted:
            Gd.remove_edge(edge)
        else:
            edge_matrix[node1_idx, node2_idx] = True
    
    # Create working copy
    Gp = deepcopy(G)
    graph_matrix = Gp.graph.astype(np.int8)  # Convert to more memory-efficient dtype
    processed = np.zeros(n_vars, dtype=bool)  # Use boolean array instead of int

    while not processed.all():
        any_processed = False
        unprocessed_mask = ~processed
        
        # Find nodes with no outgoing edges to unprocessed nodes
        outgoing_edges = (graph_matrix == 1) & unprocessed_mask[None, :]
        nodes_no_outgoing = ~outgoing_edges.any(axis=1) & unprocessed_mask
        
        for i in np.where(nodes_no_outgoing)[0]:
            # Find undirected neighbors using vectorized operations
            undirected_mask = (
                (graph_matrix[i, :] == -1) & 
                (graph_matrix[:, i] == -1) & 
                unprocessed_mask & 
                ~blacklist_matrix[:, i]  # Filter blacklisted edges in one step
            )
            undirected_neighbors = np.where(undirected_mask)[0]
            
            # Find adjacent nodes
            adjacent_mask = ((graph_matrix[i, :] == 1) | (graph_matrix[:, i] == 1)) & unprocessed_mask
            adjacent_nodes = np.where(adjacent_mask)[0]
            
            all_connected = len(undirected_neighbors) == 0 or check2(graph_matrix, undirected_neighbors, adjacent_nodes)
            
            if all_connected:
                # Add directed edges (neighbor->i)
                for j in undirected_neighbors:
                    Gd.add_edge(Edge(nodes[j], nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                
                processed[i] = True
                any_processed = True
        
        # Break deadlock if needed
        if not any_processed and not processed.all():
            i = np.where(~processed)[0][0]
            processed[i] = True
    
    return Gd

def check2(graph_matrix: np.ndarray, Nx: np.ndarray, Ax: np.ndarray) -> bool:
    """
    Helper function to check if all nodes in Nx are connected to all nodes in Ax
    except themselves
    """
    if len(Nx) == 0 or len(Ax) == 0:
        return True
    
    # Create connectivity matrix for all nodes at once
    connectivity = graph_matrix[Nx[:, None], Ax[None, :]]
    # Remove self-connections from consideration
    mask = Nx[:, None] != Ax[None, :]
    # Check if all required connections exist
    return np.all(connectivity[mask] != 0)