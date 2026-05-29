from copy import deepcopy
from typing import Optional
import numpy as np
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph

def pdag2dag(G: GeneralGraph, blacklist_matrix: Optional[np.ndarray] = None) -> GeneralGraph:
    """
    Convert a PDAG to its corresponding DAG with optional blacklist enforcement
    
    Parameters
    ----------
    G : GeneralGraph
        Partially Direct Acyclic Graph
    blacklist_matrix : numpy.ndarray, optional
        Boolean matrix where blacklist_matrix[i,j] = True indicates 
        edge i->j is forbidden.
    
    Returns
    -------
    Gd : GeneralGraph
        Direct Acyclic Graph
    """
    # # Add a flag to track if we're in a conversion cycle
    # global conversion_in_progress
    # if 'conversion_in_progress' not in globals():
    #     conversion_in_progress = False
    
    # if conversion_in_progress:
    #     return G
        
    # conversion_in_progress = True

    # try:
    #     nodes = G.get_nodes()
    #     # First create a DAG that contains all the directed edges in the PDAG
    #     Gd = deepcopy(G)
    #     edges = Gd.get_graph_edges()

    #     # Keep only directed edges that don't violate blacklist
    #     for edge in edges:
    #         if edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL:
    #             # This is a j->i edge
    #             i = nodes.index(edge.node1)
    #             j = nodes.index(edge.node2)
    #             if blacklist_matrix is not None and blacklist_matrix[j, i]:
    #                 Gd.remove_edge(edge)
    #         elif edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
    #             # This is a i->j edge
    #             i = nodes.index(edge.node1)
    #             j = nodes.index(edge.node2)
    #             if blacklist_matrix is not None and blacklist_matrix[i, j]:
    #                 Gd.remove_edge(edge)
    #         else:
    #             Gd.remove_edge(edge)

    # Create a local copy of the graph to work with
    Gd = deepcopy(G)
    nodes = Gd.get_nodes()
    
    # First create a DAG that contains all the directed edges in the PDAG
    edges = Gd.get_graph_edges()

    # Keep only directed edges that don't violate blacklist
    for edge in edges:
        if edge.endpoint1 == Endpoint.ARROW and edge.endpoint2 == Endpoint.TAIL:
            # This is a j->i edge
            i = nodes.index(edge.node1)
            j = nodes.index(edge.node2)
            if blacklist_matrix is not None and blacklist_matrix[j, i]:
                Gd.remove_edge(edge)
        elif edge.endpoint1 == Endpoint.TAIL and edge.endpoint2 == Endpoint.ARROW:
            # This is a i->j edge
            i = nodes.index(edge.node1)
            j = nodes.index(edge.node2)
            if blacklist_matrix is not None and blacklist_matrix[i, j]:
                Gd.remove_edge(edge)
        else:
            Gd.remove_edge(edge)

    Gp = deepcopy(G)
    inde = np.zeros(Gp.num_vars, dtype=np.dtype(int))  # index whether the ith node has been removed
    
    # Maximum iterations to prevent infinite loops
    max_iterations = Gp.num_vars * 2
    iteration_count = 0
    
    while 0 in inde and iteration_count < max_iterations:
        iteration_count += 1
        progress_made = False
        
        for i in range(Gp.num_vars):
            if inde[i] == 0:
                sign = 0
                if len(np.intersect1d(np.where(Gp.graph[:, i] == 1)[0],
                                    np.where(inde == 0)[0])) == 0:  # Xi has no out-going edges
                    sign = sign + 1
                    # Find neighbors and adjacent nodes
                    Nx = np.intersect1d(
                        np.intersect1d(np.where(Gp.graph[:, i] == -1)[0], 
                                      np.where(Gp.graph[i, :] == -1)[0]),
                        np.where(inde == 0)[0])
                    Ax = np.intersect1d(
                        np.union1d(np.where(Gp.graph[i, :] == 1)[0], 
                                  np.where(Gp.graph[:, i] == 1)[0]),
                        np.where(inde == 0)[0])
                    Ax = np.union1d(Ax, Nx)
                    
                    if len(Nx) > 0:
                        if check2(Gp, Nx, Ax):
                            sign = sign + 1
                    else:
                        sign = sign + 1
                        
                if sign == 2:
                    # For each undirected edge Y-X in PDAG
                    undirected_neighbors = np.intersect1d(
                        np.where(Gp.graph[:, i] == -1)[0], 
                        np.where(Gp.graph[i, :] == -1)[0]
                    )
                    
                    all_orientations_valid = True
                    for index in undirected_neighbors:
                        # Check if any orientation violates blacklist
                        if blacklist_matrix is not None:
                            if blacklist_matrix[index, i] and blacklist_matrix[i, index]:
                                all_orientations_valid = False
                                break
                    
                    if all_orientations_valid:
                        for index in undirected_neighbors:
                            if blacklist_matrix is None:
                                # If no blacklist, add Y->X
                                Gd.add_edge(Edge(nodes[index], nodes[i], 
                                          Endpoint.TAIL, Endpoint.ARROW))
                            else:
                                # Check blacklist constraints
                                if not blacklist_matrix[index, i]:
                                    # If Y->X is not blacklisted, add it
                                    Gd.add_edge(Edge(nodes[index], nodes[i], 
                                              Endpoint.TAIL, Endpoint.ARROW))
                                elif not blacklist_matrix[i, index]:
                                    # If Y->X is blacklisted but X->Y is allowed, add X->Y
                                    Gd.add_edge(Edge(nodes[i], nodes[index], 
                                              Endpoint.TAIL, Endpoint.ARROW))
                        inde[i] = 1
                        progress_made = True
        
        # If no progress was made in this iteration, we're stuck
        if not progress_made and 0 in inde:
            raise ValueError("Cannot orient all edges - graph structure may not permit a valid DAG orientation.")
    
    # Check if we hit the iteration limit
    if iteration_count >= max_iterations:
        raise ValueError("Maximum iterations reached. Graph may contain complex structure that cannot be oriented.")
    
    # Verify the resulting graph is still a DAG
    if not is_dag(Gd):
        raise ValueError("Resulting graph contains cycles. No valid DAG orientation exists.")

    return Gd

def check2(G: GeneralGraph, Nx, Ax):
    s = 1
    for i in range(len(Nx)):
        j = np.delete(Ax, np.where(Ax == Nx[i])[0])
        if len(np.where(G.graph[Nx[i], j] == 0)[0]) != 0:
            s = 0
            break
    return s

def is_dag(G: GeneralGraph) -> bool:
    """
    Check if graph is a DAG by detecting cycles.
    
    Args:
        G: Graph to check
        
    Returns:
        bool: True if graph is a DAG
    """
    N = G.get_num_nodes()
    visited = np.zeros(N) 
    path = np.zeros(N)
    
    def has_cycle(v: int) -> bool:
        visited[v] = 1
        path[v] = 1
        
        for w in range(N):
            if G.graph[w,v] == 1:  # Edge v->w
                if not visited[w]:
                    if has_cycle(w):
                        return True
                elif path[w]:
                    return True
        
        path[v] = 0
        return False
    
    for v in range(N):
        if not visited[v]:
            if has_cycle(v):
                return False
                
    return True