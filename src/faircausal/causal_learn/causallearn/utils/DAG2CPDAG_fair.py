import numpy as np
from typing import Optional, List

from src.faircausal.causal_learn.causallearn.graph.Dag import Dag
from src.faircausal.causal_learn.causallearn.graph.Edge import Edge
from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint
from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph

def dag2cpdag(G: Dag) -> GeneralGraph:
    """
    Convert a DAG to its corresponding PDAG

    Parameters
    ----------
    G : Direct Acyclic Graph

    Returns
    -------
    CPDAG : Completed Partially Direct Acyclic Graph

    Authors
    -------
    Yuequn Liu@dmirlab, Wei Chen@dmirlab, Kun Zhang@CMU
    """

    # order the edges in G
    nodes_order = list(
        map(lambda x: G.node_map[x], G.get_causal_ordering())
    )  # Perform a topological sort on the nodes of G
    # nodes_order(1) is the node which has the highest order
    # nodes_order(N) is the node which has the lowest order
    edges_order = np.empty((0, 2), dtype=np.int64)
    # edges_order(1,:) is the edge which has the highest order
    # edges_order(M,:) is the edge which has the lowest order
    M = G.get_num_edges()  # the number of edges in this DAG
    N = G.get_num_nodes()  # the number of nodes in this DAG
    i, j = 0, 0
    while edges_order.shape[0] < M:
        for ny in range(N - 1, -1, -1):
            j = nodes_order[ny]
            inci_all = np.where(G.graph[j, :] == 1)[
                0
            ]  # all the edges that incident to j
            if len(inci_all) != 0:
                if len(edges_order) != 0:
                    inci = edges_order[
                        np.where(edges_order[:, 1] == j)[0], 0
                    ]  # ordered edge that incident to j
                    if len(set(inci_all) - set(inci.tolist())) != 0:
                        break
                else:
                    break
        for nx in range(N):
            i = nodes_order[nx]
            if len(edges_order) != 0:
                if (
                    len(
                        np.intersect1d(
                            np.where(edges_order[:, 1] == j)[0],
                            np.where(edges_order[:, 0] == i)[0],
                        )
                    )
                    == 0
                    and G.graph[j, i] == 1
                ):
                    break
            else:
                if G.graph[j, i] == 1:
                    break
        edges_order = np.r_[edges_order, np.array([[i, j]])]

    ## ----------------------------------------------------------------
    sign_edges = np.zeros(M)  # 0 means unknown, 1 means compelled, -1 means reversible
    while len(np.where(sign_edges == 0)[0]) != 0:
        ss = 0
        for m in range(
            M - 1, -1, -1
        ):  # let x->y be the lowest ordered edge that is labeled "unknown"
            if sign_edges[m] == 0:
                i = edges_order[m, 0]
                j = edges_order[m, 1]
                break
        idk = np.where(edges_order[:, 1] == i)[0]
        k = edges_order[idk, 0]  # w->x
        for m in range(len(k)):
            if sign_edges[idk[m]] == 1:
                if G.graph[j, k[m]] != 1:  # if w is not a parent of y
                    _id = np.where(edges_order[:, 1] == j)[
                        0
                    ]  # label every edge that incident into y with "complled"
                    sign_edges[_id] = 1
                    ss = 1
                    break
                else:
                    _id = np.intersect1d(
                        np.where(edges_order[:, 0] == k[m])[0],
                        np.where(edges_order[:, 1] == j)[0],
                    )  # label w->y with "complled"
                    sign_edges[_id] = 1
        if ss:
            continue

        z = np.where(G.graph[j, :] == 1)[0]
        if (
            len(
                np.intersect1d(
                    np.setdiff1d(z, i),
                    np.union1d(
                        np.union1d(
                            np.where(G.graph[i, :] == 0)[0],
                            np.where(G.graph[i, :] == -1)[0],
                        ),
                        np.intersect1d(
                            np.where(G.graph[i, :] == -1)[0],
                            np.where(G.graph[:, i] == -1)[0],
                        ),
                    ),
                )
            )
            != 0
        ):
            _id = np.intersect1d(
                np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0]
            )
            sign_edges[_id] = 1  # label x->y  with "compelled"
            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = (
                1  # label all "unknown" edges incident into y  with "complled"
            )
        else:
            _id = np.intersect1d(
                np.where(edges_order[:, 0] == i)[0], np.where(edges_order[:, 1] == j)[0]
            )
            sign_edges[_id] = -1  # label x->y with "reversible"

            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = (
                -1
            )  # label all "unknown" edges incident into y with "reversible"

    # create CPDAG according the labelled edge
    nodes = G.get_nodes()
    CPDAG = GeneralGraph(nodes)
    for m in range(M):
        if sign_edges[m] == 1:
            CPDAG.add_edge(
                Edge(
                    nodes[edges_order[m, 0]],
                    nodes[edges_order[m, 1]],
                    Endpoint.TAIL,
                    Endpoint.ARROW,
                )
            )
        else:
            CPDAG.add_edge(
                Edge(
                    nodes[edges_order[m, 0]],
                    nodes[edges_order[m, 1]],
                    Endpoint.TAIL,
                    Endpoint.TAIL,
                )
            )

    return CPDAG

def get_topological_order(G: GeneralGraph) -> List[int]:
    """
    Compute a topological ordering of nodes in a DAG.
    
    Args:
        G: GeneralGraph representing a DAG
        
    Returns:
        List of node indices in topological order
    """
    N = G.get_num_nodes()
    visited = [False] * N
    order = []
    
    def visit(i: int):
        if visited[i]:
            return
        visited[i] = True
        # In DAG class, 1 represents incoming arrow, -1 represents outgoing tail
        for j in range(N):
            if G.graph[i, j] == 1:  # There is an edge j->i
                visit(j)
        order.append(i)
    
    for i in range(N):
        if not visited[i]:
            visit(i)
            
    return list(reversed(order))

def dag2cpdag_with_blacklist(G: GeneralGraph, blacklist_matrix: Optional[np.ndarray] = None) -> GeneralGraph:
    """
    Convert a DAG to its corresponding PDAG while respecting blacklist constraints.
    Works with both Dag and GeneralGraph instances.

    Parameters
    ----------
    G : GeneralGraph
        Input graph in DAG form
    blacklist_matrix : Optional[np.ndarray]
        Boolean matrix where blacklist_matrix[i,j] = True indicates edge i->j is forbidden

    Returns
    -------
    CPDAG : GeneralGraph
    """
    if blacklist_matrix is None:
        return dag2cpdag(G)
    
    N = G.get_num_nodes()
    nodes = G.get_nodes()
    
    # Get topological order
    nodes_order_indices = get_topological_order(G)
    
    # Build edges_order
    edges_order = []
    for j in nodes_order_indices:
        for i in range(N):
            # In DAG class, edge i->j is represented as:
            # G.graph[j,i] = 1 (arrow at j)
            # G.graph[i,j] = -1 (tail at i)
            if G.graph[j,i] == 1 and G.graph[i,j] == -1:
                edges_order.append([i, j])
    edges_order = np.array(edges_order)
    M = len(edges_order)
    
    # Initialize edge signs (0=unknown, 1=compelled, -1=reversible)
    sign_edges = np.zeros(M)
    
    while len(np.where(sign_edges == 0)[0]) != 0:
        ss = 0
        # Find lowest ordered unknown edge x->y
        for m in range(M - 1, -1, -1):
            if sign_edges[m] == 0:
                i = edges_order[m, 0]  # x
                j = edges_order[m, 1]  # y
                break
                
        # Check if edge should be compelled due to blacklist
        if blacklist_matrix[j, i]:  # If y->x is blacklisted
            sign_edges[m] = 1
            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = 1
            continue
            
        # Regular compelled edge checks
        idk = np.where(edges_order[:, 1] == i)[0]
        k = edges_order[idk, 0] if len(idk) > 0 else np.array([])
        
        for n in range(len(k)):
            if sign_edges[idk[n]] == 1:
                # Check if w->y is not in graph
                if not (G.graph[j,k[n]] == 1 and G.graph[k[n],j] == -1):
                    id1 = np.where(edges_order[:, 1] == j)[0]
                    sign_edges[id1] = 1
                    ss = 1
                    break
                else:
                    id1 = np.intersect1d(np.where(edges_order[:, 0] == k[n])[0],
                                       np.where(edges_order[:, 1] == j)[0])
                    if len(id1) > 0:
                        sign_edges[id1[0]] = 1
        if ss:
            continue
            
        # Check for discriminating paths
        parent_j = [idx for idx in range(N) 
                   if G.graph[j,idx] == 1 and G.graph[idx,j] == -1]
        
        any_discriminating = False
        for p in parent_j:
            if p != i:  # Skip the edge we're examining
                # Check if p is not adjacent to i
                if G.graph[i,p] == 0 and G.graph[p,i] == 0:
                    any_discriminating = True
                    break
                    
        if any_discriminating:
            # Mark edge as compelled
            id1 = np.intersect1d(np.where(edges_order[:, 0] == i)[0],
                               np.where(edges_order[:, 1] == j)[0])
            sign_edges[id1] = 1
            # Mark all unknown edges into j as compelled
            id1 = np.where(edges_order[:, 1] == j)[0]
            id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
            sign_edges[id2] = 1
        else:
            # Check blacklist before marking as reversible
            if blacklist_matrix[j, i]:
                sign_edges[m] = 1  # Keep as compelled if reverse is blacklisted
            else:
                sign_edges[m] = -1  # Mark as reversible
                # Mark remaining unknown edges into j as reversible
                id1 = np.where(edges_order[:, 1] == j)[0]
                id2 = np.intersect1d(np.where(sign_edges == 0)[0], id1)
                sign_edges[id2] = -1

    # Create CPDAG
    CPDAG = GeneralGraph(nodes)
    for m in range(M):
        i, j = edges_order[m]
        if sign_edges[m] == 1 or blacklist_matrix[j, i]:
            # Add directed edge if compelled or reverse direction is blacklisted
            CPDAG.graph[j,i] = 1  # Arrow at j
            CPDAG.graph[i,j] = -1  # Tail at i
        else:
            # For undirected edges, need to check both directions aren't blacklisted
            if not blacklist_matrix[i,j] and not blacklist_matrix[j,i]:
                CPDAG.graph[i,j] = -1
                CPDAG.graph[j,i] = -1
            else:
                # If one direction is blacklisted, add the allowed direction
                if not blacklist_matrix[i,j]:
                    CPDAG.graph[j,i] = 1
                    CPDAG.graph[i,j] = -1
                else:
                    CPDAG.graph[i,j] = 1
                    CPDAG.graph[j,i] = -1
    
    return CPDAG