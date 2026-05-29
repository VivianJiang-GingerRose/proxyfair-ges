"""
Script to visualize causal graphs from experiment results
Generates publication-ready plots for LaTeX/Overleaf papers
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict


def extract_dag_edges_from_log(log_file_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Extract directed and undirected edges from experiment log file (CPDAG before DAG expansion)
    
    Args:
        log_file_path: Path to experiment_log.txt file
        
    Returns:
        Tuple of (directed_edges, undirected_edges) where each is a list of (source, target) tuples
    """
    directed_edges = []
    undirected_edges = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the FIRST "FINAL GES RESULT" section (before DAG expansion)
    # This is the CPDAG that represents the equivalence class
    directed_pattern = r"FINAL GES RESULT - DETAILED EDGE ANALYSIS.*?Directed Edges \((\d+)\):(.*?)(?:Undirected Edges|DoWhy DAG format)"
    directed_matches = re.findall(directed_pattern, content, re.DOTALL)
    
    if directed_matches:
        # Take the FIRST match (the main CPDAG result before expansion to DAGs)
        count, edges_section = directed_matches[0]
        
        # Extract individual edges like "  1. race_cat --> pass_bar"
        edge_pattern = r"\d+\.\s+(\w+)\s+-->\s+(\w+)"
        directed_edges = re.findall(edge_pattern, edges_section)
    
    # Extract undirected edges
    undirected_pattern = r"Undirected Edges \((\d+)\):(.*?)(?:DoWhy DAG format|$)"
    undirected_matches = re.findall(undirected_pattern, content, re.DOTALL)
    
    if undirected_matches:
        count, edges_section = undirected_matches[0]
        # Extract undirected edges like "  1. decile3_cat -- zgpa_cat"
        edge_pattern = r"\d+\.\s+(\w+)\s+--\s+(\w+)"
        undirected_edges = re.findall(edge_pattern, edges_section)
    
    return directed_edges, undirected_edges


def create_readable_label(var_name: str) -> str:
    """
    Convert variable names to readable labels for publication
    
    Args:
        var_name: Variable name like 'race_cat' or 'pass_bar'
        
    Returns:
        Readable label like 'Race' or 'Pass Bar'
    """
    # Remove '_cat' suffix
    label = var_name.replace('_cat', '')
    
    # Special cases for better readability
    label_map = {
        'pass_bar': 'Pass Bar',
        'race': 'Race',
        'male': 'Gender',
        'lsat': 'LSAT',
        'ugpa': 'UGPA',
        'zfygpa': '1st Year\nGPA',
        'zgpa': 'GPA',
        'decile1b': '1st Year\nDecile',
        'decile3': '3rd Year\nDecile',
        'fulltime': 'Full-time\nStatus',
        'fam_inc': 'Family\nIncome',
        'tier': 'School\nTier'
    }
    
    return label_map.get(label, label.replace('_', ' ').title())


def get_node_category(node_name: str) -> str:
    """
    Categorize nodes for visualization coloring
    
    Args:
        node_name: Variable name
        
    Returns:
        Category: 'protected', 'target', 'academic', 'demographic', or 'other'
    """
    if 'race' in node_name or 'male' in node_name or 'gender' in node_name:
        return 'protected'
    elif 'pass_bar' in node_name:
        return 'target'
    elif any(x in node_name for x in ['lsat', 'ugpa', 'gpa', 'decile']):
        return 'academic'
    elif any(x in node_name for x in ['fam_inc', 'fulltime', 'tier']):
        return 'demographic'
    else:
        return 'other'


def plot_causal_graph(edges: List[Tuple[str, str]], 
                     title: str,
                     output_path: str,
                     undirected_edges: List[Tuple[str, str]] = None,
                     figsize: Tuple[int, int] = (14, 10),
                     node_size: int = 4000,
                     font_size: int = 9,
                     dpi: int = 300):
    """
    Create a publication-ready causal graph visualization with hierarchical layout
    
    Args:
        edges: List of (source, target) directed edge tuples
        title: Plot title
        output_path: Path to save the figure
        undirected_edges: List of (node1, node2) undirected edge tuples (optional)
        figsize: Figure size in inches
        node_size: Size of nodes in plot
        font_size: Font size for labels
        dpi: Resolution for saved figure
    """
    if undirected_edges is None:
        undirected_edges = []
    
    # Create directed graph for directed edges
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Add undirected edges to get all nodes for layout
    for u, v in undirected_edges:
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)
    
    # Get all nodes and create readable labels
    nodes = list(G.nodes())
    labels = {node: create_readable_label(node) for node in nodes}
    
    # Categorize nodes for coloring (matching Panel A palette)
    node_colors = []
    for node in nodes:
        category = get_node_category(node)
        if category == 'protected':
            node_colors.append('#E57373')  # Red for protected attributes
        elif category == 'target':
            node_colors.append('#A8D8FF')  # Blue for target variable
        elif category == 'academic':
            node_colors.append('#F8E16C')  # Yellow for academic variables
        elif category == 'demographic':
            node_colors.append('#F8E16C')  # Yellow for demographic
        else:
            node_colors.append('#F8E16C')  # Yellow for other predictor variables
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate nodes by category for hierarchical positioning
    protected_nodes = [n for n in nodes if get_node_category(n) == 'protected']
    target_nodes = [n for n in nodes if get_node_category(n) == 'target']
    academic_nodes = [n for n in nodes if get_node_category(n) == 'academic']
    demographic_nodes = [n for n in nodes if get_node_category(n) == 'demographic']
    other_nodes = [n for n in nodes if get_node_category(n) == 'other']
    
    # Create hierarchical layout with multiple layers (top to bottom)
    pos = {}
    
    # Layer 1 (Top): Protected attributes (root variables)
    for i, node in enumerate(protected_nodes):
        x = (i / max(len(protected_nodes) - 1, 1) - 0.5) * 0.6
        pos[node] = (x, 1.0)
    
    # Layer 2: Family income only
    fam_inc_nodes = [n for n in nodes if 'fam_inc' in n]
    for i, node in enumerate(fam_inc_nodes):
        pos[node] = (0.0, 0.83)
    
    # Layer 3: Full-time status, school tier, and LSAT
    layer3_nodes = [n for n in nodes if any(x in n for x in ['fulltime', 'tier', 'lsat'])]
    for i, node in enumerate(layer3_nodes):
        x = (i / max(len(layer3_nodes) - 1, 1) - 0.5) * 1.0
        pos[node] = (x, 0.67)
    
    # Layer 4: First year GPA and first year decile
    layer4_nodes = [n for n in nodes if any(x in n for x in ['zfygpa', 'decile1b'])]
    for i, node in enumerate(layer4_nodes):
        x = (i / max(len(layer4_nodes) - 1, 1) - 0.5) * 0.6
        pos[node] = (x, 0.50)
    
    # Layer 5: GPA, UGPA, and 3rd year decile
    layer5_nodes = [n for n in nodes if any(x in n for x in ['zgpa', 'ugpa']) and 'zfygpa' not in n]
    layer5_nodes += [n for n in nodes if 'decile3' in n]
    for i, node in enumerate(layer5_nodes):
        x = (i / max(len(layer5_nodes) - 1, 1) - 0.5) * 1.0
        pos[node] = (x, 0.33)
    
    # Layer 6 (Bottom): Target variable (outcome)
    for i, node in enumerate(target_nodes):
        pos[node] = (0, 0.0)
    
    # Draw nodes first with improved styling
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_size,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2.5,
                          ax=ax)
    
    # Separate edges: protected→outcome vs. regular edges
    protected_vars = ['race_cat', 'male_cat']
    outcome_var = 'pass_bar'
    
    # Identify protected→outcome edges for baseline graphs
    protected_outcome_edges = [(u, v) for u, v in edges if u in protected_vars and v == outcome_var]
    
    # Identify ANY edges pointing to outcome (to make them more visible)
    outcome_edges = [(u, v) for u, v in edges if v == outcome_var and u not in protected_vars]
    
    # Regular edges (not pointing to outcome, not protected→outcome)
    regular_edges = [(u, v) for u, v in edges if v != outcome_var and (u, v) not in protected_outcome_edges]
    
    # Draw regular edges AFTER nodes with more transparent grey
    if regular_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=regular_edges,
                              edge_color='#888',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=2.5,
                              alpha=0.3,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=30,
                              min_target_margin=30,
                              ax=ax)
    
    # Draw non-protected edges to outcome with higher opacity (important paths)
    if outcome_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=outcome_edges,
                              edge_color='#333333',  # Darker grey for legitimate outcome paths
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=3.0,
                              alpha=0.8,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=30,
                              min_target_margin=30,
                              ax=ax)
    
    # Draw protected→outcome edges in RED (for baseline graphs)
    if protected_outcome_edges and 'baseline' in title.lower():
        nx.draw_networkx_edges(G, pos,
                              edgelist=protected_outcome_edges,
                              edge_color='#FF0000',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=3.5,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=30,
                              min_target_margin=30,
                              ax=ax)
    elif protected_outcome_edges:
        # In non-baseline graphs, draw them in transparent grey like other edges
        nx.draw_networkx_edges(G, pos,
                              edgelist=protected_outcome_edges,
                              edge_color='#888',
                              arrows=True,
                              arrowsize=20,
                              arrowstyle='->',
                              width=2.5,
                              alpha=0.3,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=30,
                              min_target_margin=30,
                              ax=ax)
    
    # Draw undirected edges in PURPLE (distinct from all other colors)
    if undirected_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=undirected_edges,
                              edge_color='#9370DB',  # Medium purple
                              arrows=False,
                              width=3.5,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.15',
                              ax=ax)
    
    # Draw labels last so they're always visible
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=font_size,
                           font_weight='bold',
                           font_family='sans-serif',
                           ax=ax)
    
    # Add legend with clearer positioning
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E57373', edgecolor='#8B1D2C', label='Protected Attr.', linewidth=1.5),
        Patch(facecolor='#A8D8FF', edgecolor='#2E6DA4', label='Outcome', linewidth=1.5),
        Patch(facecolor='#F8E16C', edgecolor='#9A7B1F', label='Predictors', linewidth=1.5),
        Line2D([0], [0], color='#FF0000', linewidth=2.5, label='Discrimination', marker='>', markersize=6),
        Line2D([0], [0], color='#333333', linewidth=2.5, label='→ Outcome', marker='>', markersize=6),
        Line2D([0], [0], color='#888888', linewidth=2, alpha=0.4, label='Directed', marker='>', markersize=5),
        Line2D([0], [0], color='#9370DB', linewidth=3, label='Undirected')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
             framealpha=0.95, edgecolor='black', fancybox=False, ncol=1)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.15, 1.15)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_single_graph(ax, edges: List[Tuple[str, str]], title: str, 
                     undirected_edges: List[Tuple[str, str]] = None,
                     node_size: int = 2500, font_size: int = 7):
    """Plot a single causal graph on given axes for combined visualization"""
    if undirected_edges is None:
        undirected_edges = []
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Add undirected edges to get all nodes for layout
    for u, v in undirected_edges:
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)
    
    nodes = list(G.nodes())
    labels = {node: create_readable_label(node) for node in nodes}
    
    # Get colors by category (matching Panel A palette)
    node_colors = []
    for node in nodes:
        category = get_node_category(node)
        if category == 'protected':
            node_colors.append('#E57373')  # Red for protected attributes
        elif category == 'target':
            node_colors.append('#A8D8FF')  # Blue for target variable
        elif category == 'academic':
            node_colors.append('#F8E16C')  # Yellow for academic variables
        elif category == 'demographic':
            node_colors.append('#F8E16C')  # Yellow for demographic
        else:
            node_colors.append('#F8E16C')  # Yellow for other predictor variables
    
    # Separate nodes by category for hierarchical positioning
    protected_nodes = [n for n in nodes if get_node_category(n) == 'protected']
    target_nodes = [n for n in nodes if get_node_category(n) == 'target']
    demographic_nodes = [n for n in nodes if get_node_category(n) == 'demographic']
    
    # Create hierarchical layout (top to bottom)
    pos = {}
    
    # Layer 1 (Top): Protected attributes (root variables)
    for i, node in enumerate(protected_nodes):
        x = (i / max(len(protected_nodes) - 1, 1) - 0.5) * 0.6
        pos[node] = (x, 1.0)
    
    # Layer 2: Family income only
    fam_inc_nodes = [n for n in nodes if 'fam_inc' in n]
    for i, node in enumerate(fam_inc_nodes):
        pos[node] = (0.0, 0.83)
    
    # Layer 3: Full-time status, school tier, and LSAT
    layer3_nodes = [n for n in nodes if any(x in n for x in ['fulltime', 'tier', 'lsat'])]
    for i, node in enumerate(layer3_nodes):
        x = (i / max(len(layer3_nodes) - 1, 1) - 0.5) * 1.0
        pos[node] = (x, 0.67)
    
    # Layer 4: First year GPA and first year decile
    layer4_nodes = [n for n in nodes if any(x in n for x in ['zfygpa', 'decile1b'])]
    for i, node in enumerate(layer4_nodes):
        x = (i / max(len(layer4_nodes) - 1, 1) - 0.5) * 0.6
        pos[node] = (x, 0.50)
    
    # Layer 5: GPA, UGPA, and 3rd year decile
    layer5_nodes = [n for n in nodes if any(x in n for x in ['zgpa', 'ugpa']) and 'zfygpa' not in n]
    layer5_nodes += [n for n in nodes if 'decile3' in n]
    for i, node in enumerate(layer5_nodes):
        x = (i / max(len(layer5_nodes) - 1, 1) - 0.5) * 1.0
        pos[node] = (x, 0.33)
    
    # Layer 6 (Bottom): Target variable (outcome)
    for node in target_nodes:
        pos[node] = (0, 0.0)
    
    # Draw nodes first
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_size,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2,
                          ax=ax)
    
    # Separate edges: protected→outcome vs. regular edges
    protected_vars = ['race_cat', 'male_cat']
    outcome_var = 'pass_bar'
    
    # Identify protected→outcome edges for baseline graphs
    protected_outcome_edges = [(u, v) for u, v in edges if u in protected_vars and v == outcome_var]
    
    # Identify ANY edges pointing to outcome (to make them more visible)
    outcome_edges = [(u, v) for u, v in edges if v == outcome_var and u not in protected_vars]
    
    # Regular edges (not pointing to outcome, not protected→outcome)
    regular_edges = [(u, v) for u, v in edges if v != outcome_var and (u, v) not in protected_outcome_edges]
    
    # Draw regular edges AFTER nodes with more transparent grey
    if regular_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=regular_edges,
                              edge_color='#888',
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              width=2.0,
                              alpha=0.3,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=25,
                              min_target_margin=25,
                              ax=ax)
    
    # Draw non-protected edges to outcome with higher opacity (important paths)
    if outcome_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=outcome_edges,
                              edge_color='#333333',  # Darker grey for legitimate outcome paths
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              width=2.5,
                              alpha=0.8,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=25,
                              min_target_margin=25,
                              ax=ax)
    
    # Draw protected→outcome edges in RED (for baseline graphs)
    if protected_outcome_edges and 'baseline' in title.lower():
        nx.draw_networkx_edges(G, pos,
                              edgelist=protected_outcome_edges,
                              edge_color='#FF0000',
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              width=3.0,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=25,
                              min_target_margin=25,
                              ax=ax)
    elif protected_outcome_edges:
        # In non-baseline graphs, draw them in transparent grey like other edges
        nx.draw_networkx_edges(G, pos,
                              edgelist=protected_outcome_edges,
                              edge_color='#888',
                              arrows=True,
                              arrowsize=15,
                              arrowstyle='->',
                              width=2.0,
                              alpha=0.3,
                              connectionstyle='arc3,rad=0.15',
                              min_source_margin=25,
                              min_target_margin=25,
                              ax=ax)
    
    # Draw undirected edges in PURPLE
    if undirected_edges:
        nx.draw_networkx_edges(G, pos,
                              edgelist=undirected_edges,
                              edge_color='#9370DB',  # Medium purple
                              arrows=False,
                              width=3.0,
                              alpha=0.9,
                              connectionstyle='arc3,rad=0.15',
                              ax=ax)
    
    # Draw labels last so they're always visible
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=font_size,
                           font_weight='bold',
                           font_family='sans-serif',
                           ax=ax)
    
    ax.set_title(title, fontsize=10, fontweight='bold', pad=5)
    ax.axis('off')
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.15, 1.15)


def plot_combined_graphs(baseline_edges: List[Tuple[str, str]], 
                        fairness_edges: List[Tuple[str, str]],
                        baseline_undirected: List[Tuple[str, str]],
                        fairness_undirected: List[Tuple[str, str]],
                        output_path: str,
                        figsize: Tuple[int, int] = (12, 5),
                        dpi: int = 300):
    """Create side-by-side comparison plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot baseline
    plot_single_graph(ax1, baseline_edges, 
                     f'A) Statistical Baseline ({len(baseline_edges)} directed, {len(baseline_undirected)} undirected)',
                     undirected_edges=baseline_undirected, node_size=2000, font_size=6.5)
    
    # Plot fairness-enhanced
    plot_single_graph(ax2, fairness_edges, 
                     f'B) Anti-Subordination ({len(fairness_edges)} directed, {len(fairness_undirected)} undirected)',
                     undirected_edges=fairness_undirected, node_size=2000, font_size=6.5)
    
    # Add compact legend to right panel only
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#E57373', edgecolor='#8B1D2C', label='Protected', linewidth=1.5),
        Patch(facecolor='#A8D8FF', edgecolor='#2E6DA4', label='Outcome', linewidth=1.5),
        Patch(facecolor='#F8E16C', edgecolor='#9A7B1F', label='Predictors', linewidth=1.5),
        Line2D([0], [0], color='#FF0000', linewidth=2, label='Discrimination', marker='>', markersize=5),
        Line2D([0], [0], color='#333333', linewidth=2, label='→ Outcome', marker='>', markersize=5),
        Line2D([0], [0], color='#888888', linewidth=1.5, alpha=0.4, label='Directed', marker='>', markersize=4),
        Line2D([0], [0], color='#9370DB', linewidth=2.5, label='Undirected')
    ]
    
    ax2.legend(handles=legend_elements, loc='upper right', 
              fontsize=6.5, framealpha=0.95, edgecolor='black', ncol=1)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved combined plot to: {output_path}")
    plt.close()


def main():
    """Main function to generate causal graphs for law dataset"""
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent.parent
    results_base = project_root / "results" / "experiments" / "law"
    output_dir = project_root / "results" / "paper_results" / "causal_graphs" / "law"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Define experiments to visualize
    experiments = {
        'baseline_20250726_002642': {
            'title': 'Statistical Baseline',
            'output': 'law_baseline_graph.pdf'
        },
        'fairness_20250728_020314': {
            'title': 'Anti-Subordination',
            'output': 'law_fairness_graph.pdf'
        }
    }
    
    # Store edges for combined plot
    all_edges = {}
    all_undirected = {}
    
    # Process each experiment
    for exp_dir, config in experiments.items():
        log_file = results_base / exp_dir / "experiment_log.txt"
        
        if not log_file.exists():
            print(f"✗ Log file not found: {log_file}")
            continue
        
        print(f"\nProcessing {exp_dir}...")
        
        # Extract edges from log
        directed_edges, undirected_edges = extract_dag_edges_from_log(str(log_file))
        print(f"  Found {len(directed_edges)} directed edges and {len(undirected_edges)} undirected edges")
        
        if not directed_edges:
            print(f"  ✗ No edges found in log file")
            continue
        
        # Store edges for later
        all_edges[exp_dir] = directed_edges
        all_undirected[exp_dir] = undirected_edges
        
        # Create visualization
        output_path = output_dir / config['output']
        plot_causal_graph(
            edges=directed_edges,
            undirected_edges=undirected_edges,
            title=config['title'],
            output_path=str(output_path)
        )
        print(f"  ✓ Saved to {output_path}")
    
    print(f"\n{'='*60}")
    print("Individual graph visualization complete!")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSummary:")
    for exp_dir, config in experiments.items():
        if exp_dir in all_edges:
            total = len(all_edges[exp_dir]) + len(all_undirected.get(exp_dir, []))
            print(f"  {config['title']}: {total} total edges ({len(all_edges[exp_dir])} directed, {len(all_undirected.get(exp_dir, []))} undirected)")
    
    # Generate combined comparison plot
    if 'baseline_20250726_002642' in all_edges and 'fairness_20250728_020314' in all_edges:
        print(f"\n{'='*60}")
        print("Generating combined comparison visualization...")
        print(f"{'='*60}")
        
        baseline_edges = all_edges['baseline_20250726_002642']
        fairness_edges = all_edges['fairness_20250728_020314']
        baseline_undirected = all_undirected.get('baseline_20250726_002642', [])
        fairness_undirected = all_undirected.get('fairness_20250728_020314', [])
        
        # Create combined plot (PDF)
        pdf_path = output_dir / "law_combined_comparison.pdf"
        plot_combined_graphs(baseline_edges, fairness_edges, baseline_undirected, fairness_undirected, str(pdf_path), dpi=300)
        
        # Create combined plot (PNG for preview)
        png_path = output_dir / "law_combined_comparison.png"
        plot_combined_graphs(baseline_edges, fairness_edges, baseline_undirected, fairness_undirected, str(png_path), dpi=150)
        
        # Calculate and display differences
        baseline_set = set(baseline_edges)
        fairness_set = set(fairness_edges)
        
        removed_edges = baseline_set - fairness_set
        added_edges = fairness_set - baseline_set
        
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        print(f"Baseline edges: {len(baseline_edges)}")
        print(f"Fairness edges: {len(fairness_edges)}")
        print(f"Common edges: {len(baseline_set & fairness_set)}")
        print(f"Removed by fairness constraints: {len(removed_edges)}")
        print(f"Added by fairness constraints: {len(added_edges)}")
        
        # Check for discriminatory edges
        print(f"\n{'='*70}")
        print("KEY FAIRNESS IMPROVEMENTS")
        print(f"{'='*70}")
        
        discriminatory_edges = [
            ('race_cat', 'pass_bar'),
            ('male_cat', 'pass_bar'),
            ('fam_inc_cat', 'pass_bar')
        ]
        
        for edge in discriminatory_edges:
            src, tgt = edge
            in_baseline = edge in baseline_set
            in_fairness = edge in fairness_set
            
            if in_baseline and not in_fairness:
                print(f"✓ REMOVED: {src} → {tgt} (direct discrimination path)")
            elif in_baseline and in_fairness:
                print(f"✗ STILL PRESENT: {src} → {tgt}")
            elif not in_baseline and not in_fairness:
                print(f"  Not present in either: {src} → {tgt}")
    
    print(f"\n{'='*60}")
    print("All visualizations complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
