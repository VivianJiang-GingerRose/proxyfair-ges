#!/usr/bin/env python3
"""
Generate publication-ready visualizations for FAccT paper.
Processes robustness analysis outputs into figures and tables.

Usage:
    python generate_facct_paper_visualizations.py \
        --input-dir results/paper_results/llm_analysis/dataset_framework_analysis \
        --output-dir paper_figures

Requirements:
    - pandas, numpy, matplotlib, seaborn, networkx
    - CSV files from fairness_framework_analysis.py
"""

import argparse
import json
import math
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import Patch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class FAccTVisualizationGenerator:
    """Generate all visualizations for FAccT paper submission."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        
        # Create output directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Publication settings
        self._setup_matplotlib()
        
        # Framework ordering and colors (CONSISTENT)
        self.frameworks = ['Statistical', 'AntiClassification', 
                          'Meritocratic', 'AntiSubordination']
        # Softer gradient ensures Panel B bars read as one family while retaining order cues
        # Panel B palette intentionally diverges from Panel A node hues to tell a distinct story
        self.framework_colors = {
            'Statistical': '#A0A4AD',        # medium grey
            'AntiClassification': '#D6F1D7',  # light mint
            'Meritocratic': '#7FCF96',       # leaf green
            'AntiSubordination': '#2E6A4C'   # deep forest
        }
        self.type_b_fill_color = '#F3B7C5'
        self.type_b_line_color = '#B34763'

        # Node palette for contested-edge visualizations
        self.node_styles = {
            'predictor': {'facecolor': '#F8E16C', 'edgecolor': '#9A7B1F'},  # Sunlit yellow
            'protected': {'facecolor': '#E57373', 'edgecolor': '#8B1D2C'},  # Fairness red
            'outcome': {'facecolor': '#A8D8FF', 'edgecolor': '#2E6DA4'}  # Light sky blue
        }
        self.rare_edge_color = '#B0B8C5'
        
        # Datasets
        self.datasets = ['law', 'compas', 'dutch', 'bank']
        self.dataset_labels = {
            'bank': 'Bank',
            'compas': 'Compas',
            'dutch': 'Dutch',
            'law': 'Law'
        }
        
        # Outcome variables for each dataset
        self.outcome_variables = {
            'bank': 'y',
            'compas': 'two_year_recid',
            'dutch': 'occupation',
            'law': 'pass_bar'
        }

        # Law dataset label mapping (aligned with visualize_causal_graphs.py)
        self.law_label_map = {
            'pass_bar': 'Pass Bar',
            'race': 'Race',
            'gender': 'Gender',
            'male': 'Gender',
            'sex': 'Sex',
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
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication quality."""
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10.5
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 11.5
        plt.rcParams['axes.titlesize'] = 12.5
        plt.rcParams['xtick.labelsize'] = 9.5
        plt.rcParams['ytick.labelsize'] = 9.5
        plt.rcParams['legend.fontsize'] = 9.0

    @staticmethod
    def _darken_color(color: str, factor: float = 0.75) -> str:
        """Return a darker shade of the given hex color."""
        import matplotlib.colors as mcolors

        rgb = np.array(mcolors.to_rgb(color))
        darker = np.clip(rgb * factor, 0, 1)
        return mcolors.to_hex(darker)
    
    def _find_latest_dataset_dir(self, dataset: str) -> Path:
        """Find the latest timestamped directory for a dataset."""
        pattern = f"{dataset}_*"
        matching_dirs = list(self.input_dir.glob(pattern))
        
        if not matching_dirs:
            raise FileNotFoundError(f"No directory found for dataset '{dataset}' in {self.input_dir}")
        
        # Sort by timestamp (directory name) and return the latest
        latest_dir = sorted(matching_dirs)[-1]
        print(f"  Using directory: {latest_dir.name}")
        return latest_dir
    
    def _load_contested_report(self, dataset: str) -> pd.DataFrame:
        """Load contested zone report for a dataset."""
        dataset_dir = self._find_latest_dataset_dir(dataset)
        report_path = dataset_dir / f"{dataset}_contested_zone_report.csv"
        
        if not report_path.exists():
            raise FileNotFoundError(f"Contested zone report not found: {report_path}")
        
        return pd.read_csv(report_path)
    
    def _load_framework_comparison(self, dataset: str) -> pd.DataFrame:
        """Load framework comparison for a dataset."""
        dataset_dir = self._find_latest_dataset_dir(dataset)
        comparison_path = dataset_dir / f"{dataset}_framework_comparison.csv"
        
        if not comparison_path.exists():
            raise FileNotFoundError(f"Framework comparison not found: {comparison_path}")
        
        return pd.read_csv(comparison_path)
    
    def _load_parsed_constraints(self, dataset: str) -> List[Dict]:
        """Load parsed constraint JSON for a dataset."""
        dataset_dir = self._find_latest_dataset_dir(dataset)
        parsed_path = dataset_dir / f"{dataset}_parsed_constraints.json"
        
        if not parsed_path.exists():
            raise FileNotFoundError(f"Parsed constraints not found: {parsed_path}")
        
        with parsed_path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    
    @staticmethod
    def _normalize_edge(edge: List[str]) -> Optional[Tuple[str, str]]:
        """Normalize a blocked edge representation into a (source, target) tuple."""
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source = str(edge[0])
            target = str(edge[1])
            return source, target
        return None
    
    def _format_variable_name(self, variable: Optional[str], dataset: Optional[str] = None) -> str:
        """Return display-ready variable names with dataset-specific formatting."""
        if variable is None:
            return ''
        if isinstance(variable, float) and math.isnan(variable):
            return ''
        if pd.isna(variable):
            return ''
        value = str(variable).strip()
        if not value:
            return value
        
        if dataset == 'law':
            label = value.replace('_cat', '')
            label_key = label.lower()
            if label_key in self.law_label_map:
                return self.law_label_map[label_key]
            return label.replace('_', ' ').title()
        
        normalized = value
        if normalized == 'male_cat':
            normalized = 'gender'
        if normalized.endswith('_cat'):
            normalized = normalized[:-4]
        return normalized
    
    @staticmethod
    def _is_protected_attribute(node_name: Optional[str]) -> bool:
        """Identify protected attribute nodes using keyword matching."""
        if not node_name:
            return False
        name = str(node_name).lower()
        keywords = ('gender', 'race', 'sex', 'male', 'female', 'ethnicity')
        return any(keyword in name for keyword in keywords)
    
    def _clean_variable_names(self, df: pd.DataFrame, dataset: Optional[str] = None) -> pd.DataFrame:
        """Clean and format variable names for visualization outputs."""
        df = df.copy()
        df['source'] = df['source'].apply(lambda val: self._format_variable_name(val, dataset))
        df['target'] = df['target'].apply(lambda val: self._format_variable_name(val, dataset))
        return df
    
    def create_readable_label(self, var_name: str, use_line_breaks: bool = True) -> str:
        """
        Convert variable names to readable labels for publication (matching visualize_causal_graphs.py).
        
        Args:
            var_name: Variable name like 'race_cat' or 'pass_bar' (or already formatted like 'Pass Bar')
            use_line_breaks: If True, use line breaks for multi-word labels (for network plots).
                           If False, use spaces instead (for tables/bar charts).
        """
        # Remove '_cat' suffix
        label = var_name.replace('_cat', '')
        
        # Convert to lowercase for lookup (to handle already-formatted names)
        label_key = label.lower().replace(' ', '_').replace('\n', '_')
        
        # Special cases for better readability (law dataset focused)
        # Using line breaks for network visualizations
        label_map = {
            'pass_bar': 'Pass Bar',
            'race': 'Race',
            'male': 'Gender',
            'gender': 'Gender',
            'lsat': 'LSAT',
            'ugpa': 'UGPA',
            'zfygpa': '1st Year\nGPA' if use_line_breaks else '1st Year GPA',
            'zgpa': 'GPA',
            'decile1b': '1st Year\nDecile' if use_line_breaks else '1st Year Decile',
            'decile3': '3rd Year\nDecile' if use_line_breaks else '3rd Year Decile',
            'fulltime': 'Full-time\nStatus' if use_line_breaks else 'Full-time Status',
            'fam_inc': 'Family\nIncome' if use_line_breaks else 'Family Income',
            'family_income': 'Family\nIncome' if use_line_breaks else 'Family Income',
            'tier': 'School\nTier' if use_line_breaks else 'School Tier',
            'school_tier': 'School\nTier' if use_line_breaks else 'School Tier',
            '1st_year_gpa': '1st Year\nGPA' if use_line_breaks else '1st Year GPA',
            '1st_year_decile': '1st Year\nDecile' if use_line_breaks else '1st Year Decile',
            '3rd_year_decile': '3rd Year\nDecile' if use_line_breaks else '3rd Year Decile',
            'full-time_status': 'Full-time\nStatus' if use_line_breaks else 'Full-time Status'
        }
        
        # Try exact match first, then try normalized key
        if label in label_map:
            return label_map[label]
        elif label_key in label_map:
            return label_map[label_key]
        
        # If already formatted (contains spaces/newlines), return as-is
        if ' ' in label or '\n' in label:
            return label
        
        # Default: title case with underscores replaced by spaces
        return label.replace('_', ' ').title()
    
    def generate_panel_a_network(self, dataset: str = 'law') -> plt.Axes:
        """
        Generate Panel A: Color-coded network showing contested and consensus_prohibited edges.
        Edge color represents severity (how many fairness frameworks block it).
        
        Statistical framework permits these edges, but fairness frameworks block them.
        Includes both contested (2/4 frameworks block) and consensus_prohibited (3/4 frameworks block).
        Also includes weakly blocked edges (stored under category 'rarely_blocked' and
        shown with faint dotted lines).
        """
        print(f"  Loading contested zone report for {dataset}...")
        df = self._load_contested_report(dataset)
        
        # Clean variable names for better readability
        df = self._clean_variable_names(df, dataset)
        
        # Check if p_blocked column exists, otherwise compute it
        if 'p_blocked' not in df.columns:
            if 'n_frameworks_blocking' in df.columns and 'n_frameworks_total' in df.columns:
                df['p_blocked'] = df['n_frameworks_blocking'] / df['n_frameworks_total']
            else:
                df['p_blocked'] = 0.0
        
        # Filter to show contested AND consensus_prohibited edges
        # (Statistical permits, but fairness frameworks block)
        contested_df = df[df['category'].isin(['contested', 'consensus_prohibited'])].copy()
        
        # Also get weakly blocked edges (category label: rarely_blocked) to show with faint dotted lines
        rarely_blocked_df = df[df['category'] == 'rarely_blocked'].copy()
        
        print(f"    Found {len(df[df['category'] == 'contested'])} contested edges")
        print(f"    Found {len(df[df['category'] == 'consensus_prohibited'])} consensus_prohibited edges")
        print(f"    Found {len(rarely_blocked_df)} weakly blocked edges")
        print(f"    Total edges to visualize: {len(contested_df) + len(rarely_blocked_df)}")
        
        # Calculate severity: count how many fairness frameworks block each edge
        # Exclude Statistical framework from severity calculation
        fairness_frameworks = ['AntiClassification', 'Meritocratic', 'AntiSubordination']
        
        def calculate_severity(row):
            """Count how many fairness frameworks block this edge."""
            count = 0
            for fw in fairness_frameworks:
                blocked_col = f'blocked_by_{fw}'
                if blocked_col in row.index and row[blocked_col] == True:
                    count += 1
            return count
        
        contested_df['severity'] = contested_df.apply(calculate_severity, axis=1)
        
        # Calculate severity for weakly blocked edges too
        if len(rarely_blocked_df) > 0:
            rarely_blocked_df['severity'] = rarely_blocked_df.apply(calculate_severity, axis=1)
        
        # Create directed graph with contested AND weakly blocked edges
        G = nx.DiGraph()
        
        # Add contested edges with severity attribute
        for _, row in contested_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                severity=row['severity'],
                edge_type='contested'
            )
        
        # Add weakly blocked edges with severity attribute
        for _, row in rarely_blocked_df.iterrows():
            G.add_edge(
                row['source'],
                row['target'],
                severity=row['severity'],
                edge_type='rarely_blocked'
            )
        
        # Auto-layout using hierarchical positioning
        raw_outcome = self.outcome_variables.get(dataset, 'y')
        outcome_label = self._format_variable_name(raw_outcome, dataset)
        pos = self._auto_hierarchical_layout(G, outcome_label)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Separate edges by severity and type
        # Contested edges (solid lines)
        severity_3_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 3 and d.get('edge_type') == 'contested']
        severity_2_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 2 and d.get('edge_type') == 'contested']
        severity_1_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 1 and d.get('edge_type') == 'contested']
        
        # Weakly blocked edges (dotted lines) - grouped by severity for consistency
        rarely_blocked_edges_all = [(u, v) for u, v, d in G.edges(data=True) 
                                    if d.get('edge_type') == 'rarely_blocked']
        
        # Draw edges with color-coded severity
        # Dark red: All 3 fairness frameworks block
        if severity_3_edges:
            nx.draw_networkx_edges(
                G, pos, 
                edgelist=severity_3_edges,
                edge_color='#8B0000',  # Dark red
                width=2.5,
                alpha=0.9,
                style='solid',
                arrows=True,
                arrowsize=16,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                min_source_margin=32,
                min_target_margin=32,
                ax=ax
            )
        
        # Red: 2 fairness frameworks block
        if severity_2_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=severity_2_edges,
                edge_color='#D62728',  # Red
                width=2.5,
                alpha=0.9,
                style='solid',
                arrows=True,
                arrowsize=16,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                min_source_margin=32,
                min_target_margin=32,
                ax=ax
            )
        
        # Orange: 1 fairness framework blocks
        if severity_1_edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=severity_1_edges,
                edge_color='#FF8C00',  # Dark orange
                width=2.5,
                alpha=0.9,
                style='solid',
                arrows=True,
                arrowsize=16,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                min_source_margin=32,
                min_target_margin=32,
                ax=ax
            )
        
        if rarely_blocked_edges_all:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=rarely_blocked_edges_all,
                edge_color=self.rare_edge_color,
                width=2,
                alpha=0.5,  # Faint
                style='dotted',  # Dotted line style
                arrows=True,
                arrowsize=16,
                arrowstyle='-|>',
                connectionstyle='arc3,rad=0.1',
                min_source_margin=32,
                min_target_margin=32,
                ax=ax
            )
        
        # Draw nodes
        # Identify protected attributes (using cleaned names without _cat suffix)
        protected_nodes = [n for n in G.nodes() if self._is_protected_attribute(n)]
        
        # Get outcome variable for this dataset
        outcome_nodes = [n for n in G.nodes() if n == outcome_label]
        
        # Regular nodes (neither protected nor outcome)
        regular_nodes = [n for n in G.nodes() if n not in protected_nodes and n not in outcome_nodes]
        
        predictor_style = self.node_styles['predictor']
        protected_style = self.node_styles['protected']
        outcome_style = self.node_styles['outcome']

        # Draw regular nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_nodes,
            node_color=predictor_style['facecolor'],
            node_size=2500,
            edgecolors=predictor_style['edgecolor'],
            linewidths=2.0,
            ax=ax
        )
        
        # Draw protected attribute nodes (pastel blue)
        if protected_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=protected_nodes,
                node_color=protected_style['facecolor'],
                node_size=2800,
                edgecolors=protected_style['edgecolor'],
                linewidths=2.5,
                ax=ax
            )
        
        # Draw outcome nodes (pale green)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=outcome_nodes,
            node_color=outcome_style['facecolor'],
            node_size=3200,
            edgecolors=outcome_style['edgecolor'],
            linewidths=2.0,
            ax=ax
        )
        
        # Draw labels with readable names
        labels = {node: self.create_readable_label(node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor=predictor_style['facecolor'], edgecolor=predictor_style['edgecolor'], linewidth=2,
                  label='Predictor Variables'),
            Patch(facecolor=protected_style['facecolor'], edgecolor=protected_style['edgecolor'], linewidth=2.5,
                  label='Protected Attributes'),
            Patch(facecolor=outcome_style['facecolor'], edgecolor=outcome_style['edgecolor'], linewidth=2,
                  label='Outcome'),
            Line2D([0], [0], color='#8B0000', linewidth=3.5, alpha=0.9,
                  label='Type B ban (3 FWs)'),
            Line2D([0], [0], color='#D62728', linewidth=3.0, alpha=0.85,
                  label='Type B ban (2 FWs)'),
            Line2D([0], [0], color='#FF8C00', linewidth=2.5, alpha=0.8,
                  label='Type B flag (1 FW)'),
            Line2D([0], [0], color=self.rare_edge_color, linewidth=1.5, alpha=0.4,
                   linestyle='dotted',
                   label='Type B provisional\n(weak signal)')
        ]

        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
              fontsize=7.5, prop={'weight': 'bold'})
        ax.set_title(
            f'Type B (Normative) Edge Blocks: {self.dataset_labels.get(dataset, dataset).title()} Dataset',
            fontsize=12.5,
            fontweight='bold'
        )
        ax.axis('off')
        
        plt.tight_layout()
        return ax
    
    def _auto_hierarchical_layout(self, G: nx.DiGraph, outcome_var: str = 'y') -> Dict:
        """Auto-generate hierarchical layout that shows intermediate edges."""
        # Use specified outcome variable if it exists in graph
        if outcome_var in G.nodes():
            outcome = outcome_var
        else:
            # Find outcome node (assuming it has no out-edges)
            outcome_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
            if not outcome_nodes:
                # Fallback to spring layout
                return nx.spring_layout(G, k=2, iterations=50)
            outcome = outcome_nodes[0]
        
        # Compute hierarchical levels using longest path from each node to outcome
        levels = {}
        
        # Initialize: outcome is at level 0 (rightmost)
        levels[outcome] = 0
        
        # Work backwards from outcome using BFS
        from collections import deque
        queue = deque([outcome])
        visited = {outcome}
        
        while queue:
            node = queue.popleft()
            current_level = levels[node]
            
            # All predecessors should be at least one level earlier
            for pred in G.predecessors(node):
                if pred not in levels:
                    levels[pred] = current_level + 1
                else:
                    levels[pred] = max(levels[pred], current_level + 1)
                
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        
        # Handle any nodes not connected to outcome (shouldn't happen but just in case)
        for node in G.nodes():
            if node not in levels:
                levels[node] = max(levels.values()) + 1 if levels else 1
        
        # Group nodes by level
        max_level = max(levels.values())
        level_nodes = {i: [] for i in range(max_level + 1)}
        for node, level in levels.items():
            level_nodes[level].append(node)
        
        # Position nodes
        pos = {}
        x_spacing = 3.0  # Horizontal spacing between levels
        y_spacing = 1.5  # Vertical spacing between nodes
        
        for level in range(max_level + 1):
            nodes_at_level = level_nodes[level]
            n_nodes = len(nodes_at_level)
            
            # X position: higher level = further left
            x = (max_level - level) * x_spacing
            
            # Y positions: center the nodes vertically
            if n_nodes == 1:
                y_positions = [0]
            else:
                y_start = -(n_nodes - 1) * y_spacing / 2
                y_positions = [y_start + i * y_spacing for i in range(n_nodes)]
            
            # Sort nodes alphabetically for consistent positioning
            # Reverse sort so higher alphabetically = higher on graph (top)
            nodes_at_level.sort(reverse=True)
            
            for node, y in zip(nodes_at_level, y_positions):
                pos[node] = (x, y)
        
        return pos
    
    def generate_framework_blocking_summary(self, force_recompute: bool = True) -> pd.DataFrame:
        """Compute and persist majority-agreed blocked edges per dataset and framework."""
        summary_path = self.tables_dir / 'framework_blocking_majority_summary.csv'
        if not force_recompute and summary_path.exists():
            print(f"\nLoading existing framework blocking summary: {summary_path}")
            summary_df = pd.read_csv(summary_path)
            return summary_df
        
        print("\nAnalyzing parsed constraints for framework blocking summary (majority vote)...")
        dataset_order = {ds: idx for idx, ds in enumerate(self.datasets)}
        framework_order = {fw: idx for idx, fw in enumerate(self.frameworks)}
        summary_records: List[Dict] = []
        fairness_note = ("    {dataset_label} - {framework}: {total} majority-agreed blocked edges "
                 "(typeA={type_a}, typeB={type_b}, providers={providers}, threshold={threshold})")
        
        for dataset in self.datasets:
            dataset_label = self.dataset_labels.get(dataset, dataset).title()
            try:
                parsed_constraints = self._load_parsed_constraints(dataset)
            except FileNotFoundError as exc:
                print(f"  WARNING: {exc}")
                parsed_constraints = []
            
            for framework in self.frameworks:
                framework_entries = [c for c in parsed_constraints if c.get('framework') == framework]
                provider_names = set()
                edge_votes = {
                    'type_a': defaultdict(set),
                    'type_b': defaultdict(set)
                }
                
                for entry in framework_entries:
                    provider = entry.get('provider') or entry.get('model') or entry.get('run_id')
                    if provider is None:
                        provider = f"provider_{len(provider_names) + 1}"
                    provider = str(provider)
                    provider_names.add(provider)
                    for edge in entry.get('blacklisted_type_a') or []:
                        normalized_edge = self._normalize_edge(edge)
                        if normalized_edge:
                            edge_votes['type_a'][normalized_edge].add(provider)
                    for edge in entry.get('blacklisted_type_b') or []:
                        normalized_edge = self._normalize_edge(edge)
                        if normalized_edge:
                            edge_votes['type_b'][normalized_edge].add(provider)
                
                provider_count = len(provider_names)
                threshold = math.floor(provider_count / 2) + 1 if provider_count > 0 else 0
                if threshold == 0:
                    agreed_edges_a = []
                    agreed_edges_b = []
                else:
                    agreed_edges_a = [edge for edge, voters in edge_votes['type_a'].items() if len(voters) >= threshold]
                    agreed_edges_b = [edge for edge, voters in edge_votes['type_b'].items() if len(voters) >= threshold]
                total_agreed = len(agreed_edges_a) + len(agreed_edges_b)
                
                summary_records.append({
                    'dataset': dataset,
                    'Dataset': dataset_label,
                    'Framework': framework,
                    'MajorityBlockedEdges': total_agreed,
                    'MajorityBlockedEdges_TypeA': len(agreed_edges_a),
                    'MajorityBlockedEdges_TypeB': len(agreed_edges_b),
                    'ProvidersReporting': provider_count,
                    'MajorityThreshold': threshold,
                    'UniqueEdgesObserved': len(edge_votes['type_a']) + len(edge_votes['type_b']),
                    'dataset_order': dataset_order[dataset],
                    'framework_order': framework_order[framework]
                })
                print(fairness_note.format(
                    dataset_label=dataset_label,
                    framework=framework,
                    total=total_agreed,
                    type_a=len(agreed_edges_a),
                    type_b=len(agreed_edges_b),
                    providers=provider_count,
                    threshold=threshold
                ))
        
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df = summary_df.sort_values(['dataset_order', 'framework_order']).reset_index(drop=True)
            summary_df = summary_df.drop(columns=['dataset_order', 'framework_order'])
            int_cols = ['MajorityBlockedEdges', 'MajorityBlockedEdges_TypeA', 'MajorityBlockedEdges_TypeB',
                        'ProvidersReporting', 'MajorityThreshold', 'UniqueEdgesObserved']
            summary_df[int_cols] = summary_df[int_cols].astype(int)
        else:
            summary_df = pd.DataFrame(columns=[
                'dataset', 'Dataset', 'Framework', 'MajorityBlockedEdges',
                'MajorityBlockedEdges_TypeA', 'MajorityBlockedEdges_TypeB',
                'ProvidersReporting', 'MajorityThreshold', 'UniqueEdgesObserved'
            ])
        
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved summary CSV: {summary_path}")
        summary_df = pd.read_csv(summary_path)
        return summary_df
    
    def _prepare_majority_edge_counts(self, force_recompute: bool = False):
        """Load majority summary and return counts per dataset/framework for fairness plots."""
        summary_df = self.generate_framework_blocking_summary(force_recompute=force_recompute)
        edge_counts = {
            dataset: {
                framework: {'type_a': 0, 'type_b': 0}
                for framework in self.frameworks
            }
            for dataset in self.datasets
        }
        if not summary_df.empty:
            for _, row in summary_df.iterrows():
                dataset_id = row['dataset']
                framework = row['Framework']
                if dataset_id in edge_counts and framework in edge_counts[dataset_id]:
                    edge_counts[dataset_id][framework]['type_a'] = int(row.get('MajorityBlockedEdges_TypeA', 0))
                    edge_counts[dataset_id][framework]['type_b'] = int(row.get('MajorityBlockedEdges_TypeB', 0))
        return edge_counts, summary_df
    
    def generate_panel_b_strictness(self, force_recompute: bool = False) -> plt.Axes:
        """
        Generate Panel B: Framework blocking comparison showing majority-voted blocked edges.
        Uses summary CSV derived from parsed constraints, focusing exclusively on Type B (normative) bans.
        """
        print("  Building Panel B counts from majority summary CSV...")
        edge_counts, summary_df = self._prepare_majority_edge_counts(force_recompute=force_recompute)
        if summary_df.empty:
            raise ValueError("Framework blocking summary is empty; cannot plot Panel B.")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(self.datasets))
        frameworks_ordered = [fw for fw in self.frameworks if fw != 'Statistical']
        n_frameworks = len(frameworks_ordered)
        width = 0.18 if n_frameworks >= 4 else 0.25
        center_offset = (n_frameworks - 1) / 2
        
        max_bar_height = 0
        for i, framework in enumerate(frameworks_ordered):
            counts_type_b = [edge_counts[ds][framework]['type_b'] for ds in self.datasets]
            if counts_type_b:
                max_bar_height = max(max_bar_height, max(counts_type_b))
            offset = (i - center_offset) * width
            color = self.framework_colors[framework]
            label = framework.replace('Anti', 'Anti-')
            ax.bar(
                x + offset,
                counts_type_b,
                width,
                label=label,
                color=color,
                edgecolor='black',
                linewidth=0.5
            )
        
        # Styling
        ax.set_xlabel('Dataset', fontweight='bold')
        ax.set_ylabel('Blocked Edges (LLM majority vote)', fontweight='bold')
        ax.set_title('Type B Edge Blocks by Fairness Framework', fontsize=12.5, fontweight='bold', pad=26)
        ax.annotate(
            'Statistical baseline has 0 blocked edges',
            xy=(0.5, 1.025),
            xycoords='axes fraction',
            ha='center',
            fontsize=11,
            color='#444444',
            fontstyle='italic'
        )
        ax.set_xticks(x)
        ax.set_xticklabels([self.dataset_labels[ds] for ds in self.datasets], fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        legend_kwargs = dict(ncol=1, loc='upper right', framealpha=0.95,
                  fontsize=7.0, prop={'weight': 'bold'})
        ax.legend(handles, labels, **legend_kwargs)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        y_max = max_bar_height * 1.2 if max_bar_height > 0 else 1.0
        ax.set_ylim(bottom=0, top=y_max)
        
        # Add annotation about vote logic
        plt.tight_layout()
        return ax
    
    def generate_panel_c_edge_types(self, dataset: str = 'law') -> plt.Axes:
        """
        Generate Panel C: Contested edge type breakdown across ALL datasets.
        Categories:
        - Protected Attributes → Outcome (e.g., race→outcome, gender→outcome)
        - Proxies → Outcome (e.g., income→outcome, zip→outcome)
        - Merit Variables → Outcome (e.g., LSAT→outcome)
        - Intermediate Pathways (e.g., race→GPA, income→LSAT)
        """
        print(f"  Generating edge type breakdown for all datasets...")
        
        # Define categories based on common patterns
        protected_attrs = ['race', 'gender', 'sex', 'male']
        merit_vars = ['lsat', 'gpa', 'ugpa', 'test', 'score', 'education', 'educ']
        proxy_vars = ['income', 'zip', 'zipcode', 'neighborhood', 'family_income', 
                     'priors', 'priors_count', 'age']
        
        category_names = [
            'Protected Attributes → Outcome',
            'Proxies → Outcome',
            'Merit Variables → Outcome',
            'Intermediate Pathways'
        ]
        
        # Collect data for all datasets
        all_data = {ds: {cat: 0 for cat in category_names} for ds in self.datasets}
        
        for ds in self.datasets:
            # Load contested zone report
            df = self._load_contested_report(ds)
            df = self._clean_variable_names(df, ds)
            
            # Include ALL contested AND consensus_prohibited edges (including blacklisted_type_b)
            contested_df = df[df['category'].isin(['contested', 'consensus_prohibited'])].copy()
            
            if len(contested_df) == 0:
                print(f"    No contested or consensus_prohibited edges found for {ds}")
                continue
            
            # Get outcome variable
            outcome_var_display = self._format_variable_name(
                self.outcome_variables.get(ds, 'y'),
                ds
            ).lower()
            
            # Categorize edges
            for _, row in contested_df.iterrows():
                source = row['source'].lower()
                target = row['target'].lower()
                
                # Check if target is outcome
                if target == outcome_var_display:
                    # Edges to outcome
                    if any(pa in source for pa in protected_attrs):
                        all_data[ds]['Protected Attributes → Outcome'] += 1
                    elif any(pv in source for pv in proxy_vars):
                        all_data[ds]['Proxies → Outcome'] += 1
                    elif any(mv in source for mv in merit_vars):
                        all_data[ds]['Merit Variables → Outcome'] += 1
                    else:
                        # Default: treat as proxy
                        all_data[ds]['Proxies → Outcome'] += 1
                else:
                    # Intermediate pathways (includes blacklisted_type_b edges)
                    all_data[ds]['Intermediate Pathways'] += 1
        
        # Create grouped horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Prepare data
        n_categories = len(category_names)
        n_datasets = len(self.datasets)
        y_pos = np.arange(n_categories)
        bar_height = 0.2
        
        # Dataset colors
        dataset_colors = {
            'bank': '#8B4789',
            'compas': '#2E86AB',
            'dutch': '#F77F00',
            'law': '#06A77D'
        }
        
        # Plot bars for each dataset
        for i, ds in enumerate(self.datasets):
            counts = [all_data[ds][cat] for cat in category_names]
            offset = (i - 1.5) * bar_height
            
            bars = ax.barh(
                y_pos + offset,
                counts,
                bar_height,
                label=self.dataset_labels[ds],
                color=dataset_colors[ds],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.85
            )
            
            # Add count labels on bars (only if count > 0)
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_width() + 0.1,
                        bar.get_y() + bar.get_height()/2,
                        str(count),
                        va='center',
                        fontsize=7.5,
                        fontweight='bold'
                    )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(category_names, fontweight='bold', fontsize=10.5)
        ax.set_xlabel('Number of Contested Edges', fontweight='bold')
        ax.set_title('Contested Edge Type Breakdown: All Datasets',
                    fontsize=12.5, fontweight='bold')
        ax.legend(ncol=4, loc='upper right', framealpha=0.95, fontsize=8.5,
                  prop={'weight': 'bold'})
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        return ax
    
    def generate_blocking_matrix_heatmap(self, dataset: str) -> plt.Axes:
        """
        Generate blocking matrix heatmap for a dataset.
        Shows EXACTLY which frameworks block which edges.
        Includes both contested and consensus_prohibited edges.
        
        Format:
        Edge                    Stat  Anti-C  Merit  Anti-S
        race → outcome           □      ■       ■       ■
        income → outcome         □      □       ■       ■
        
        Args:
            dataset: Dataset name
            
        Returns:
            matplotlib axes object
        """
        print(f"  Generating blocking matrix for {dataset}...")
        
        # Load contested zone report
        df = self._load_contested_report(dataset)
        df = self._clean_variable_names(df, dataset)
        
        # Show ALL contested edges (not just those to outcome variable)
        # This includes blacklisted_type_b edges between any variables
        contested_edges = df[df['category'].isin(['contested', 'consensus_prohibited'])].copy()
        
        if len(contested_edges) == 0:
            print(f"    No contested or consensus_prohibited edges found for {dataset}")
            return None
        
        # Create matrix: rows = edges, columns = frameworks
        edge_labels = [f"{row['source']} → {row['target']}" for _, row in contested_edges.iterrows()]
        
        # Build blocking matrix
        matrix_data = []
        for _, row in contested_edges.iterrows():
            edge_row = []
            for framework in self.frameworks:
                blocked_col = f'blocked_by_{framework}'
                if blocked_col in row.index:
                    # 1 = blocked (black), 0 = permitted (white)
                    edge_row.append(1 if row[blocked_col] else 0)
                else:
                    edge_row.append(0)
            matrix_data.append(edge_row)
        
        matrix = np.array(matrix_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(4, len(edge_labels) * 0.4)))
        
        # Use binary colormap: white = permitted, black = blocked
        im = ax.imshow(matrix, cmap='Greys', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.frameworks)))
        ax.set_yticks(np.arange(len(edge_labels)))
        
        # Framework labels (shorten for readability)
        framework_labels = [fw.replace('Anti', 'Anti-') for fw in self.frameworks]
        ax.set_xticklabels(framework_labels, fontweight='bold')
        ax.set_yticklabels(edge_labels)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add grid
        ax.set_xticks(np.arange(len(self.frameworks)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(edge_labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
        
        # Add text annotations
        for i in range(len(edge_labels)):
            for j in range(len(self.frameworks)):
                text_color = 'white' if matrix[i, j] == 1 else 'black'
                symbol = '■' if matrix[i, j] == 1 else '□'
                ax.text(j, i, symbol, ha="center", va="center", 
                      color=text_color, fontsize=14.5, fontweight='bold')
        
        ax.set_title(f'Framework Blocking Matrix: {self.dataset_labels[dataset].title()}\n(All Contested & Consensus Prohibited Edges)\n(■ = blocked, □ = permitted)',
                fontsize=12.5, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return ax
    
    def generate_within_framework_agreement_chart(self, dataset: str = 'law') -> plt.Axes:
        """
        Generate chart showing LLM provider agreement within each framework.
        
        Shows how many edges have unanimous (3/3), majority (2/3), or minority (1/3) agreement
        among LLM providers within each framework.
        
        Args:
            dataset: Dataset name to analyze
            
        Returns:
            matplotlib axes object
        """
        print(f"  Generating within-framework agreement analysis for {dataset}...")
        
        # Load constraint sets and create analyzer
        from pathlib import Path
        import sys
        
        # Add parent directory to path for imports
        eval_dir = Path(__file__).parent
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        
        from ConstraintParser import ConstraintParser
        from ContestedZoneAnalyzer import ContestedZoneAnalyzer
        
        parser = ConstraintParser()
        all_constraints = []
        
        # Load constraints from the input directory
        dataset_dir = self._find_latest_dataset_dir(dataset)
        llm_responses_dir = Path("src/faircausal/llm/llm_responses") / dataset
        
        if not llm_responses_dir.exists():
            print(f"    Warning: LLM responses directory not found: {llm_responses_dir}")
            return None
        
        for framework in self.frameworks:
            try:
                constraints = parser.parse_batch(llm_responses_dir, framework, dataset)
                all_constraints.extend(constraints)
            except Exception as e:
                print(f"    Warning: Could not load {framework}: {e}")
        
        if not all_constraints:
            print(f"    Error: No constraint sets loaded")
            return None
        
        # Create analyzer and get agreement data
        analyzer = ContestedZoneAnalyzer(all_constraints)
        agreement_df = analyzer.analyze_within_framework_agreement(dataset, 'fairness')
        
        if len(agreement_df) == 0:
            print(f"    Warning: No agreement data generated")
            return None
        
        # Count agreement levels per framework
        agreement_summary = agreement_df.groupby(['framework', 'agreement_level']).size().unstack(fill_value=0)
        
        # Reorder columns: unanimous, majority, minority
        agreement_levels = ['unanimous', 'majority', 'minority']
        for level in agreement_levels:
            if level not in agreement_summary.columns:
                agreement_summary[level] = 0
        agreement_summary = agreement_summary[agreement_levels]
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for agreement levels
        agreement_colors = {
            'unanimous': '#2E7D32',  # Dark green
            'majority': '#FFA726',   # Orange
            'minority': '#E53935'    # Red
        }
        
        # Plot stacked bars
        x = np.arange(len(self.frameworks))
        width = 0.6
        
        bottom = np.zeros(len(self.frameworks))
        for level in agreement_levels:
            if level in agreement_summary.columns:
                values = [agreement_summary.loc[fw, level] if fw in agreement_summary.index else 0 
                         for fw in self.frameworks]
                ax.bar(
                    x,
                    values,
                    width,
                    label=level.capitalize(),
                    bottom=bottom,
                    color=agreement_colors[level],
                    alpha=0.85
                )
                
                # Add count labels on bars
                for i, (val, bot) in enumerate(zip(values, bottom)):
                    if val > 0:
                        ax.text(
                            i,
                            bot + val/2,
                            str(int(val)),
                            ha='center',
                            va='center',
                            fontweight='bold',
                            fontsize=9.5
                        )
                
                bottom += values
        
        # Styling
        ax.set_xlabel('Fairness Framework', fontweight='bold')
        ax.set_ylabel('Number of Blocked Edges', fontweight='bold')
        ax.set_title(
            f'LLM Provider Agreement Within Frameworks: {self.dataset_labels[dataset].title()}\n(3/3 = Unanimous, 2/3 = Majority, 1/3 = Minority)',
            fontsize=12.5,
            fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels([fw.replace('Anti', 'Anti-') for fw in self.frameworks], rotation=15, ha='right')
        ax.legend(
            title='Agreement Level',
            loc='upper right',
            framealpha=0.95,
            prop={'weight': 'bold'},
            fontsize=8.0
        )
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Add annotation
        ax.text(
            0.02,
            0.98,
            'Agreement = how many of 3 LLM providers blocked the edge',
            transform=ax.transAxes,
            fontsize=8.5,
            style='italic',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
        
        plt.tight_layout()
        return ax
    
    def generate_figure1_combined(self):
        """
        Combine Panel A + Panel B into single two-panel figure.
        Panel A: Color-coded contested edges network (law dataset)
        Panel B: Framework blocking comparison showing Type B-only counts (all datasets)
        """
        print("\nGenerating combined Figure 1...")
        
        fig = plt.figure(figsize=(14, 5.25))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.6, 0.4], 
                             wspace=0.3)
        
        # Panel A: Contested edges network with severity coloring
        ax1 = fig.add_subplot(gs[0])
        print("  Rendering Panel A (Contested Edges Network)...")
        
        df = self._load_contested_report('law')
        df = self._clean_variable_names(df, dataset='law')
        
        # Check if p_blocked column exists, otherwise use n_frameworks_blocking / n_frameworks_total
        if 'p_blocked' not in df.columns:
            if 'n_frameworks_blocking' in df.columns and 'n_frameworks_total' in df.columns:
                df['p_blocked'] = df['n_frameworks_blocking'] / df['n_frameworks_total']
            else:
                df['p_blocked'] = 0.0
        
        # Filter to contested AND consensus_prohibited edges
        contested_df = df[df['category'].isin(['contested', 'consensus_prohibited'])].copy()
        
        # Also get weakly blocked edges (category label: rarely_blocked) to show with faint dotted lines
        rarely_blocked_df = df[df['category'] == 'rarely_blocked'].copy()
        
        # Calculate severity for each edge
        fairness_frameworks = ['AntiClassification', 'Meritocratic', 'AntiSubordination']
        
        def calculate_severity(row):
            count = 0
            for fw in fairness_frameworks:
                blocked_col = f'blocked_by_{fw}'
                if blocked_col in row.index and row[blocked_col] == True:
                    count += 1
            return count
        
        contested_df['severity'] = contested_df.apply(calculate_severity, axis=1)
        
        # Calculate severity for weakly blocked edges too
        if len(rarely_blocked_df) > 0:
            rarely_blocked_df['severity'] = rarely_blocked_df.apply(calculate_severity, axis=1)
        
        # Create graph with contested AND weakly blocked edges
        G = nx.DiGraph()
        for _, row in contested_df.iterrows():
            G.add_edge(row['source'], row['target'], 
                      severity=row['severity'], edge_type='contested')
        
        # Add weakly blocked edges
        for _, row in rarely_blocked_df.iterrows():
            G.add_edge(row['source'], row['target'], 
                      severity=row['severity'], edge_type='rarely_blocked')
        
        # Auto-layout
        raw_outcome = self.outcome_variables.get('law', 'y')
        outcome_label = self._format_variable_name(raw_outcome, 'law')
        pos = self._auto_hierarchical_layout(G, outcome_label)
        
        # Separate edges by severity and type
        # Contested edges (solid lines)
        severity_3_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 3 and d.get('edge_type') == 'contested']
        severity_2_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 2 and d.get('edge_type') == 'contested']
        severity_1_edges = [(u, v) for u, v, d in G.edges(data=True) 
                           if d.get('severity') == 1 and d.get('edge_type') == 'contested']
        
        # Weakly blocked edges (dotted lines)
        rarely_blocked_edges_all = [(u, v) for u, v, d in G.edges(data=True) 
                                    if d.get('edge_type') == 'rarely_blocked']
        
        # Draw edges with severity coloring
        if severity_3_edges:
            nx.draw_networkx_edges(G, pos, edgelist=severity_3_edges,
                                  edge_color='#8B0000', width=3.5, alpha=0.9,
                                  arrows=True, arrowsize=18, arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=30, min_target_margin=30,
                                  ax=ax1)
        if severity_2_edges:
            nx.draw_networkx_edges(G, pos, edgelist=severity_2_edges,
                                  edge_color='#D62728', width=3.0, alpha=0.85,
                                  arrows=True, arrowsize=16, arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=30, min_target_margin=30,
                                  ax=ax1)
        if severity_1_edges:
            nx.draw_networkx_edges(G, pos, edgelist=severity_1_edges,
                                  edge_color='#FF8C00', width=2.5, alpha=0.8,
                                  arrows=True, arrowsize=14, arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=30, min_target_margin=30,
                                  ax=ax1)
        
        if rarely_blocked_edges_all:
            nx.draw_networkx_edges(G, pos, edgelist=rarely_blocked_edges_all,
                                  edge_color=self.rare_edge_color, width=1.5, alpha=0.4,
                                  style='dotted', arrows=True, arrowsize=14,
                                  arrowstyle='-|>', connectionstyle='arc3,rad=0.1',
                                  min_source_margin=30, min_target_margin=30,
                                  ax=ax1)
        
        # Draw nodes
        protected_nodes = [n for n in G.nodes() if self._is_protected_attribute(n)]
        outcome_nodes = [n for n in G.nodes() if n == outcome_label]
        regular_nodes = [n for n in G.nodes() if n not in protected_nodes and n not in outcome_nodes]
        
        predictor_style = self.node_styles['predictor']
        protected_style = self.node_styles['protected']
        outcome_style = self.node_styles['outcome']

        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes,
                              node_color=predictor_style['facecolor'], node_size=2500,
                              edgecolors=predictor_style['edgecolor'], linewidths=2.0, ax=ax1)
        
        if protected_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=protected_nodes,
                                  node_color=protected_style['facecolor'], node_size=2800,
                                  edgecolors=protected_style['edgecolor'], linewidths=2.5, ax=ax1)
        
        nx.draw_networkx_nodes(G, pos, nodelist=outcome_nodes,
                              node_color=outcome_style['facecolor'], node_size=3200,
                              edgecolors=outcome_style['edgecolor'], linewidths=2.0, ax=ax1)
        
        # Draw labels with readable names
        labels = {node: self.create_readable_label(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', ax=ax1)
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor=predictor_style['facecolor'], edgecolor=predictor_style['edgecolor'], linewidth=2,
                  label='Predictor Variables'),
            Patch(facecolor=protected_style['facecolor'], edgecolor=protected_style['edgecolor'], linewidth=2.5,
                  label='Protected Attributes'),
            Patch(facecolor=outcome_style['facecolor'], edgecolor=outcome_style['edgecolor'], linewidth=2,
                  label='Outcome'),
            Line2D([0], [0], color='#8B0000', linewidth=3.5, alpha=0.9,
                  label='Type B ban (3 FWs)'),
            Line2D([0], [0], color='#D62728', linewidth=3.0, alpha=0.85,
                  label='Type B ban (2 FWs)'),
            Line2D([0], [0], color='#FF8C00', linewidth=2.5, alpha=0.8,
                  label='Type B flag (1 FW)'),
            Line2D([0], [0], color=self.rare_edge_color, linewidth=1.5, alpha=0.4,
                  linestyle='dotted', label='Type B provisional\n(weak signal)')
        ]
        
        ax1.legend(handles=legend_elements, loc='lower right', framealpha=0.95,
                   fontsize=6.0, prop={'weight': 'bold'})
        ax1.set_title(
             'A) Type B (Normative) Edge Blocks (Law)',
            fontsize=11.5,
            fontweight='bold',
            loc='left'
        )
        ax1.axis('off')
        
        # Panel B: Blocking comparison bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        print("  Rendering Panel B (Majority-Voted Blocked Edges)...")
        edge_counts, summary_df = self._prepare_majority_edge_counts(force_recompute=True)
        if summary_df.empty:
            print("      WARNING: Framework blocking summary is empty; Panel B will show zeros.")
        
        x = np.arange(len(self.datasets))
        frameworks_ordered = [fw for fw in self.frameworks if fw != 'Statistical']
        n_frameworks = len(frameworks_ordered)
        width = 0.18 if n_frameworks >= 4 else 0.25
        center_offset = (n_frameworks - 1) / 2
        max_bar_height = 0
        for i, framework in enumerate(frameworks_ordered):
            counts_type_b = [edge_counts[ds][framework]['type_b'] for ds in self.datasets]
            if counts_type_b:
                max_bar_height = max(max_bar_height, max(counts_type_b))
            offset = (i - center_offset) * width
            color = self.framework_colors[framework]
            ax2.bar(
                x + offset,
                counts_type_b,
                width,
                label=framework.replace('Anti', 'Anti-'),
                color=color,
                edgecolor='black',
                linewidth=0.5
            )
        
        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_ylabel('Blocked Edges (LLM majority vote)', fontweight='bold')
        ax2.set_title('B) Type B Edge Blocks by Fairness Framework', fontsize=11.5, fontweight='bold', loc='left', pad=26)
        ax2.annotate(
            'Statistical baseline has 0 blocked edges',
            xy=(0.0, 1.025),
            xycoords='axes fraction',
            ha='left',
            fontsize=11,
            color='#444444',
            fontstyle='italic'
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([self.dataset_labels[ds] for ds in self.datasets], fontweight='bold')
        handles, labels = ax2.get_legend_handles_labels()
        legend_kwargs = dict(ncol=1, loc='upper right', framealpha=0.95,
                  fontsize=7.5, prop={'weight': 'bold'})
        ax2.legend(handles, labels, **legend_kwargs)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        y_max = max_bar_height * 1.2 if max_bar_height > 0 else 1.0
        ax2.set_ylim(bottom=0, top=y_max)

        # Create additional headroom so the legend never overlaps the bars
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_path = self.figures_dir / 'figure1_combined.pdf'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_path}")
        
        # Also save as PNG for preview
        png_path = self.figures_dir / 'figure1_combined.png'
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {png_path}")
        
        plt.close()
    
    def generate_table1_summary(self):
        """
        Generate Table 1: Contested zone quantification summary.
        """
        print("\nGenerating Table 1: Contested Zone Summary...")
        
        summary_data = []
        
        for dataset in self.datasets:
            print(f"  Processing {dataset}...")
            df = self._load_contested_report(dataset)
            
            # Ensure p_blocked column exists
            if 'p_blocked' not in df.columns:
                if 'n_frameworks_blocking' in df.columns and 'n_frameworks_total' in df.columns:
                    df['p_blocked'] = df['n_frameworks_blocking'] / df['n_frameworks_total']
                else:
                    df['p_blocked'] = 0.0
            
            # Include ALL edges (not just to outcome) to capture blacklisted_type_b
            all_edges = df.copy()
            
            total_edges = len(all_edges)
            contested_edges = all_edges[all_edges['category'] == 'contested']
            contested_count = len(contested_edges)
            contested_pct = (contested_count / total_edges * 100) if total_edges > 0 else 0
            
            # Get top contested edges (by severity, not just p_blocked)
            # Calculate severity for sorting
            if len(contested_edges) > 0:
                fairness_frameworks = ['AntiClassification', 'Meritocratic', 'AntiSubordination']
                contested_edges = contested_edges.copy()
                
                def calculate_severity(row):
                    count = 0
                    for fw in fairness_frameworks:
                        blocked_col = f'blocked_by_{fw}'
                        if blocked_col in row.index and row[blocked_col] == True:
                            count += 1
                    return count
                
                contested_edges['severity'] = contested_edges.apply(calculate_severity, axis=1)
                top_contested = contested_edges.nlargest(3, 'severity')
            else:
                top_contested = contested_edges
            
            top_edges_str = ', '.join([f"{row['source']}→{row['target']}" 
                                      for _, row in top_contested.iterrows()])
            
            summary_data.append({
                'Dataset': self.dataset_labels.get(dataset, dataset).title(),
                'Total_Edges': total_edges,
                'Contested_Count': contested_count,
                'Contested_Pct': f"{contested_pct:.1f}%",
                'Top_Contested_Edges': top_edges_str
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save CSV
        csv_path = self.tables_dir / 'table1_contested_zone_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Generate LaTeX table
        self._generate_latex_table(summary_df)
        
        return summary_df
    
    def _generate_latex_table(self, df: pd.DataFrame):
        """Generate LaTeX table code from DataFrame."""
        latex_path = self.tables_dir / 'table1_latex.tex'
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Contested Zone Quantification Across Datasets}\n")
            f.write("\\label{tab:contested_zone}\n")
            f.write("\\begin{tabular}{lrrrp{5cm}}\n")
            f.write("\\toprule\n")
            f.write("Dataset & Total Edges & Contested & Contested \\% & Top Contested Edges \\\\\n")
            f.write("\\midrule\n")
            
            for _, row in df.iterrows():
                dataset = row['Dataset']
                total = row['Total_Edges']
                contested = row['Contested_Count']
                pct = row['Contested_Pct']
                # Replace arrow with LaTeX command and escape underscores
                top_edges = row['Top_Contested_Edges'].replace('→', r'$\rightarrow$').replace('_', '\\_')
                
                f.write(f"{dataset} & {total} & {contested} & {pct} & {top_edges} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"  Saved LaTeX: {latex_path}")
    
    def generate_figure2_jaccard_heatmap(self):
        """
        Generate Figure 2: Averaged Jaccard similarity heatmap (Appendix).
        """
        print("\nGenerating Figure 2: Jaccard Similarity Heatmap...")
        
        # Load all framework comparison matrices
        all_matrices = []
        
        for dataset in self.datasets:
            print(f"  Loading {dataset} framework comparison...")
            df = self._load_framework_comparison(dataset)
            
            # Ensure it's a proper matrix (frameworks as both rows and columns)
            if 'Framework' in df.columns or df.columns[0] == 'Unnamed: 0':
                df = df.set_index(df.columns[0])
            
            all_matrices.append(df)
        
        # Average across all datasets
        avg_matrix = sum(all_matrices) / len(all_matrices)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            avg_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            square=True,
            cbar_kws={'label': 'Jaccard Similarity'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Framework Constraint Divergence\n(Averaged Across Datasets)',
                fontsize=12.5, fontweight='bold', pad=20)
        
        # Add annotation below
        fig.text(
            0.5, -0.05,
            "Low similarity between Statistical and fairness frameworks confirms\n"
            "ethical constraints produce substantively different causal hypotheses.",
            ha='center',
            fontsize=9.5,
            style='italic',
            wrap=True
        )
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'figure2_jaccard_heatmap.pdf'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {output_path}")
        
        png_path = self.figures_dir / 'figure2_jaccard_heatmap.png'
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {png_path}")
        
        plt.close()
    
    def validate_strictness_ordering(self, edge_counts: Dict) -> bool:
        """
        Validate that framework strictness ordering holds.
        
        Args:
            edge_counts: Dict[dataset][framework] = count
            
        Returns:
            True if all datasets show Statistical >= AC >= Merit >= AS
        """
        print("\n  Validating strictness ordering (Stat >= AC >= Merit >= AS)...")
        
        all_valid = True
        
        for dataset in self.datasets:
            counts = edge_counts[dataset]

            def total(framework: str) -> int:
                data = counts.get(framework, 0)
                if isinstance(data, dict):
                    return int(data.get('type_a', 0)) + int(data.get('type_b', 0))
                return int(data)

            stat = total('Statistical')
            ac = total('AntiClassification')
            merit = total('Meritocratic')
            asub = total('AntiSubordination')
            
            valid = (stat >= ac >= merit >= asub)
            
            status = "[OK]" if valid else "[X]"
            print(f"    {status} {self.dataset_labels.get(dataset, dataset)}: "
                  f"Stat={stat} >= AC={ac} >= Merit={merit} >= AS={asub}")
            
            if not valid:
                print(f"      WARNING: Ordering violated for {dataset}!")
                all_valid = False
        
        if all_valid:
            print("  [OK] All datasets satisfy strictness ordering!")
        else:
            print("  [X] Some datasets violate expected ordering - check prompt engineering.")
        
        return all_valid
    
    def run_all(self):
        """Execute full visualization pipeline."""
        print("="*80)
        print("GENERATING FACCT PAPER VISUALIZATIONS")
        print("="*80)
        
        print("\n[1/2] Generating Figure 1: Network Visualization...")
        self.generate_figure1_combined()
        
        print("\n[2/2] Generating Individual Panel PDFs...")
        
        # Panel A standalone
        fig_a, ax_a = plt.subplots(figsize=(8, 6))
        self.generate_panel_a_network('law')
        plt.savefig(self.figures_dir / 'figure1_panel_a_network.pdf', 
                   bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'figure1_panel_a_network.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {self.figures_dir / 'figure1_panel_a_network.pdf'}")
        
        # Panel B standalone
        self.generate_panel_b_strictness()
        plt.savefig(self.figures_dir / 'figure1_panel_b_blocking.pdf',
                   bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'figure1_panel_b_blocking.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
        print(f"  Saved: {self.figures_dir / 'figure1_panel_b_blocking.pdf'}")
        
        print("\n" + "="*80)
        print("COMPLETE!")
        print("="*80)
        print(f"\nOutputs saved to:")
        print(f"  Figures: {self.figures_dir}")
        print(f"  Tables: {self.tables_dir}")
        print(f"\nGenerated files:")
        print(f"  Figures:")
        print(f"    - figure1_combined.pdf (two-panel: network + blocking comparison)")
        print(f"    - figure1_panel_a_network.pdf (contested edges network standalone)")
        print(f"    - figure1_panel_b_blocking.pdf (framework blocking comparison standalone)")
        print(f"  Tables:")
        print(f"    - framework_blocking_majority_summary.csv")
        print(f"\nAll figures also saved as PNG for preview.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FAccT paper visualizations"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('results/paper_results/llm_analysis/dataset_framework_analysis'),
        help='Directory with CSV outputs from robustness analysis'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/paper_results/llm_analysis/cross_dataset_analysis'),
        help='Output directory for combined cross-dataset figures and tables'
    )
    
    args = parser.parse_args()
    
    generator = FAccTVisualizationGenerator(args.input_dir, args.output_dir)
    generator.run_all()


if __name__ == "__main__":
    main()
