"""
Generate visualizations for framework robustness evaluation.
Creates consensus spectrum plots, disagreement matrices, and stability heatmaps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import warnings


class VisualizationGenerator:
    """
    Generate publication-quality visualizations for robustness analysis.
    """
    
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory to save generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality defaults
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_consensus_spectrum(
        self,
        report_df: pd.DataFrame,
        title: str = "Consensus Spectrum",
        filename: str = "consensus_spectrum.pdf",
        top_n_contested: int = 5
    ):
        """
        Create horizontal bar chart showing consensus spectrum.
        
        Args:
            report_df: DataFrame from ContestedZoneAnalyzer.create_contested_zone_report()
            title: Plot title
            filename: Output filename
            top_n_contested: Number of top contested edges to show in inset
        """
        # Sort by blocking probability
        df_sorted = report_df.sort_values('p_blocked', ascending=True)
        
        # Color mapping
        colors = []
        for _, row in df_sorted.iterrows():
            if row['category'] == 'consensus_prohibited':
                colors.append('#d62728')  # Red
            elif row['category'] == 'contested':
                colors.append('#ff7f0e')  # Orange
            else:  # consensus_permitted
                colors.append('#2ca02c')  # Green
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(df_sorted) * 0.2)))
        
        # Main plot
        y_pos = np.arange(len(df_sorted))
        ax.barh(y_pos, df_sorted['p_blocked'], color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['edge'], fontsize=8)
        ax.set_xlabel('Fraction of Frameworks Blocking Edge', fontsize=11)
        ax.set_xlim(0, 1)
        ax.axvline(0.75, color='red', linestyle='--', alpha=0.5, linewidth=1, 
                   label='Consensus Prohibited (≥0.75)')
        ax.axvline(0.25, color='green', linestyle='--', alpha=0.5, linewidth=1,
                   label='Consensus Permitted (≤0.25)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        # Add category counts
        n_prohibited = len(report_df[report_df['category'] == 'consensus_prohibited'])
        n_contested = len(report_df[report_df['category'] == 'contested'])
        n_permitted = len(report_df[report_df['category'] == 'consensus_permitted'])
        
        text = f"Prohibited: {n_prohibited} | Contested: {n_contested} | Permitted: {n_permitted}"
        fig.text(0.5, 0.02, text, ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved consensus spectrum to {output_path}")
    
    def plot_disagreement_matrix(
        self,
        report_df: pd.DataFrame,
        frameworks: List[str],
        filename: str = "disagreement_matrix.pdf",
        max_edges: int = 50
    ):
        """
        Create heatmap showing which frameworks block which edges.
        
        Args:
            report_df: DataFrame from ContestedZoneAnalyzer.create_contested_zone_report()
            frameworks: List of framework names
            filename: Output filename
            max_edges: Maximum number of edges to display
        """
        # Filter to contested edges only
        contested_df = report_df[report_df['category'] == 'contested'].copy()
        
        if len(contested_df) == 0:
            print("  No contested edges to plot")
            return
        
        # Limit to top N for readability
        if len(contested_df) > max_edges:
            contested_df = contested_df.nlargest(max_edges, 'p_blocked')
        
        # Sort by blocking probability
        contested_df = contested_df.sort_values('p_blocked', ascending=False)
        
        # Create matrix: rows = edges, cols = frameworks
        matrix_data = []
        edge_labels = []
        
        for _, row in contested_df.iterrows():
            edge_labels.append(row['edge'])
            row_data = []
            for fw in frameworks:
                col_name = f'blocked_by_{fw}'
                if col_name in row:
                    row_data.append(1 if row[col_name] else 0)
                else:
                    row_data.append(np.nan)
            matrix_data.append(row_data)
        
        matrix = np.array(matrix_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(6, len(edge_labels) * 0.3)))
        
        sns.heatmap(
            matrix,
            xticklabels=frameworks,
            yticklabels=edge_labels,
            cmap='RdYlGn_r',  # Red = blocked, Green = allowed
            cbar_kws={'label': 'Blocked'},
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )
        
        ax.set_title('Framework Disagreement Matrix (Contested Edges Only)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Framework', fontsize=11)
        ax.set_ylabel('Edge', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(fontsize=8)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved disagreement matrix to {output_path}")
    
    def plot_stability_heatmap(
        self,
        comparison_matrix: pd.DataFrame,
        filename: str = "stability_heatmap.pdf",
        title: str = "Framework Pairwise Jaccard Similarity"
    ):
        """
        Create heatmap of pairwise Jaccard similarities between frameworks.
        
        Args:
            comparison_matrix: DataFrame from StabilityAnalyzer.all_frameworks_comparison()
            filename: Output filename
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Create heatmap with annotations
        sns.heatmap(
            comparison_matrix,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            square=True,
            linewidths=0.5,
            linecolor='gray',
            cbar_kws={'label': 'Jaccard Similarity'},
            ax=ax
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Framework', fontsize=11)
        ax.set_ylabel('Framework', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved stability heatmap to {output_path}")
    
    def plot_stability_trends(
        self,
        stability_df: pd.DataFrame,
        filename: str = "stability_trends.pdf"
    ):
        """
        Create bar plot comparing stability metrics across frameworks.
        
        Args:
            stability_df: DataFrame with stability metrics (from StabilityAnalyzer)
            filename: Output filename
        """
        if 'mean_jaccard' not in stability_df.columns:
            print("  No stability metrics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Mean Jaccard similarity
        ax1 = axes[0]
        frameworks = stability_df.index
        jaccard_means = stability_df['mean_jaccard']
        jaccard_stds = stability_df['std_jaccard']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(frameworks)))
        ax1.bar(frameworks, jaccard_means, yerr=jaccard_stds, 
               capsize=5, alpha=0.8, color=colors)
        ax1.set_ylabel('Mean Jaccard Similarity', fontsize=11)
        ax1.set_xlabel('Framework', fontsize=11)
        ax1.set_title('Within-Framework Stability', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 2: Agreement rate
        if 'agreement_rate' in stability_df.columns:
            ax2 = axes[1]
            agreement_rates = stability_df['agreement_rate']
            
            ax2.bar(frameworks, agreement_rates, alpha=0.8, color=colors)
            ax2.set_ylabel('Agreement Rate', fontsize=11)
            ax2.set_xlabel('Framework', fontsize=11)
            ax2.set_title('Within-Framework Agreement', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved stability trends to {output_path}")
    
    def plot_framework_comparison(
        self,
        unique_edges: Dict[str, set],
        filename: str = "framework_unique_edges.pdf"
    ):
        """
        Create Venn-like diagram showing unique constraints per framework.
        
        Args:
            unique_edges: Dict from ContestedZoneAnalyzer.get_framework_specific_edges()
            filename: Output filename
        """
        frameworks = list(unique_edges.keys())
        counts = [len(edges) for edges in unique_edges.values()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(frameworks)))
        bars = ax.bar(frameworks, counts, alpha=0.8, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Number of Unique Edges', fontsize=11)
        ax.set_xlabel('Framework', fontsize=11)
        ax.set_title('Framework-Specific Constraints', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved framework comparison to {output_path}")


# Example usage
if __name__ == "__main__":
    from .ConstraintParser import ConstraintParser
    from .ContestedZoneAnalyzer import ContestedZoneAnalyzer
    from .StabilityAnalyzer import StabilityAnalyzer
    from pathlib import Path
    
    # Parse constraints
    parser = ConstraintParser()
    results_dir = Path("src/faircausal/llm/llm_responses/bank/")
    output_dir = Path("results/visualizations/")
    
    all_constraints = []
    for framework in ['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic']:
        try:
            constraints = parser.parse_batch(results_dir, framework, 'bank')
            all_constraints.extend(constraints)
        except Exception as e:
            print(f"Could not load {framework}: {e}")
    
    if all_constraints:
        # Generate visualizations
        viz_gen = VisualizationGenerator(output_dir)
        
        # Contested zone analysis
        contested_analyzer = ContestedZoneAnalyzer(all_constraints)
        report = contested_analyzer.create_contested_zone_report('bank')
        
        viz_gen.plot_consensus_spectrum(
            report,
            title="Consensus Spectrum: Bank Marketing Dataset"
        )
        
        viz_gen.plot_disagreement_matrix(
            report,
            frameworks=['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic']
        )
