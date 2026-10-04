"""
Main script to run complete robustness analysis.
Orchestrates parsing, analysis, visualization, and downstream evaluation.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

from src.faircausal.llm.evaluation.ConstraintParser import ConstraintParser
from src.faircausal.llm.evaluation.ContestedZoneAnalyzer import ContestedZoneAnalyzer


def run_full_analysis(
    llm_outputs_dir: Path,
    output_dir: Path,
    dataset: str = 'bank',
    frameworks: list = None
):
    """
    Run complete robustness analysis pipeline for LLM responses.
    
    Args:
        llm_outputs_dir: Directory with LLM constraint responses (JSON files)
        output_dir: Where to save results
        dataset: Which dataset to analyze
        frameworks: List of frameworks (default: all 4)
    """
    if frameworks is None:
        frameworks = ['Statistical', 'AntiClassification', 
                     'AntiSubordination', 'Meritocratic']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"LLM RESPONSE ANALYSIS: {dataset.upper()}")
    print("="*80)
    
    # Step 1: Parse LLM outputs
    print("\n[1/2] Parsing LLM constraint outputs...")
    parser = ConstraintParser()
    all_constraints = []
    
    # Look for JSON files in the dataset subdirectory
    dataset_llm_dir = llm_outputs_dir / dataset
    if not dataset_llm_dir.exists():
        dataset_llm_dir = llm_outputs_dir
    
    for framework in frameworks:
        try:
            constraints = parser.parse_batch(dataset_llm_dir, framework, dataset)
            all_constraints.extend(constraints)
            print(f"  ✓ Parsed {len(constraints)} runs for {framework}")
        except Exception as e:
            print(f"  ✗ Error parsing {framework}: {e}")
    
    if not all_constraints:
        raise ValueError("No constraint sets were successfully parsed!")
    
    # Save parsed constraints
    parsed_file = output_dir / f"{dataset}_parsed_constraints.json"
    with open(parsed_file, 'w') as f:
        json.dump([cs.to_dict() for cs in all_constraints], f, indent=2)
    print(f"  Saved parsed constraints to {parsed_file}")
    
    # Step 2: Contested Zone Analysis
    print("\n[2/2] Identifying contested zone...")
    contested_analyzer = ContestedZoneAnalyzer(all_constraints)
    
    # Analyze ONLY fairness constraints (Type B - Direct Discrimination)
    # This excludes Type A (theoretical impossibilities like temporal violations)
    categories = contested_analyzer.categorize_edges(dataset, 'fairness')
    print(f"  Consensus Prohibited (fairness): {len(categories['consensus_prohibited'])} edges")
    print(f"  Rarely Blocked (fairness): {len(categories['rarely_blocked'])} edges")
    print(f"  Contested (fairness disagreement): {len(categories['contested'])} edges")
    
    # Detailed report - ONLY fairness-based contested edges (Type B)
    report = contested_analyzer.create_contested_zone_report(dataset, 'fairness')
    report_file = output_dir / f"{dataset}_contested_zone_report.csv"
    report.to_csv(report_file, index=False)
    print(f"  Saved contested zone report (fairness only) to {report_file}")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nKey files:")
    print(f"  - Parsed constraints: {parsed_file.name}")
    print(f"  - Contested zone: {report_file.name}")
    

def main():
    parser = argparse.ArgumentParser(
        description="Run robustness analysis for LLM-generated fairness constraints"
    )
    parser.add_argument(
        '--llm-outputs',
        type=Path,
        default=Path('src/faircausal/llm/llm_responses'),
        help='Directory containing LLM constraint outputs'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/robustness_analysis'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='bank',
        choices=['bank', 'compas', 'law', 'dutch'],
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--frameworks',
        nargs='+',
        default=['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic'],
        help='Frameworks to analyze'
    )
    
    args = parser.parse_args()
    
    run_full_analysis(
        llm_outputs_dir=args.llm_outputs,
        output_dir=args.output_dir,
        dataset=args.dataset,
        frameworks=args.frameworks
    )


if __name__ == "__main__":
    main()
