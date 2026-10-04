"""
Unified pipeline: Generate prompts + Generate LLM responses + Evaluate + Visualize
Combines prompt generation, prompt dispatch, and robustness analysis in a single workflow.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Import the three main processes
from .templates.fairness_prompts.FairnessPromptGenerator import generate_all_frameworks
from .prompt_dispatcher import dispatch_prompts
from .evaluation.run_robustness_analysis import run_full_analysis


def run_fairness_framework_analysis(
    dataset: str,
    frameworks: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    prompts_dir: Optional[Path] = None,
    llm_output_dir: Optional[Path] = None,
    analysis_output_dir: Optional[Path] = None,
    config_file: Optional[str] = None,
    max_prompts: Optional[int] = None,
    skip_prompt_generation: bool = False,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
):
    """
    Run the complete pipeline from prompt generation to evaluation.
    
    Args:
        dataset: Dataset name (e.g., 'bank', 'compas', 'law')
        frameworks: List of frameworks to process (default: all 4)
        providers: List of LLM providers to use (default: all configured)
        prompts_dir: Directory with prompt files
        llm_output_dir: Where to save LLM responses
        analysis_output_dir: Where to save analysis results
        config_file: LLM provider config file
        max_prompts: Limit number of prompts (for testing)
        skip_prompt_generation: Skip prompt generation, use existing prompts
        skip_generation: Skip LLM generation, only run analysis
        skip_evaluation: Only generate LLM responses, skip analysis
    """
    if frameworks is None:
        frameworks = ['Statistical', 'AntiClassification', 'AntiSubordination', 'Meritocratic']
    
    # Set default paths
    if prompts_dir is None:
        prompts_dir = Path(__file__).parent / 'final_prompts'
    if llm_output_dir is None:
        llm_output_dir = Path(__file__).parent / 'llm_responses'
    if analysis_output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_output_dir = Path(__file__).parent.parent.parent.parent / 'results' / 'paper_results' / 'llm_analysis' / 'dataset_framework_analysis' / f"{dataset}_{timestamp}"
    
    print("="*80)
    print("UNIFIED FAIRNESS ANALYSIS PIPELINE")
    print("="*80)
    print(f"\nDataset: {dataset}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Providers: {', '.join(providers) if providers else 'all configured'}")
    print(f"Max prompts per framework: {max_prompts if max_prompts else 'unlimited'}")
    print("="*80)
    
    # ========================================================================
    # STEP 0: Generate Prompts from Dataset Context
    # ========================================================================
    if not skip_prompt_generation:
        print("\n" + "="*80)
        print("STEP 0: GENERATING PROMPTS FROM DATASET CONTEXT")
        print("="*80)
        
        start_time = time.time()
        
        try:
            config_filename = f"DatasetContext_{dataset}.json"
            print(f"\nGenerating prompts for {dataset} across all frameworks...")
            generate_all_frameworks(config_filename)
            
            generation_time = time.time() - start_time
            
            print(f"\n✓ Prompt Generation Complete")
            print(f"  Time elapsed: {generation_time:.1f}s")
            print(f"  Prompts saved to: {prompts_dir}")
                
        except Exception as e:
            print(f"\n✗ Error during prompt generation: {e}")
            import traceback
            traceback.print_exc()
            if not skip_generation:
                print("  Continuing with existing prompts...")
            else:
                raise
    else:
        print("\n[SKIPPED] Prompt Generation")
        print(f"  Using existing prompts from: {prompts_dir}")
    
    # ========================================================================
    # STEP 1: Generate LLM Responses
    # ========================================================================
    if not skip_generation:
        print("\n" + "="*80)
        print("STEP 1: GENERATING LLM RESPONSES")
        print("="*80)
        
        start_time = time.time()
        
        try:
            results = dispatch_prompts(
                prompts_dir=prompts_dir,
                output_dir=llm_output_dir,
                datasets=[dataset],
                frameworks=frameworks,
                providers=providers,
                config_file=config_file,
                max_prompts=max_prompts,
            )
            
            generation_time = time.time() - start_time
            
            print("\n✓ LLM Response Generation Complete")
            print(f"  Time elapsed: {generation_time:.1f}s")
            print(f"  Responses saved to: {llm_output_dir}")
            for provider, files in results.items():
                print(f"    - {provider}: {len(files)} responses")
                
        except Exception as e:
            print(f"\n✗ Error during LLM generation: {e}")
            if not skip_evaluation:
                print("  Continuing to evaluation with existing responses...")
            else:
                raise
    else:
        print("\n[SKIPPED] LLM Response Generation")
        print(f"  Using existing responses from: {llm_output_dir}")
    
    # ========================================================================
    # STEP 2: Contested Zone Analysis
    # ========================================================================
    if not skip_evaluation:
        print("\n" + "="*80)
        print("STEP 2: CONTESTED ZONE ANALYSIS")
        print("="*80)
        
        start_time = time.time()
        
        try:
            run_full_analysis(
                llm_outputs_dir=llm_output_dir,
                output_dir=analysis_output_dir,
                dataset=dataset,
                frameworks=frameworks,
            )
            
            analysis_time = time.time() - start_time
            
            print(f"\n✓ Analysis Complete")
            print(f"  Time elapsed: {analysis_time:.1f}s")
            
        except Exception as e:
            print(f"\n✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("\n[SKIPPED] Contested Zone Analysis")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    if not skip_prompt_generation:
        print(f"\nPrompts: {prompts_dir}")
    if not skip_generation:
        print(f"LLM Responses: {llm_output_dir / dataset}")
    if not skip_evaluation:
        print(f"Analysis Results: {analysis_output_dir}")
        print("\nGenerated artifacts:")
        print(f"  - Parsed constraints (JSON)")
        print(f"  - Contested zone report (CSV)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Fairness Framework Analysis: Generate LLM responses across multiple fairness frameworks, evaluate, and visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with prompt generation
  poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset compas
  
  # Test run (1 prompt per framework)
  poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset bank --max-prompts 1
  
  # Skip prompt generation, use existing prompts
  poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset bank --skip-prompt-generation
  
  # Only run analysis on existing responses
  poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset bank --skip-prompt-generation --skip-generation
  
  # Generate with specific providers
  poetry run python -m src.faircausal.llm.fairness_framework_analysis --dataset bank --providers openai anthropic
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['bank', 'compas', 'law', 'dutch'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--frameworks',
        nargs='*',
        default=None,
        help='Frameworks to process (default: all 4)'
    )
    
    parser.add_argument(
        '--providers',
        nargs='*',
        choices=['openai', 'anthropic', 'google'],
        help='LLM providers to use (default: all configured)'
    )
    
    parser.add_argument(
        '--prompts-dir',
        type=Path,
        help='Directory with prompt files (default: src/faircausal/llm/final_prompts)'
    )
    
    parser.add_argument(
        '--llm-output-dir',
        type=Path,
        help='Where to save LLM responses (default: src/faircausal/llm/llm_responses)'
    )
    
    parser.add_argument(
        '--analysis-output-dir',
        type=Path,
        help='Where to save analysis results (default: results/paper_results/llm_analysis/dataset_framework_analysis/<dataset>_<timestamp>)'
    )
    
    parser.add_argument(
        '--config-file',
        help='LLM provider config file'
    )
    
    parser.add_argument(
        '--max-prompts',
        type=int,
        help='Limit number of prompts processed (for testing)'
    )
    
    parser.add_argument(
        '--skip-prompt-generation',
        action='store_true',
        help='Skip prompt generation, use existing prompts'
    )
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip LLM generation, only run analysis on existing responses'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Only generate LLM responses, skip analysis'
    )
    
    args = parser.parse_args()
    
    run_fairness_framework_analysis(
        dataset=args.dataset,
        frameworks=args.frameworks,
        providers=args.providers,
        prompts_dir=args.prompts_dir,
        llm_output_dir=args.llm_output_dir,
        analysis_output_dir=args.analysis_output_dir,
        config_file=args.config_file,
        max_prompts=args.max_prompts,
        skip_prompt_generation=args.skip_prompt_generation,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
    )


if __name__ == "__main__":
    main()
