# scr/faircausal/core/counterfactual_fairness_runner.py
"""
Counterfactual Fairness Analysis Module

This module provides functions for analyzing counterfactual fairness
across multiple DAGs derived from a CPDAG structure.

Updated version with improved flow: SCM Fitting -> Synthetic Data Generation -> Fairness Evaluation
"""

import os
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pickle
import itertools
import networkx as nx
import traceback
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

import dowhy.gcm as gcm
from dowhy.gcm.ml import create_random_forest_classifier 

from src.faircausal.core.causal_data_utils import get_data_for_algorithm
from src.faircausal.core.fairness_metrics import calculate_individual_fairness
from src.faircausal.core.ml_classifer_runner import preprocess_data, run_samples

def _wrapper_lambda_fn(val):
    """Helper function for counterfactual fairness analysis"""
    return lambda x: val

def validate_and_add_missing_columns(samples_df, required_cols, fallback_source_df, source_index, context="samples"):
    """
    Utility function to validate required columns exist in samples and add missing ones.
    
    Args:
        samples_df: DataFrame containing the samples 
        required_cols: List of column names that are required
        fallback_source_df: DataFrame to get fallback values from (original data)
        source_index: Index to use for fallback values (can be integer or actual index)
        context: String describing the context for logging
        
    Returns:
        tuple: (samples_df with missing columns added, list of missing columns that were added)
    """
    available_cols = list(samples_df.columns)
    missing_cols = [col for col in required_cols if col not in available_cols]
    
    if missing_cols:
        print(f"Warning: Missing columns in {context}: {missing_cols}")
        print(f"  Available columns: {available_cols}")
        print(f"  Required columns: {required_cols}")
        
        for missing_col in missing_cols:
            fallback_value = None
            
            # Try multiple sources for fallback value
            if fallback_source_df is not None and missing_col in fallback_source_df.columns:
                try:
                    # Handle different ways to access the fallback value
                    if isinstance(fallback_source_df, pd.DataFrame):
                        if len(fallback_source_df) == 1:
                            # Single row DataFrame - use iloc[0]
                            fallback_value = fallback_source_df[missing_col].iloc[0]
                        elif source_index in fallback_source_df.index:
                            # Multi-row DataFrame with specific index
                            fallback_value = fallback_source_df.loc[source_index, missing_col]
                        else:
                            # Fallback to first row if index not found
                            fallback_value = fallback_source_df[missing_col].iloc[0]
                    else:
                        # Not a DataFrame, try direct access
                        fallback_value = fallback_source_df[missing_col]
                    
                except Exception as e:
                    print(f"  ❌ Failed to extract fallback value for '{missing_col}': {str(e)}")
                    fallback_value = None
            
            # If we still don't have a fallback value, try some common defaults
            if fallback_value is None:
                if missing_col in ['age_binary', 'decile_score_cat']:
                    # These might be derived features that should default to 0
                    fallback_value = 0
                    print(f"  ⚠️  Using common default value 0 for '{missing_col}'")
                else:
                    # Generic default
                    fallback_value = 0
                    print(f"  ⚠️  Using generic default value 0 for '{missing_col}'")
            
            # Add the missing column to samples_df
            # Fix: Use scalar value repeated for all rows, not a single-element list
            samples_df[missing_col] = fallback_value
            print(f"  ✅ Added missing column '{missing_col}' with value: {fallback_value}")
    
    return samples_df, missing_cols

def calculate_natural_indirect_effect(individual_predictions, classifier, protected_attr, 
                                    causal_model, categorical_info, X_cols, mediator_vars=None,
                                    dag_edges=None, target_var=None):
    """
    Calculate Natural Indirect Effect (NIE) for counterfactual fairness.
    
    NIE measures the indirect discrimination that flows through mediator variables
    when the protected attribute changes from one value to another.
    
    Args:
        individual_predictions: Individual prediction data structure
        classifier: Trained ML classifier
        protected_attr: Name of the protected attribute
        causal_model: Fitted structural causal model
        categorical_info: Categorical variable mapping information
        X_cols: Feature column names for the classifier
        mediator_vars: List of mediator variable names (if None, infers from DAG)
        dag_edges: List of DAG edges to infer mediators
    
    Returns:
        dict: NIE results including indirect effects and confidence intervals
    """
    try:
        print(f"Calculating Natural Indirect Effect (NIE) for {protected_attr}...")
        
        # Infer mediator variables if not provided
        if mediator_vars is None and dag_edges is not None:
            mediator_vars = infer_mediator_variables(protected_attr, dag_edges, X_cols, target_var)
            print(f"Inferred mediator variables: {mediator_vars}")
        elif mediator_vars is None:
            print("Warning: No mediator variables specified and no DAG edges provided")
            return {'nie_score': np.nan, 'error': 'No mediator variables specified'}
        
        nie_results = []
        
        # Process each individual
        for individual in individual_predictions:
            if protected_attr not in individual['predictions_by_attr']:
                continue
                
            attr_predictions = individual['predictions_by_attr'][protected_attr]
            
            # Get factual and counterfactual values
            factual_value = individual['factual_values'][protected_attr]
            
            nie_individual = {}
            
            # For each counterfactual value
            for cf_key, cf_features in attr_predictions.items():
                if cf_key.startswith('cf_'):
                    cf_value = cf_key.replace('cf_', '')
                    
                    # Calculate NIE using three-step decomposition
                    nie_score = calculate_nie_three_step(
                        individual, protected_attr, factual_value, cf_value,
                        causal_model, categorical_info, X_cols, mediator_vars, classifier
                    )
                    
                    nie_individual[f'nie_{factual_value}_to_{cf_value}'] = nie_score
            
            if nie_individual:
                nie_results.append(nie_individual)
        
        # Aggregate NIE results
        if nie_results:
            aggregated_nie = aggregate_nie_results(nie_results)
            print(f"✅ NIE calculated for {len(nie_results)} individuals")
            return aggregated_nie
        else:
            return {'nie_score': np.nan, 'error': 'No NIE calculations completed'}
            
    except Exception as e:
        print(f"Error calculating NIE: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'nie_score': np.nan, 'error': str(e)}


def calculate_path_specific_effect(individual_predictions, classifier, protected_attr,
                                 causal_model, categorical_info, X_cols, 
                                 specific_paths=None, dag_edges=None, target_var=None,
                                 pse_sample_size=None, max_paths=None, random_state=42,
                                 return_individual_scores=False):
    """
    IMPROVED: Calculate Path-Specific Effect with better path handling
    """
    try:
        print(f"Calculating Path-Specific Effect (PSE) for {protected_attr}...")
        
        # Identify causal paths if not provided
        if specific_paths is None and dag_edges is not None:
            if target_var is None:
                # Try to infer target from context
                print("Warning: target_var not provided for PSE calculation, paths may include target variable as mediator")
                target_var = None  # Let identify_causal_paths handle this case
            
            specific_paths = identify_causal_paths(
                protected_attr, dag_edges, X_cols, target_var, max_paths=max_paths
            )
            print(f"Identified causal paths: {specific_paths}")
        elif specific_paths is None:
            print("Warning: No specific paths specified and no DAG edges provided")
            return {'pse_score': np.nan, 'error': 'No causal paths specified'}
        
        if max_paths is not None and max_paths > 0 and specific_paths:
            specific_paths = specific_paths[:max_paths]

        if not specific_paths:
            return {'pse_score': np.nan, 'error': 'No valid causal paths found'}

        # Optional subsampling for faster PSE without affecting other fairness metrics.
        evaluation_individuals = individual_predictions
        if pse_sample_size is not None and pse_sample_size > 0 and len(individual_predictions) > pse_sample_size:
            rng = np.random.default_rng(random_state)
            sample_indices = rng.choice(len(individual_predictions), size=pse_sample_size, replace=False)
            evaluation_individuals = [individual_predictions[i] for i in sample_indices]
            print(f"Subsampled PSE evaluation from {len(individual_predictions)} to {len(evaluation_individuals)} individuals")
        
        pse_results = {}
        path_items = []
        pse_path_results_by_path = {}
        
        # Initialize path containers once; individual loop below reuses cached predictions.
        for path_idx, path in enumerate(specific_paths):
            path_name = f"path_{path_idx}_{'-'.join(path)}"
            print(f"  Calculating PSE for path: {path}")
            path_items.append((path_name, path))
            pse_path_results_by_path[path_name] = []

        # Build batch for factual and total counterfactual predictions first.
        baseline_feature_rows = []
        prepared_individuals = []

        for individual in evaluation_individuals:
            if protected_attr not in individual['predictions_by_attr']:
                continue

            attr_predictions = individual['predictions_by_attr'][protected_attr]
            factual_value = individual['factual_values'][protected_attr]

            # Shared observed_data preparation for this individual.
            individual_data_str = pd.DataFrame([individual['factual_values']])
            scm_nodes = set(causal_model.graph.nodes())
            missing_scm_cols = scm_nodes - set(individual_data_str.columns)
            for col in missing_scm_cols:
                individual_data_str[col] = 0

            for col in scm_nodes:
                if col in individual_data_str.columns:
                    if col in categorical_info:
                        individual_data_str[col] = individual_data_str[col].map(categorical_info[col]['int_to_string'])
                    else:
                        individual_data_str[col] = individual_data_str[col].astype(str)

            entry = {
                'individual': individual,
                'factual_value': factual_value,
                'attr_predictions': attr_predictions,
                'individual_data_str': individual_data_str,
                'factual_row_idx': None,
                'cf_row_idx_by_value': {},
                'y_factual': None,
                'y_total_cf_by_value': {},
                'per_path_individual_scores': {path_name: {} for path_name, _ in path_items}
            }

            entry['factual_row_idx'] = len(baseline_feature_rows)
            baseline_feature_rows.append(attr_predictions['factual'])

            for cf_key, cf_features in attr_predictions.items():
                if cf_key.startswith('cf_'):
                    cf_value = cf_key.replace('cf_', '')
                    entry['cf_row_idx_by_value'][cf_value] = len(baseline_feature_rows)
                    baseline_feature_rows.append(cf_features)

            prepared_individuals.append(entry)

        if not prepared_individuals:
            return {'pse_score': np.nan, 'error': 'No valid individuals for PSE calculation'}

        # Batch-score factual and total counterfactual rows in one classifier call.
        baseline_features_matrix = pd.concat(baseline_feature_rows, ignore_index=True)
        baseline_scores = classifier.predict_proba(baseline_features_matrix.astype(float))[:, 1]

        for entry in prepared_individuals:
            entry['y_factual'] = baseline_scores[entry['factual_row_idx']]
            for cf_value, cf_row_idx in entry['cf_row_idx_by_value'].items():
                entry['y_total_cf_by_value'][cf_value] = baseline_scores[cf_row_idx]

        # Build path-blocked feature batch and score once.
        path_blocked_feature_rows = []
        path_blocked_meta = []

        task_specs = []
        for entry in prepared_individuals:
            individual = entry['individual']
            factual_value = entry['factual_value']
            y_factual = entry['y_factual']

            for cf_value, y_total_cf in entry['y_total_cf_by_value'].items():
                for path_name, path in path_items:
                    task_specs.append((
                        entry,
                        individual,
                        factual_value,
                        cf_value,
                        y_factual,
                        y_total_cf,
                        path_name,
                        path
                    ))

        # Use threaded generation for expensive SCM path-blocking calls.
        max_workers = min(8, os.cpu_count() or 1)

        def _build_pse_feature_payload(task_spec):
            (entry, individual, factual_value, cf_value, y_factual, y_total_cf, path_name, path) = task_spec
            pse_feature_payload = calculate_pse_path_blocking(
                individual, protected_attr, factual_value, cf_value,
                path, causal_model, categorical_info, X_cols, classifier,
                y_factual=y_factual,
                y_total_cf=y_total_cf,
                individual_data_str=entry['individual_data_str'],
                return_features_only=True
            )

            if isinstance(pse_feature_payload, dict) and 'path_blocked_features' in pse_feature_payload:
                return {
                    'feature_row': pse_feature_payload['path_blocked_features'],
                    'meta': {
                        'entry': entry,
                        'path_name': path_name,
                        'comparison_key': f"pse_{factual_value}_to_{cf_value}",
                        'payload': pse_feature_payload
                    }
                }

            return None

        threaded_generation_failed = False
        if task_specs and max_workers > 1:
            try:
                print(f"Generating path-blocked SCM samples with {max_workers} workers...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(_build_pse_feature_payload, task_spec) for task_spec in task_specs]
                    # Preserve deterministic ordering by reading futures in submission order.
                    for future in futures:
                        result = future.result()
                        if result is not None:
                            path_blocked_feature_rows.append(result['feature_row'])
                            path_blocked_meta.append(result['meta'])
            except Exception as e:
                threaded_generation_failed = True
                print(f"Warning: threaded PSE path-blocking generation failed ({str(e)}), falling back to serial execution")
        else:
            threaded_generation_failed = True

        if threaded_generation_failed:
            path_blocked_feature_rows = []
            path_blocked_meta = []
            for task_spec in task_specs:
                result = _build_pse_feature_payload(task_spec)
                if result is not None:
                    path_blocked_feature_rows.append(result['feature_row'])
                    path_blocked_meta.append(result['meta'])

        if path_blocked_feature_rows:
            path_blocked_matrix = pd.concat(path_blocked_feature_rows, ignore_index=True)
            path_blocked_scores = classifier.predict_proba(path_blocked_matrix.astype(float))[:, 1]

            for meta, y_path_blocked in zip(path_blocked_meta, path_blocked_scores):
                payload = meta['payload']
                total_effect = payload['total_effect']
                y_factual = payload['y_factual']
                effect_without_path = y_path_blocked - y_factual
                path_specific_effect = total_effect - effect_without_path

                path_vars = payload.get('blocked_path', [])
                if abs(path_specific_effect) > 1.0:
                    print(f"Warning: Large PSE value ({path_specific_effect:.4f}) for path {path_vars}")

                pse_score = {
                    'pse': path_specific_effect,
                    'total_effect': total_effect,
                    'effect_without_path': effect_without_path,
                    'y_factual': y_factual,
                    'y_counterfactual': payload['y_counterfactual'],
                    'y_path_blocked': y_path_blocked,
                    'blocked_path': path_vars,
                    'validation_passed': abs(path_specific_effect) <= 1.0
                }

                entry = meta['entry']
                entry['per_path_individual_scores'][meta['path_name']][meta['comparison_key']] = pse_score

        for entry in prepared_individuals:
            for path_name, path_scores in entry['per_path_individual_scores'].items():
                if path_scores:
                    pse_path_results_by_path[path_name].append(path_scores)
            
        # Aggregate PSE results for each path
        for path_name, path in path_items:
            path_results = pse_path_results_by_path[path_name]
            if path_results:
                aggregated_pse = aggregate_pse_results(path_results)
                aggregated_pse['path_variables'] = path  # Store path info
                pse_results[path_name] = aggregated_pse
            else:
                pse_results[path_name] = {
                    'pse_score': np.nan, 
                    'error': 'No PSE calculations for this path',
                    'path_variables': path
                }
        
        print(f"✅ PSE calculated for {len(specific_paths)} paths")

        if return_individual_scores:
            individual_scores = []
            for entry in prepared_individuals:
                individual_scores.append({
                    'original_index': entry['individual']['original_index'],
                    'path_scores_by_path': entry['per_path_individual_scores']
                })

            return {
                'aggregated_results': pse_results,
                'individual_scores': individual_scores
            }

        return pse_results
        
    except Exception as e:
        print(f"Error calculating PSE: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'pse_score': np.nan, 'error': str(e)}


def infer_mediator_variables(protected_attr, dag_edges, feature_cols, target_var=None):
    """
    IMPROVED: Better mediator inference with path analysis
    """
    graph = nx.DiGraph(dag_edges)
    
    if protected_attr not in graph.nodes():
        return []
    
    # Find all descendants of protected attribute
    descendants = nx.descendants(graph, protected_attr)
    
    # Filter to only include feature columns
    potential_mediators = [var for var in descendants if var in feature_cols]
    
    # Direct parents of the target are valid mediators for A -> M -> Y paths.
    # Exclude only the target variable itself if it appears in feature columns.
    if target_var:
        potential_mediators = [var for var in potential_mediators if var != target_var]
    
    return potential_mediators


def identify_causal_paths(protected_attr, dag_edges, feature_cols, target_var=None, max_path_length=4, max_paths=None):
    """
    IMPROVED: Better path identification for PSE analysis
    """
    graph = nx.DiGraph(dag_edges)
    paths = []
    
    if protected_attr not in graph.nodes():
        return paths
    
    # If target_var is provided, find paths specifically to target
    if target_var and target_var in graph.nodes():
        try:
            # Find all simple paths from protected_attr to target_var
            all_paths = list(nx.all_simple_paths(graph, protected_attr, target_var, cutoff=max_path_length))
            
            for path in all_paths:
                if len(path) > 2:  # Must have intermediate nodes
                    # Extract intermediate nodes (exclude source and target)
                    intermediate_nodes = path[1:-1]
                    
                    # Only include paths with nodes that are in feature_cols
                    valid_intermediates = [node for node in intermediate_nodes if node in feature_cols]
                    
                    if valid_intermediates:
                        paths.append(valid_intermediates)
            
        except nx.NetworkXNoPath:
            print(f"No paths found from {protected_attr} to {target_var}")
            return []
    else:
        # Original logic for backward compatibility
        # When no target_var is specified, find paths to all possible nodes
        # but exclude known target variables like 'y' from being treated as intermediates
        common_target_names = ['y', 'target', 'outcome', 'label']
        
        # Add the specific target_var to the list if provided
        if target_var and target_var not in common_target_names:
            common_target_names.append(target_var)
        
        for node in graph.nodes():
            if node != protected_attr and node in feature_cols:
                # Skip if this node looks like a target variable
                if node in common_target_names:
                    print(f"  Skipping node '{node}' as it appears to be a target variable")
                    continue
                    
                try:
                    # Find all simple paths from protected_attr to node
                    simple_paths = list(nx.all_simple_paths(graph, protected_attr, node, cutoff=max_path_length))
                    for path in simple_paths:
                        # Only include intermediate nodes (exclude start and end)
                        if len(path) > 2:
                            intermediate_path = path[1:-1]  # Exclude protected_attr and outcome
                            if intermediate_path and intermediate_path not in paths:
                                paths.append(intermediate_path)
                except nx.NetworkXNoPath:
                    continue
    
    # IMPROVEMENT: Remove duplicate paths and filter out paths containing target variables
    unique_paths = []
    common_target_names = ['y', 'target', 'outcome', 'label']
    
    # Add the specific target_var to the list if provided
    if target_var and target_var not in common_target_names:
        common_target_names.append(target_var)
    
    for path in paths:
        # Skip duplicate paths
        if path in unique_paths:
            continue
            
        # Filter out paths that contain target variables as intermediates
        path_contains_target = any(node in common_target_names for node in path)
        if path_contains_target:
            print(f"  Filtering out path {path} as it contains target variable")
            continue
            
        unique_paths.append(path)
    
    # Sort by path length (shorter paths first for interpretability)
    unique_paths.sort(key=len)

    if max_paths is not None and max_paths > 0:
        unique_paths = unique_paths[:max_paths]
    
    print(f"Identified {len(unique_paths)} unique causal paths from {protected_attr}")
    for i, path in enumerate(unique_paths):
        if target_var:
            full_path = [protected_attr] + path + [target_var]
        else:
            full_path = [protected_attr] + path
        print(f"  Path {i+1}: {' -> '.join(full_path)}")
    
    return unique_paths


def calculate_nie_three_step(individual, protected_attr, factual_value, cf_value,
                           causal_model, categorical_info, X_cols, mediator_vars, classifier):
    """
    IMPROVED: Calculate NIE using the three-step decomposition approach with better error handling
    """
    try:
        # Step 1: Get factual prediction (baseline)
        factual_features = individual['predictions_by_attr'][protected_attr]['factual']
        y_factual = classifier.predict_proba(factual_features.astype(float))[0, 1]
        
        # Step 2: Get total counterfactual effect
        cf_key = f'cf_{cf_value}'
        if cf_key not in individual['predictions_by_attr'][protected_attr]:
            return np.nan
            
        cf_features = individual['predictions_by_attr'][protected_attr][cf_key]
        y_total_cf = classifier.predict_proba(cf_features.astype(float))[0, 1]
        
        # Step 3: Calculate Natural Direct Effect (NDE)
        # IMPROVEMENT: More robust mediator intervention
        nde_intervention = {protected_attr: lambda x, val=str(cf_value): val}
        
        # Add interventions to keep mediators at factual levels
        if mediator_vars:
            for mediator in mediator_vars:
                if mediator in factual_features.columns:
                    factual_med_val = factual_features[mediator].iloc[0]
                    
                    # IMPROVEMENT: Better categorical handling
                    if mediator in categorical_info:
                        # Ensure we're using the exact string format expected by SCM
                        factual_med_val_str = str(int(factual_med_val))
                        nde_intervention[mediator] = lambda x, val=factual_med_val_str: val
                    else:
                        # For continuous variables (if any)
                        nde_intervention[mediator] = lambda x, val=float(factual_med_val): val
        
        # IMPROVEMENT: Better data preparation for SCM
        individual_data_str = pd.DataFrame([individual['factual_values']])
        
        # Ensure all columns from the causal model are present
        scm_nodes = set(causal_model.graph.nodes())
        missing_scm_cols = scm_nodes - set(individual_data_str.columns)
        
        if missing_scm_cols:
            print(f"Warning: Adding missing SCM columns for NIE: {missing_scm_cols}")
            for col in missing_scm_cols:
                # Add default values for missing columns
                individual_data_str[col] = 0
        
        # Convert to strings with better error handling - process ALL SCM nodes
        for col in scm_nodes:
            if col in individual_data_str.columns:
                if col in categorical_info:
                    try:
                        individual_data_str[col] = individual_data_str[col].map(categorical_info[col]['int_to_string'])
                    except KeyError as e:
                        print(f"Warning: Missing mapping for {col}, value {individual_data_str[col].iloc[0]}")
                        # Fallback: convert to string directly
                        individual_data_str[col] = individual_data_str[col].astype(str)
                else:
                    # For columns not in categorical_info, convert to string directly
                    individual_data_str[col] = individual_data_str[col].astype(str)
        
        try:
            nde_samples = gcm.interventional_samples(
                causal_model,
                nde_intervention,
                observed_data=individual_data_str
            )
            
            # IMPROVEMENT: More robust conversion back to integers
            for col, info in categorical_info.items():
                if col in nde_samples.columns:
                    # Use the safe conversion function
                    converted_values = []
                    for val in nde_samples[col]:
                        converted_val = safe_convert_categorical_to_int(val, col, categorical_info)
                        converted_values.append(converted_val)
                    nde_samples[col] = converted_values
            
            # IMPROVEMENT: Validate and add missing X_cols
            # Use the original individual factual values as fallback source
            individual_factual_df = pd.DataFrame([individual['factual_values']])
            
            nde_samples, missing_cols = validate_and_add_missing_columns(
                nde_samples, X_cols, individual_factual_df, 0, "NDE samples"
            )
            
            # Get NDE prediction
            try:
                # Additional validation before column selection
                available_cols = list(nde_samples.columns)
                missing_x_cols = [col for col in X_cols if col not in available_cols]
                
                if missing_x_cols:
                    print(f"ERROR: Still missing columns after validation: {missing_x_cols}")
                    print(f"Available columns in nde_samples: {available_cols}")
                    print(f"Required X_cols: {X_cols}")
                    return np.nan
                
                # Safe column selection
                nde_features = nde_samples[X_cols].copy()
                y_nde = classifier.predict_proba(nde_features.astype(float))[0, 1]
            except Exception as e:
                print(f"Error in NDE prediction: {str(e)}")
                print(f"nde_samples columns: {list(nde_samples.columns)}")
                print(f"X_cols: {X_cols}")
                return np.nan
            
            # Calculate effects
            total_effect = y_total_cf - y_factual
            natural_direct_effect = y_nde - y_factual
            natural_indirect_effect = total_effect - natural_direct_effect
            
            # IMPROVEMENT: Add validation
            if abs(natural_indirect_effect) > 1.0:
                print(f"Warning: Large NIE value ({natural_indirect_effect:.4f}) - check implementation")
            
            return {
                'nie': natural_indirect_effect,
                'nde': natural_direct_effect,
                'total_effect': total_effect,
                'y_factual': y_factual,
                'y_counterfactual': y_total_cf,
                'y_nde': y_nde,
                'mediators_used': mediator_vars,
                'validation_passed': abs(natural_indirect_effect) <= 1.0
            }
            
        except Exception as e:
            print(f"Error in NIE SCM intervention: {str(e)}")
            return np.nan
            
    except Exception as e:
        print(f"Error in calculate_nie_three_step: {str(e)}")
        return np.nan


def calculate_pse_path_blocking(individual, protected_attr, factual_value, cf_value,
                              path_vars, causal_model, categorical_info, X_cols, classifier,
                              y_factual=None, y_total_cf=None, individual_data_str=None,
                              return_features_only=False):
    """
    IMPROVED: Better path blocking with validation
    """
    try:
        # Get factual and counterfactual predictions
        factual_features = individual['predictions_by_attr'][protected_attr]['factual']
        if y_factual is None:
            y_factual = classifier.predict_proba(factual_features.astype(float))[0, 1]
        
        cf_key = f'cf_{cf_value}'
        if y_total_cf is None and cf_key not in individual['predictions_by_attr'][protected_attr]:
            return np.nan

        if y_total_cf is None:
            cf_features = individual['predictions_by_attr'][protected_attr][cf_key]
            y_total_cf = classifier.predict_proba(cf_features.astype(float))[0, 1]
        
        # IMPROVEMENT: Validate that path variables exist in the data
        # Check both in factual features AND in individual's original data
        available_path_vars = []
        for var in path_vars:
            if var in factual_features.columns:
                available_path_vars.append(var)
            elif var in individual['factual_values']:
                available_path_vars.append(var)
                print(f"  Note: Path variable '{var}' found in individual data but not in factual features")
        
        missing_vars = set(path_vars) - set(available_path_vars)
        if missing_vars:
            print(f"Warning: Path variables not found anywhere: {missing_vars}")
            print(f"  Factual features columns: {list(factual_features.columns)}")
            print(f"  Individual factual values: {list(individual['factual_values'].keys())}")
        
        if not available_path_vars:
            print(f"Error: No valid path variables found for path {path_vars}")
            return np.nan
        
        # Block path by keeping path variables at their factual values
        path_blocked_intervention = {protected_attr: lambda x, val=str(cf_value): val}
        
        # Add interventions to block the specific path
        for path_var in available_path_vars:
            # Try to get the value from factual features first, then from individual data
            if path_var in factual_features.columns:
                factual_path_val = factual_features[path_var].iloc[0]
            else:
                factual_path_val = individual['factual_values'][path_var]
                print(f"  Using path variable '{path_var}' from individual data: {factual_path_val}")
            
            if path_var in categorical_info:
                factual_path_val_str = str(int(factual_path_val))
                path_blocked_intervention[path_var] = lambda x, val=factual_path_val_str: val
            else:
                path_blocked_intervention[path_var] = lambda x, val=float(factual_path_val): val
        
        try:
            # Create individual data for intervention
            preformatted_observed_data = individual_data_str is not None
            if individual_data_str is None:
                individual_data_str_local = pd.DataFrame([individual['factual_values']])
            else:
                individual_data_str_local = individual_data_str.copy()
            
            # Ensure all columns from the causal model are present
            scm_nodes = set(causal_model.graph.nodes())
            missing_scm_cols = scm_nodes - set(individual_data_str_local.columns)
            
            if missing_scm_cols:
                print(f"Warning: Adding missing SCM columns for PSE: {missing_scm_cols}")
                for col in missing_scm_cols:
                    # Add default values for missing columns
                    individual_data_str_local[col] = '0' if preformatted_observed_data else 0
            
            # Convert to strings for categorical variables if data is not already preformatted.
            if not preformatted_observed_data:
                for col in scm_nodes:
                    if col in individual_data_str_local.columns:
                        if col in categorical_info:
                            individual_data_str_local[col] = individual_data_str_local[col].map(categorical_info[col]['int_to_string'])
                        else:
                            # For columns not in categorical_info, convert to string directly
                            individual_data_str_local[col] = individual_data_str_local[col].astype(str)
            
            path_blocked_samples = gcm.interventional_samples(
                causal_model,
                path_blocked_intervention,
                observed_data=individual_data_str_local
            )
            
            # Convert back to integers
            for col, info in categorical_info.items():
                if col in path_blocked_samples.columns:
                    converted_values = []
                    for val in path_blocked_samples[col]:
                        converted_val = safe_convert_categorical_to_int(val, col, categorical_info)
                        converted_values.append(converted_val)
                    path_blocked_samples[col] = converted_values
            
            # IMPROVEMENT: Validate and add missing X_cols
            # Use the original individual factual values as fallback source
            individual_factual_df = pd.DataFrame([individual['factual_values']])
            
            path_blocked_samples, missing_cols = validate_and_add_missing_columns(
                path_blocked_samples, X_cols, individual_factual_df, 0, "path-blocked samples"
            )
            
            # Prepare path-blocked feature row for downstream prediction.
            try:
                # Additional validation before column selection
                available_cols = list(path_blocked_samples.columns)
                missing_x_cols = [col for col in X_cols if col not in available_cols]
                
                if missing_x_cols:
                    print(f"ERROR: Still missing columns after validation: {missing_x_cols}")
                    print(f"Available columns in path_blocked_samples: {available_cols}")
                    print(f"Required X_cols: {X_cols}")
                    return np.nan
                
                # Safe column selection
                path_blocked_features = path_blocked_samples[X_cols].copy()
            except Exception as e:
                print(f"Error in path-blocked prediction: {str(e)}")
                print(f"path_blocked_samples columns: {list(path_blocked_samples.columns)}")
                print(f"X_cols: {X_cols}")
                return np.nan

            total_effect = y_total_cf - y_factual

            if return_features_only:
                return {
                    'path_blocked_features': path_blocked_features,
                    'total_effect': total_effect,
                    'y_factual': y_factual,
                    'y_counterfactual': y_total_cf,
                    'blocked_path': available_path_vars
                }

            if classifier is None:
                print("Error: classifier is required when return_features_only is False")
                return np.nan

            y_path_blocked = classifier.predict_proba(path_blocked_features.astype(float))[0, 1]
            
            # Calculate PSE
            effect_without_path = y_path_blocked - y_factual
            path_specific_effect = total_effect - effect_without_path
            
            # IMPROVEMENT: Add validation
            if abs(path_specific_effect) > 1.0:
                print(f"Warning: Large PSE value ({path_specific_effect:.4f}) for path {path_vars}")
            
            return {
                'pse': path_specific_effect,
                'total_effect': total_effect,
                'effect_without_path': effect_without_path,
                'y_factual': y_factual,
                'y_counterfactual': y_total_cf,
                'y_path_blocked': y_path_blocked,
                'blocked_path': available_path_vars,
                'validation_passed': abs(path_specific_effect) <= 1.0
            }
            
        except Exception as e:
            print(f"Error in PSE path blocking intervention: {str(e)}")
            return np.nan
            
    except Exception as e:
        print(f"Error in calculate_pse_path_blocking: {str(e)}")
        return np.nan


def aggregate_nie_results(nie_results):
    """
    Aggregate NIE results across all individuals.
    """
    all_nie_scores = []
    all_nde_scores = []
    all_te_scores = []
    all_total_effects = []
    
    for individual_result in nie_results:
        for comparison, nie_data in individual_result.items():
            if isinstance(nie_data, dict) and 'nie' in nie_data:
                all_nie_scores.append(nie_data['nie'])
                if 'nde' in nie_data:
                    all_nde_scores.append(nie_data['nde'])
                if 'total_effect' in nie_data:
                    all_total_effects.append(nie_data['total_effect'])
    
    if all_nie_scores:
        return {
            'nie_score': np.mean(all_nie_scores),
            'nie_std': np.std(all_nie_scores),
            'nie_median': np.median(all_nie_scores),
            'nie_min': np.min(all_nie_scores),
            'nie_max': np.max(all_nie_scores),
            'nde_score': np.mean(all_nde_scores) if all_nde_scores else np.nan,
            'total_effect': np.mean(all_total_effects) if all_total_effects else np.nan,
            'n_individuals': len(all_nie_scores),
            'raw_nie_scores': all_nie_scores
        }
    else:
        return {'nie_score': np.nan, 'error': 'No valid NIE scores calculated'}


def aggregate_pse_results(pse_results):
    """
    Aggregate PSE results across all individuals for a specific path.
    """
    all_pse_scores = []
    all_total_effects = []
    all_effects_without_path = []
    
    for individual_result in pse_results:
        for comparison, pse_data in individual_result.items():
            if isinstance(pse_data, dict) and 'pse' in pse_data:
                all_pse_scores.append(pse_data['pse'])
                if 'total_effect' in pse_data:
                    all_total_effects.append(pse_data['total_effect'])
                if 'effect_without_path' in pse_data:
                    all_effects_without_path.append(pse_data['effect_without_path'])
    
    if all_pse_scores:
        return {
            'pse_score': np.mean(all_pse_scores),
            'pse_std': np.std(all_pse_scores),
            'pse_median': np.median(all_pse_scores),
            'pse_min': np.min(all_pse_scores),
            'pse_max': np.max(all_pse_scores),
            'total_effect': np.mean(all_total_effects) if all_total_effects else np.nan,
            'effect_without_path': np.mean(all_effects_without_path) if all_effects_without_path else np.nan,
            'n_individuals': len(all_pse_scores),
            'raw_pse_scores': all_pse_scores
        }
    else:
        return {'pse_score': np.nan, 'error': 'No valid PSE scores calculated'}


def aggregate_precomputed_pse_for_group(precomputed_attr_pse, filtered_individuals):
    """
    Aggregate precomputed per-individual PSE results for a specific group.
    """
    if not isinstance(precomputed_attr_pse, dict):
        return {'pse_score': np.nan, 'error': 'Invalid precomputed PSE payload'}

    precomputed_individual_scores = precomputed_attr_pse.get('individual_scores', [])
    precomputed_aggregated = precomputed_attr_pse.get('aggregated_results', {})

    group_indices = {individual['original_index'] for individual in filtered_individuals}
    selected_individual_scores = [
        score for score in precomputed_individual_scores
        if score.get('original_index') in group_indices
    ]

    if not selected_individual_scores:
        return {'pse_score': np.nan, 'error': 'No precomputed PSE results for this group'}

    grouped_path_scores = {}
    for score_entry in selected_individual_scores:
        for path_name, path_scores in score_entry.get('path_scores_by_path', {}).items():
            if path_name not in grouped_path_scores:
                grouped_path_scores[path_name] = []
            if path_scores:
                grouped_path_scores[path_name].append(path_scores)

    aggregated_group_pse = {}
    for path_name, path_scores in grouped_path_scores.items():
        if path_scores:
            aggregated_pse = aggregate_pse_results(path_scores)
            if path_name in precomputed_aggregated and isinstance(precomputed_aggregated[path_name], dict):
                if 'path_variables' in precomputed_aggregated[path_name]:
                    aggregated_pse['path_variables'] = precomputed_aggregated[path_name]['path_variables']
            aggregated_group_pse[path_name] = aggregated_pse

    if not aggregated_group_pse:
        return {'pse_score': np.nan, 'error': 'No valid precomputed PSE path scores for this group'}

    return aggregated_group_pse


def calculate_individual_treatment_effect(individual_predictions, classifier, protected_attr, df_sample=None):
    """
    IMPROVED: Calculate ITE with stratified group-level aggregation for fairness evaluation
    """
    try:
        print(f"Calculating stratified Individual Treatment Effect (ITE) for {protected_attr}...")
        
        # Organize by protected group first
        groups_data = {}
        
        # Process each individual and group them by protected attribute value
        for individual in individual_predictions:
            if protected_attr not in individual['predictions_by_attr']:
                continue
                
            # Get the protected group this individual belongs to
            protected_value = individual['factual_values'][protected_attr]
            
            if protected_value not in groups_data:
                groups_data[protected_value] = {
                    'individuals': [],
                    'ite_values': [],
                    'ite_details': []
                }
            
            attr_predictions = individual['predictions_by_attr'][protected_attr]
            factual_value = individual['factual_values'][protected_attr]
            original_idx = individual['original_index']
            
            # Get factual prediction
            if 'factual' not in attr_predictions:
                continue
                
            factual_features = attr_predictions['factual']
            y_factual = classifier.predict_proba(factual_features.astype(float))[0, 1]
            
            individual_ites = []
            
            # Calculate ITE for each counterfactual value
            for cf_key, cf_features in attr_predictions.items():
                if cf_key.startswith('cf_'):
                    cf_value = cf_key.replace('cf_', '')
                    
                    # Get counterfactual prediction
                    y_counterfactual = classifier.predict_proba(cf_features.astype(float))[0, 1]
                    
                    # Calculate ITE: Y_i(a') - Y_i(a) - PRESERVE SIGN
                    ite = y_counterfactual - y_factual
                    individual_ites.append(ite)
                    
                    # Store detailed information
                    ite_detail = {
                        'individual_idx': original_idx,
                        'protected_attr': protected_attr,
                        'protected_group': protected_value,
                        'factual_value': factual_value,
                        'counterfactual_value': cf_value,
                        'y_factual': y_factual,
                        'y_counterfactual': y_counterfactual,
                        'ite': ite,
                        'beneficial': ite > 0,
                        'harmful': ite < 0,
                        'effect_magnitude': abs(ite)
                    }
                    groups_data[protected_value]['ite_details'].append(ite_detail)
            
            # Store individual's ITEs in their group
            if individual_ites:
                groups_data[protected_value]['individuals'].append(individual)
                groups_data[protected_value]['ite_values'].extend(individual_ites)
        
        # Calculate group-level statistics first
        group_results = {}
        all_individual_ites = []  # For overall calculation
        
        for group_value, group_data in groups_data.items():
            if not group_data['ite_values']:
                continue
                
            group_ites = group_data['ite_values']
            all_individual_ites.extend(group_ites)  # Collect for overall
            
            # Group-level statistics (preserving direction)
            beneficial_ites = [ite for ite in group_ites if ite > 0]
            harmful_ites = [ite for ite in group_ites if ite < 0]
            
            group_stats = {
                'n_individuals': len(group_data['individuals']),
                'n_comparisons': len(group_ites),
                'mean_ite': np.mean(group_ites),
                'median_ite': np.median(group_ites),
                'std_ite': np.std(group_ites),
                'min_ite': np.min(group_ites),
                'max_ite': np.max(group_ites),
                
                # Treatment effect directions
                'beneficial_rate': len(beneficial_ites) / len(group_ites) * 100,
                'harmful_rate': len(harmful_ites) / len(group_ites) * 100,
                'neutral_rate': len([ite for ite in group_ites if ite == 0]) / len(group_ites) * 100,
                
                # Conditional means (only for non-zero effects)
                'mean_beneficial_effect': np.mean(beneficial_ites) if beneficial_ites else 0,
                'mean_harmful_effect': np.mean(harmful_ites) if harmful_ites else 0,
                
                # Magnitude analysis
                'mean_absolute_effect': np.mean([abs(ite) for ite in group_ites]),
                'max_absolute_effect': np.max([abs(ite) for ite in group_ites]),
                
                # Raw data for further analysis
                'raw_ites': group_ites,
                'detailed_comparisons': group_data['ite_details']
            }
            
            group_results[f'group_{group_value}'] = group_stats
            
            print(f"  Group {protected_attr}={group_value}: n={group_stats['n_individuals']}, "
                  f"mean_ite={group_stats['mean_ite']:.4f}, "
                  f"beneficial={group_stats['beneficial_rate']:.1f}%")
        
        # Calculate cross-group fairness metrics
        if len(group_results) >= 2:
            group_means = [stats['mean_ite'] for stats in group_results.values()]
            group_beneficial_rates = [stats['beneficial_rate'] for stats in group_results.values()]
            group_harmful_rates = [stats['harmful_rate'] for stats in group_results.values()]
            group_sizes = [stats['n_individuals'] for stats in group_results.values()]
            
            fairness_metrics = {
                'ite_disparity': max(group_means) - min(group_means),
                'ite_range': max(group_means) - min(group_means),
                'beneficial_rate_disparity': max(group_beneficial_rates) - min(group_beneficial_rates),
                'harmful_rate_disparity': max(group_harmful_rates) - min(group_harmful_rates),
                'group_size_ratio': max(group_sizes) / min(group_sizes) if min(group_sizes) > 0 else np.inf,
                'between_group_variance': np.var(group_means),
                'worst_group_mean': min(group_means),  # Most negative mean ITE
                'best_group_mean': max(group_means)    # Most positive mean ITE
            }
        else:
            fairness_metrics = {}
        
        # Overall aggregated results (recommended approach)
        if all_individual_ites:
            # THIS IS YOUR overall_ite_mean - calculated from ALL individual ITEs across ALL groups
            overall_ite_mean = np.mean(all_individual_ites)
            
            overall_results = {
                # Main metric for overall_ite_mean
                'ite_mean': overall_ite_mean,  # This becomes your overall_ite_mean
                'ite_median': np.median(all_individual_ites),
                'ite_std': np.std(all_individual_ites),
                'ite_min': np.min(all_individual_ites),
                'ite_max': np.max(all_individual_ites),
                
                # Direction analysis
                'n_individuals_total': len(all_individual_ites),
                'pct_positive_effects': np.mean([ite > 0 for ite in all_individual_ites]) * 100,
                'pct_negative_effects': np.mean([ite < 0 for ite in all_individual_ites]) * 100,
                'pct_zero_effects': np.mean([ite == 0 for ite in all_individual_ites]) * 100,
                
                # Magnitude analysis
                'mean_absolute_effect': np.mean([abs(ite) for ite in all_individual_ites]),
                'max_absolute_effect': np.max([abs(ite) for ite in all_individual_ites]),
                
                # Fairness evaluation
                'fairness_metrics': fairness_metrics,
                'group_results': group_results,
                'n_groups': len(group_results),
                
                # Raw data for detailed analysis
                'raw_ite_values': all_individual_ites,
                'detailed_comparisons': [detail for group_data in groups_data.values() 
                                       for detail in group_data['ite_details']]
            }
            
            print(f"✅ Stratified ITE calculated:")
            print(f"   Overall ITE mean: {overall_ite_mean:.6f}")
            print(f"   Total individuals: {len(all_individual_ites)}")
            print(f"   Groups analyzed: {len(group_results)}")
            if fairness_metrics:
                print(f"   ITE disparity between groups: {fairness_metrics['ite_disparity']:.6f}")
            
            return overall_results
        else:
            return {'ite_mean': np.nan, 'error': 'No ITE calculations completed'}
            
    except Exception as e:
        print(f"Error in stratified ITE calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'ite_mean': np.nan, 'error': str(e)}


def aggregate_ite_results(individual_ites, ite_details, protected_attr, effect_pairs=None):
    """
    IMPROVED: Better ITE aggregation with effect pair analysis
    """
    if not individual_ites:
        return {'ite_mean': np.nan, 'error': 'No ITE values to aggregate'}
    
    ite_array = np.array(individual_ites)
    
    # Basic distribution statistics
    ite_stats = {
        'ite_mean': np.mean(ite_array),
        'ite_median': np.median(ite_array),
        'ite_std': np.std(ite_array),
        'ite_var': np.var(ite_array),
        'ite_min': np.min(ite_array),
        'ite_max': np.max(ite_array),
        'ite_range': np.max(ite_array) - np.min(ite_array),
        'n_individuals': len(individual_ites),
        'n_effect_pairs': len(effect_pairs) if effect_pairs else 0,
        'effect_pairs_analyzed': effect_pairs if effect_pairs else [],
        'protected_attribute': protected_attr
    }
    
    # IMPROVEMENT: Analyze by effect pair
    if effect_pairs:
        pair_analysis = {}
        df_details = pd.DataFrame(ite_details)
        
        for pair in effect_pairs:
            pair_data = df_details[df_details['effect_pair'] == pair]
            if len(pair_data) > 0:
                pair_ites = pair_data['ite'].values
                pair_analysis[pair] = {
                    'n_individuals': len(pair_ites),
                    'mean_ite': np.mean(pair_ites),
                    'median_ite': np.median(pair_ites),
                    'std_ite': np.std(pair_ites),
                    'min_ite': np.min(pair_ites),
                    'max_ite': np.max(pair_ites),
                    'pct_positive': (pair_ites > 0).mean() * 100,
                    'pct_negative': (pair_ites < 0).mean() * 100,
                    'pct_neutral': (pair_ites == 0).mean() * 100
                }
        
        ite_stats['effect_pair_analysis'] = pair_analysis
    
    # Percentile statistics
    percentiles = [5, 10, 25, 75, 90, 95]
    for p in percentiles:
        ite_stats[f'ite_p{p}'] = np.percentile(ite_array, p)
    
    # Distribution shape statistics
    try:
        from scipy import stats
        ite_stats['ite_skewness'] = stats.skew(ite_array)
        ite_stats['ite_kurtosis'] = stats.kurtosis(ite_array)
    except ImportError:
        ite_stats['ite_skewness'] = np.nan
        ite_stats['ite_kurtosis'] = np.nan
    
    # Effect direction analysis
    positive_effects = ite_array[ite_array > 0]
    negative_effects = ite_array[ite_array < 0]
    zero_effects = ite_array[ite_array == 0]
    
    ite_stats.update({
        'n_positive_effects': len(positive_effects),
        'n_negative_effects': len(negative_effects),
        'n_zero_effects': len(zero_effects),
        'pct_positive_effects': len(positive_effects) / len(ite_array) * 100,
        'pct_negative_effects': len(negative_effects) / len(ite_array) * 100,
        'pct_zero_effects': len(zero_effects) / len(ite_array) * 100
    })
    
    # Mean effects by direction
    ite_stats['mean_positive_effect'] = np.mean(positive_effects) if len(positive_effects) > 0 else np.nan
    ite_stats['mean_negative_effect'] = np.mean(negative_effects) if len(negative_effects) > 0 else np.nan
    
    # Heterogeneity measures
    ite_stats['effect_heterogeneity'] = np.std(ite_array)
    ite_stats['coefficient_of_variation'] = np.std(ite_array) / abs(np.mean(ite_array)) if np.mean(ite_array) != 0 else np.inf
    
    # Magnitude analysis (absolute effects)
    abs_effects = np.abs(ite_array)
    ite_stats.update({
        'mean_absolute_effect': np.mean(abs_effects),
        'median_absolute_effect': np.median(abs_effects),
        'max_absolute_effect': np.max(abs_effects)
    })
    
    # IMPROVEMENT: Statistical significance analysis
    se_approx = np.std(ite_array) / np.sqrt(len(ite_array))
    significant_threshold = 2 * se_approx
    significant_effects = ite_array[np.abs(ite_array) > significant_threshold]
    
    ite_stats.update({
        'n_significant_effects': len(significant_effects),
        'pct_significant_effects': len(significant_effects) / len(ite_array) * 100,
        'significance_threshold': significant_threshold,
        'standard_error': se_approx
    })
    
    # IMPROVEMENT: Add confidence intervals for mean ITE
    ci_95 = 1.96 * se_approx
    ite_stats.update({
        'ite_mean_ci_lower': np.mean(ite_array) - ci_95,
        'ite_mean_ci_upper': np.mean(ite_array) + ci_95,
        'ci_width': 2 * ci_95
    })
    
    # Store raw data for detailed analysis
    ite_stats['raw_ite_values'] = individual_ites
    ite_stats['detailed_comparisons'] = ite_details
    
    return ite_stats


def analyze_ite_heterogeneity(ite_results, individual_predictions, df_sample, demographic_vars=None):
    """
    Analyze heterogeneity in Individual Treatment Effects across different subgroups.
    
    Args:
        ite_results: Results from calculate_individual_treatment_effect
        individual_predictions: Individual prediction data
        df_sample: Sample dataframe with demographic information
        demographic_vars: List of variables to analyze heterogeneity across
    
    Returns:
        dict: Heterogeneity analysis results
    """
    try:
        print("Analyzing ITE heterogeneity across subgroups...")
        
        if 'detailed_comparisons' not in ite_results:
            return {'error': 'No detailed comparison data available for heterogeneity analysis'}
        
        detailed_comparisons = ite_results['detailed_comparisons']
        
        # Default demographic variables if not specified
        if demographic_vars is None:
            demographic_vars = [col for col in df_sample.columns 
                              if col in ['age', 'education', 'income', 'race', 'gender', 'sex']]
            demographic_vars = demographic_vars[:3]  # Limit to first 3 to avoid over-analysis
        
        heterogeneity_results = {}
        
        for demo_var in demographic_vars:
            if demo_var not in df_sample.columns:
                continue
                
            print(f"  Analyzing ITE heterogeneity by {demo_var}...")
            
            # Group ITEs by demographic variable
            demo_groups = {}
            
            for comparison in detailed_comparisons:
                individual_idx = comparison['individual_idx']
                ite_value = comparison['ite']
                
                if individual_idx in df_sample.index:
                    demo_value = df_sample.loc[individual_idx, demo_var]
                    
                    if demo_value not in demo_groups:
                        demo_groups[demo_value] = []
                    demo_groups[demo_value].append(ite_value)
            
            # Calculate statistics for each group
            group_stats = {}
            for group_value, group_ites in demo_groups.items():
                if len(group_ites) > 0:
                    group_stats[group_value] = {
                        'n': len(group_ites),
                        'mean_ite': np.mean(group_ites),
                        'median_ite': np.median(group_ites),
                        'std_ite': np.std(group_ites),
                        'min_ite': np.min(group_ites),
                        'max_ite': np.max(group_ites)
                    }
            
            # Calculate between-group heterogeneity
            if len(group_stats) > 1:
                group_means = [stats['mean_ite'] for stats in group_stats.values()]
                between_group_var = np.var(group_means)
                total_var = np.var([ite for group_ites in demo_groups.values() for ite in group_ites])
                heterogeneity_ratio = between_group_var / total_var if total_var > 0 else 0
                
                heterogeneity_results[demo_var] = {
                    'group_statistics': group_stats,
                    'between_group_variance': between_group_var,
                    'total_variance': total_var,
                    'heterogeneity_ratio': heterogeneity_ratio,
                    'n_groups': len(group_stats)
                }
            else:
                heterogeneity_results[demo_var] = {
                    'group_statistics': group_stats,
                    'error': 'Insufficient groups for heterogeneity analysis'
                }
        
        return heterogeneity_results
        
    except Exception as e:
        print(f"Error in ITE heterogeneity analysis: {str(e)}")
        return {'error': str(e)}

def fit_scm_for_categorical_data(df, dag, var_types, random_state=42):
    """
    Fit SCM for categorical data following DoWhy's official approach.
    Use StructuralCausalModel with ClassifierFCM for categorical data.
    """
    try:
        print("Creating StructuralCausalModel for categorical data...")
        graph = nx.DiGraph(dag)
        # Using the standard StructuralCausalModel
        causal_model = gcm.StructuralCausalModel(graph)
        
        # Convert ALL data to strings as required by DoWhy for categorical data
        df_categorical = df.copy()
        categorical_info = {}
        
        for col in df_categorical.columns:
            if col in var_types and var_types[col] in ['categorical', 'binary']:
                # Store original integer values for later conversion
                unique_values = sorted(df_categorical[col].unique())
                int_to_string = {int(v): str(v) for v in unique_values}
                string_to_int = {str(v): int(v) for v in unique_values}
                
                # Convert to strings as required by DoWhy
                df_categorical[col] = df_categorical[col].astype(str)
                
                categorical_info[col] = {
                    'original_int_values': unique_values,
                    'int_to_string': int_to_string,
                    'string_to_int': string_to_int,
                    'fallback_value': unique_values[0] if unique_values else 0
                }
                print(f"Converted {col} ({var_types[col]}) to strings: {unique_values}")
        
        # Store categorical info for later use
        causal_model._categorical_info = categorical_info
        
        # Identify root nodes
        root_nodes = set(graph.nodes())
        for edge in dag:
            if edge[1] in root_nodes:
                root_nodes.remove(edge[1])
        
        print(f"Root nodes: {root_nodes}")
        
        # Assign mechanisms using DoWhy's official approach for categorical data
        for node in graph.nodes():
            if node in df_categorical.columns:
                if node in root_nodes:
                    print(f"Setting EmpiricalDistribution for ROOT node: {node}")
                    causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
                else:
                    # Use ClassifierFCM for non-root categorical nodes.
                    # This uses a classifier (like RandomForestClassifier) instead of a regressor.
                    print(f"Assigning ClassifierFCM(RandomForest) to NON-ROOT node: {node}")
                    causal_model.set_causal_mechanism(
                        node, 
                        gcm.ClassifierFCM(classifier_model=create_random_forest_classifier())
                    )

        print("Fitting causal model...")
        df_scm = get_data_for_algorithm(df_categorical, 'scm')
        gcm.fit(causal_model, df_scm)
        
        print("✅ StructuralCausalModel fitted successfully for categorical data")
        return causal_model
        
    except Exception as e:
        print(f"Error fitting causal model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def safe_convert_categorical_to_int(value, col, categorical_info):
    """
    Safely convert a categorical string value back to integer with robust fallbacks.
    """
    if col not in categorical_info:
        # If no categorical info, try direct conversion
        try:
            return int(float(str(value)))
        except:
            return 0
    
    info = categorical_info[col]
    value_str = str(value)
    
    # Try direct lookup first
    if value_str in info['string_to_int']:
        return info['string_to_int'][value_str]
    
    # Try without decimals (e.g., "2.0" -> "2")
    try:
        clean_value = str(int(float(value_str)))
        if clean_value in info['string_to_int']:
            return info['string_to_int'][clean_value]
    except:
        pass
    
    # Try direct integer conversion
    try:
        int_val = int(float(value_str))
        if int_val in info['original_int_values']:
            return int_val
    except:
        pass
    
    # Final fallback: use a valid value from original data
    fallback = info['fallback_value']
    print(f"WARNING: Could not convert '{value}' in column '{col}', using fallback {fallback}")
    return fallback

def generate_synthetic_data(causal_model, 
                            n_samples, 
                            interventions=None, 
                            var_types=None,
                            balance_outcome=False,
                            balance_protected=False,
                            target=None,
                            protected_attrs=None,
                            original_data=None):
    """
    Generate synthetic data with string categorical handling and convert back to integers.
    Handle string categorical data from SCM and convert back to integers
    """
    if causal_model is None:
        print("Error: No causal model provided for synthetic data generation")
        return None

    try:
        print(f"Generating {n_samples} synthetic samples WITHOUT type conversion issues...")
        
        # Generate basic synthetic data
        if interventions:
            synthetic_data = gcm.interventional_samples(
                causal_model,
                interventions=interventions,
                num_samples_to_draw=n_samples
            )
        else:
            synthetic_data = gcm.draw_samples(causal_model, num_samples=n_samples)
        
        print(f"Generated synthetic data shape: {synthetic_data.shape}")
        print(f"Columns: {list(synthetic_data.columns)}")
        
        # Get categorical info from the fitted model
        categorical_info = getattr(causal_model, '_categorical_info', {})
        
        # ROBUST processing - handle all edge cases
        for col in synthetic_data.columns:
            if col in categorical_info:
                print(f"Converting categorical column {col} from strings back to integers...")
                
                # Apply safe conversion to each value
                converted_values = []
                for value in synthetic_data[col]:
                    converted_val = safe_convert_categorical_to_int(value, col, categorical_info)
                    converted_values.append(converted_val)
                
                # Update the column
                synthetic_data[col] = converted_values
                
                # Ensure integer type
                synthetic_data[col] = pd.Series(converted_values).astype(int)
                
                # Validate all values are in original range
                original_values = categorical_info[col]['original_int_values']
                invalid_mask = ~synthetic_data[col].isin(original_values)
                
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    print(f"WARNING: {invalid_count} values outside original range in {col}")
                    
                    # Replace with random valid values
                    replacement_values = np.random.choice(original_values, size=invalid_count)
                    synthetic_data.loc[invalid_mask, col] = replacement_values
                    print(f"FIXED: Replaced out-of-range values in {col}")
                
                print(f"✅ {col}: converted to integers, range: {synthetic_data[col].min()}-{synthetic_data[col].max()}")
            
            else:
                # For any remaining columns, ensure no NaN and proper type
                if synthetic_data[col].isna().any():
                    if original_data is not None and col in original_data.columns:
                        fill_value = original_data[col].mode().iloc[0] if len(original_data[col].mode()) > 0 else 0
                    else:
                        fill_value = synthetic_data[col].mode().iloc[0] if len(synthetic_data[col].mode()) > 0 else 0
                    synthetic_data[col] = synthetic_data[col].fillna(fill_value)
                
                # Convert to int if it should be categorical
                if col in var_types and var_types[col] == 'categorical':
                    synthetic_data[col] = synthetic_data[col].astype(int)
        
        # Final validation: check data integrity
        print("Final validation - checking data integrity...")
        for col in synthetic_data.columns:
            if col in var_types and var_types[col] == 'categorical':
                # Check for any remaining invalid values
                unique_vals = synthetic_data[col].unique()
                print(f"Final {col}: unique values = {sorted(unique_vals)}")
                
                # Ensure no negative values (common SCM issue)
                if (synthetic_data[col] < 0).any():
                    negative_count = (synthetic_data[col] < 0).sum()
                    print(f"CRITICAL: Found {negative_count} negative values in {col}")
                    
                    # Replace with valid values
                    if col in categorical_info:
                        valid_values = categorical_info[col]['original_int_values']
                        replacement_values = np.random.choice(valid_values, size=negative_count)
                        synthetic_data.loc[synthetic_data[col] < 0, col] = replacement_values
                        print(f"FIXED: Replaced negative values in {col}")
        
        # Apply balancing if requested
        if balance_outcome and target and target in synthetic_data.columns:
            print(f"Balancing target variable '{target}'...")
            target_dist = synthetic_data[target].value_counts()
            print(f"Target distribution before balancing: {target_dist.to_dict()}")
            
            if len(target_dist) >= 2:
                # Get the two most frequent classes
                top_classes = target_dist.head(2).index
                class_data = [synthetic_data[synthetic_data[target] == cls] for cls in top_classes]
                min_size = min(len(data) for data in class_data)
                
                if min_size > 0:
                    balanced_data = pd.concat([
                        data.sample(min_size, random_state=42) for data in class_data
                    ])
                    synthetic_data = balanced_data
                    print(f"Balanced to {len(synthetic_data)} samples")
        
        print(f"Successfully generated {len(synthetic_data)} ROBUST synthetic samples")
        return synthetic_data
        
    except Exception as e:
        print(f"Error generating synthetic data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_fairness_on_data(df,
                              protected_attrs,
                              target,
                              disadvantage_group,
                              estimator,
                              X_cols,
                              dag,
                              var_types,
                              causal_model=None, 
                              random_state=42,
                              sample_percentage=1.0,
                              min_samples=10,
                              calculate_pse=False,
                              pse_sample_size=None,
                              max_paths=None):
    """
    Counterfactual fairness evaluation with Sequential Single-Attribute Analysis
    across Multiple Disadvantaged Groups.
    
    Analyzes fairness for:
    1. Individual protected attributes (single-attribute disadvantage)
    2. Pairwise combinations (dual disadvantage) 
    3. Full intersection (intersectional disadvantage)
    
    Args:
        calculate_pse: bool, default False. Whether to calculate Path-Specific Effect (PSE).
                      Set to False to skip PSE calculation for faster processing.
    """
    try:
        print(f"Evaluating COUNTERFACTUAL fairness on dataset with {len(df)} samples...")
        print(f"Protected attributes to analyze: {protected_attrs}")
        target_dist = df[target].value_counts()
        print(f"Target distribution: {target_dist.to_dict()}")

        if len(target_dist) < 2:
            print("Warning: Only one class found in target variable")
            return {'fairness_score': np.nan, 'error': 'Single class in target variable'}

        if causal_model is None:
            print("Error: A fitted causal model must be provided.")
            return {'fairness_score': np.nan, 'error': 'No causal model provided'}

        # Sample data for processing
        total_rows = len(df)
        num_samples = max(min_samples, int(total_rows * sample_percentage))
        num_samples = min(num_samples, total_rows)
        print(f"Processing {num_samples} samples for counterfactual analysis...")
        
        sample_indices = np.random.choice(df.index, size=num_samples, replace=False)
        df_sample = df.loc[sample_indices].copy()

        # Generate all disadvantaged groups to analyze
        disadvantaged_groups = generate_disadvantaged_groups(protected_attrs, disadvantage_group, df_sample)
        
        # Convert sample to string format for SCM
        df_sample_str = df_sample.copy()
        categorical_info = getattr(causal_model, '_categorical_info', {})
        for col, info in categorical_info.items():
            if col in df_sample_str.columns:
                 df_sample_str[col] = df_sample_str[col].map(info['int_to_string'])

        print(f"\nEvaluating COUNTERFACTUAL fairness with Multiple Group Analysis...")
        print(f"Will analyze {len(disadvantaged_groups)} different disadvantaged groups")
        
        # Generate counterfactuals for ALL sampled individuals (do this once)
        print("\nStep 1: Generating counterfactuals for all individuals across all protected attributes...")
        all_individual_predictions = generate_all_counterfactuals(
            df_sample_str, protected_attrs, causal_model, categorical_info, X_cols, df_sample, target
        )
        
        if not all_individual_predictions:
            print("Error: No counterfactual predictions could be generated.")
            return {'fairness_score': np.nan, 'error': 'Prediction generation failed'}
        
        print(f"✅ Generated counterfactuals for {len(all_individual_predictions)} individuals")
        
        # Train a single classifier on all factual data
        print("\nStep 2: Training classifier on factual data...")
        classifier = train_classifier_on_factual_data(all_individual_predictions, estimator, random_state)
        
        # Analyze fairness for each disadvantaged group
        print(f"\nStep 3: Analyzing fairness across {len(disadvantaged_groups)} disadvantaged groups...")

        cached_pse_paths = None
        if calculate_pse and dag:
            print("Pre-computing causal paths for PSE (reused across groups)...")
            cached_pse_paths = {}
            dag_edges = dag if isinstance(dag, list) else list(dag.edges()) if hasattr(dag, 'edges') else list(dag)
            for attr in protected_attrs:
                cached_pse_paths[attr] = identify_causal_paths(
                    attr,
                    dag_edges,
                    X_cols,
                    target_var=target,
                    max_paths=max_paths
                )
        
        precomputed_pse = None
        if calculate_pse and causal_model and categorical_info and X_cols:
            print("\nStep 2.5: Precomputing PSE once per attribute for reuse across groups...")
            precomputed_pse = {}
            for attr in protected_attrs:
                try:
                    specific_paths = None
                    if cached_pse_paths and attr in cached_pse_paths:
                        specific_paths = cached_pse_paths[attr]

                    dag_edges = None
                    if specific_paths is None:
                        dag_edges = dag if isinstance(dag, list) else list(dag.edges()) if hasattr(dag, 'edges') else list(dag) if dag else None

                    precomputed_pse[attr] = calculate_path_specific_effect(
                        all_individual_predictions,
                        classifier,
                        attr,
                        causal_model,
                        categorical_info,
                        X_cols,
                        specific_paths=specific_paths,
                        dag_edges=dag_edges,
                        target_var=target,
                        pse_sample_size=pse_sample_size,
                        max_paths=max_paths,
                        random_state=random_state,
                        return_individual_scores=True
                    )
                    print(f"  ✅ Precomputed PSE for {attr}")
                except Exception as e:
                    print(f"  ⚠️ Failed to precompute PSE for {attr}: {str(e)}")
                    precomputed_pse[attr] = {'pse_score': np.nan, 'error': str(e)}

        group_results = {}
        all_fairness_scores = []
        
        for group_name, group_definition in disadvantaged_groups.items():
            print(f"\n--- Analyzing Group: {group_name} ---")
            print(f"Definition: {group_definition}")
            
            group_fairness = analyze_group_fairness(
                group_definition, 
                group_name, 
                all_individual_predictions, 
                df_sample, 
                protected_attrs, 
                classifier, 
                min_samples, 
                causal_model, 
                categorical_info, 
                X_cols,
                dag,
                target,  # Pass the target variable name
                calculate_pse,
                pse_sample_size,
                max_paths,
                random_state,
                cached_pse_paths,
                precomputed_pse)  # Pass PSE controls and cached paths
            
            group_results[group_name] = group_fairness
            
            if 'max_fairness_score' in group_fairness and not pd.isna(group_fairness['max_fairness_score']):
                all_fairness_scores.append(abs(group_fairness['max_fairness_score']))
        
        # Compile overall results
        print(f"\n{'='*80}")
        print("MULTIPLE GROUP COUNTERFACTUAL FAIRNESS ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        overall_results = compile_overall_results(group_results, all_fairness_scores, protected_attrs)
        
        # Print summary
        print_fairness_summary(group_results, overall_results)
        
        return overall_results
        
    except Exception as e:
        print(f"An unexpected error occurred in evaluate_fairness_on_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'fairness_score': np.nan, 'error': str(e), 'analysis_type': 'multiple_group_analysis'}


def generate_disadvantaged_groups(protected_attrs, original_disadvantage_group, df_sample):
    """Generate all relevant disadvantaged groups to analyze."""
    
    print("\n=== GENERATING DISADVANTAGED GROUPS TO ANALYZE ===")
    
    # Get unique values for each protected attribute
    attr_values = {}
    for attr in protected_attrs:
        unique_vals = sorted(df_sample[attr].unique())
        attr_values[attr] = unique_vals
        print(f"{attr}: {unique_vals}")
    
    disadvantaged_groups = {}
    
    # 1. Single-attribute disadvantaged groups
    print("\n1. Single-attribute disadvantaged groups:")
    for attr in protected_attrs:
        if attr in original_disadvantage_group:
            disadvantaged_value = original_disadvantage_group[attr]

           # FIX: Type-tolerant matching
            matching_value = None
            
            # Direct match first
            if disadvantaged_value in attr_values[attr]:
                matching_value = disadvantaged_value
            else:
                # Try string conversion
                str_disadvantaged = str(disadvantaged_value)
                for data_val in attr_values[attr]:
                    if str(data_val) == str_disadvantaged:
                        matching_value = data_val
                        break
            
            if matching_value is not None:
                group_name = f"single_{attr}"
                group_def = {attr: matching_value}
                disadvantaged_groups[group_name] = group_def
                
                # Count samples with proper type handling
                if isinstance(matching_value, str):
                    count = len(df_sample.query(f"`{attr}` == '{matching_value}'"))
                else:
                    count = len(df_sample.query(f"`{attr}` == {matching_value}"))
                print(f"  {group_name}: {group_def} -> {count} samples")
            else:
                print(f"  WARNING: {attr} value {disadvantaged_value} not found in data {attr_values[attr]}")
    
    
    # 2. Pairwise combinations
    print("\n2. Pairwise disadvantaged groups:")
    from itertools import combinations
    for attr1, attr2 in combinations(protected_attrs, 2):
        if attr1 in original_disadvantage_group and attr2 in original_disadvantage_group:
            # Apply type-tolerant matching for both attributes
            matched_values = {}
            all_matched = True
            
            for attr in [attr1, attr2]:
                disadvantaged_value = original_disadvantage_group[attr]
                matching_value = None
                
                # Direct match first
                if disadvantaged_value in attr_values[attr]:
                    matching_value = disadvantaged_value
                else:
                    # Try string conversion
                    str_disadvantaged = str(disadvantaged_value)
                    for data_val in attr_values[attr]:
                        if str(data_val) == str_disadvantaged:
                            matching_value = data_val
                            break
                
                if matching_value is not None:
                    matched_values[attr] = matching_value
                else:
                    print(f"  WARNING: {attr} value {disadvantaged_value} not found in data {attr_values[attr]}")
                    all_matched = False
                    break
            
            if all_matched:
                group_name = f"pair_{attr1}_{attr2}"
                group_def = matched_values
                disadvantaged_groups[group_name] = group_def
                
                # Count samples with proper type handling
                query_parts = []
                for attr, val in group_def.items():
                    if isinstance(val, str):
                        query_parts.append(f"`{attr}` == '{val}'")
                    else:
                        query_parts.append(f"`{attr}` == {val}")
                query = " and ".join(query_parts)
                count = len(df_sample.query(query))
                print(f"  {group_name}: {group_def} -> {count} samples")
    
    # 3. Full intersection (original disadvantage group)
    print("\n3. Full intersection (original):")
    if len(protected_attrs) >= 3:
        # Apply type-tolerant matching for all attributes
        matched_values = {}
        all_matched = True
        
        for attr in protected_attrs:
            if attr in original_disadvantage_group:
                disadvantaged_value = original_disadvantage_group[attr]
                matching_value = None
                
                # Direct match first
                if disadvantaged_value in attr_values[attr]:
                    matching_value = disadvantaged_value
                else:
                    # Try string conversion
                    str_disadvantaged = str(disadvantaged_value)
                    for data_val in attr_values[attr]:
                        if str(data_val) == str_disadvantaged:
                            matching_value = data_val
                            break
                
                if matching_value is not None:
                    matched_values[attr] = matching_value
                else:
                    print(f"  WARNING: {attr} value {disadvantaged_value} not found in data {attr_values[attr]}")
                    all_matched = False
                    break
        
        if all_matched:
            group_name = "full_intersection"
            group_def = matched_values
            disadvantaged_groups[group_name] = group_def
            
            # Count samples with proper type handling
            query_parts = []
            for attr, val in group_def.items():
                if attr in df_sample.columns:
                    if isinstance(val, str):
                        query_parts.append(f"`{attr}` == '{val}'")
                    else:
                        query_parts.append(f"`{attr}` == {val}")
            query = " and ".join(query_parts)
            count = len(df_sample.query(query))
            print(f"  {group_name}: {group_def} -> {count} samples")
    
    print(f"\nTotal groups to analyze: {len(disadvantaged_groups)}")
    return disadvantaged_groups


def generate_all_counterfactuals(df_sample_str, protected_attrs, causal_model, categorical_info, X_cols, df_sample, target):
    """Generate counterfactuals for all individuals across all protected attributes."""
    
    all_individual_predictions = []
    
    # Get unique values for each protected attribute
    attr_unique_values = {}
    for attr in protected_attrs:
        unique_values = sorted(df_sample_str[attr].unique())
        attr_unique_values[attr] = unique_values
        print(f"  {attr}: {unique_values}")
    
    # Process each individual
    for idx, row in tqdm(df_sample_str.iterrows(), 
                       total=len(df_sample_str), 
                       desc="Generating counterfactuals"):
        try:
            observed_data_for_individual = pd.DataFrame([row])
            
            # Ensure all columns from the causal model are present
            scm_nodes = set(causal_model.graph.nodes())
            missing_scm_cols = scm_nodes - set(observed_data_for_individual.columns)
            
            if missing_scm_cols:
                print(f"Warning: Adding missing SCM columns for counterfactual generation: {missing_scm_cols}")
                for col in missing_scm_cols:
                    # Add default values for missing columns
                    observed_data_for_individual[col] = "0"  # Use string since this is df_sample_str
            
            individual_predictions = {
                'original_index': idx,
                'factual_values': dict(df_sample.loc[idx]),  # Store all column values as integers from original data
                'predictions_by_attr': {},
                'original_target': df_sample.loc[idx, target]
            }
            
            # For each protected attribute
            for attr in protected_attrs:
                attr_predictions = {}
                factual_value_str = row[attr]
                
                # Factual scenario
                intervention_factual = {attr: lambda x, val=factual_value_str: val}
                obs_samples = gcm.interventional_samples(
                    causal_model,
                    intervention_factual,
                    observed_data=observed_data_for_individual,
                )
                
                # Convert back to integers
                for col, info in categorical_info.items():
                    if col in obs_samples.columns:
                        obs_samples[col] = obs_samples[col].map(info['string_to_int']).fillna(info['fallback_value']).astype(int)
                
                # IMPROVEMENT: Validate and add missing X_cols for factual samples
                obs_samples, missing_cols = validate_and_add_missing_columns(
                    obs_samples, X_cols, df_sample, idx, f"factual samples for individual {idx}"
                )
                
                # Safe column selection for factual samples
                try:
                    available_cols = list(obs_samples.columns)
                    missing_x_cols = [col for col in X_cols if col not in available_cols]
                    
                    if missing_x_cols:
                        print(f"ERROR: Missing factual columns after validation: {missing_x_cols}")
                        print(f"Available columns: {available_cols}")
                        continue  # Skip this individual
                    
                    attr_predictions['factual'] = obs_samples[X_cols].copy()
                except Exception as e:
                    print(f"Error in factual column selection for individual {idx}: {str(e)}")
                    continue  # Skip this individual
                
                # Counterfactual scenarios for each possible value
                for cf_value_str in attr_unique_values[attr]:
                    if cf_value_str != factual_value_str:
                        intervention_cf = {attr: lambda x, val=cf_value_str: val}
                        cf_samples = gcm.interventional_samples(
                            causal_model,
                            intervention_cf,
                            observed_data=observed_data_for_individual,
                        )
                        
                        # Convert back to integers
                        for col, info in categorical_info.items():
                            if col in cf_samples.columns:
                                cf_samples[col] = cf_samples[col].map(info['string_to_int']).fillna(info['fallback_value']).astype(int)
                        
                        # IMPROVEMENT: Validate and add missing X_cols for counterfactual samples
                        cf_samples, missing_cols = validate_and_add_missing_columns(
                            cf_samples, X_cols, df_sample, idx, f"CF samples for individual {idx}, cf_value {cf_value_str}"
                        )
                        
                        # Safe column selection for counterfactual samples
                        try:
                            available_cols = list(cf_samples.columns)
                            missing_x_cols = [col for col in X_cols if col not in available_cols]
                            
                            if missing_x_cols:
                                print(f"ERROR: Missing CF columns after validation: {missing_x_cols}")
                                print(f"Available columns: {available_cols}")
                                continue  # Skip this counterfactual
                            
                            attr_predictions[f'cf_{cf_value_str}'] = cf_samples[X_cols].copy()
                        except Exception as e:
                            print(f"Error in CF column selection for individual {idx}, cf_value {cf_value_str}: {str(e)}")
                            continue  # Skip this counterfactual
                
                individual_predictions['predictions_by_attr'][attr] = attr_predictions
            
            all_individual_predictions.append(individual_predictions)
            
        except Exception as e:
            print(f"Error processing individual {idx}: {str(e)}")
            continue
    
    return all_individual_predictions


def train_classifier_on_factual_data(all_individual_predictions, estimator, random_state):
    """Train a single classifier on all factual data."""
    
    # Collect all factual data
    factual_features_list = []
    factual_targets = []
    
    for individual in all_individual_predictions:
        # Use factual data from the first protected attribute (they should be the same)
        first_attr = list(individual['predictions_by_attr'].keys())[0]
        factual_features = individual['predictions_by_attr'][first_attr]['factual']
        factual_features_list.append(factual_features)
        factual_targets.append(individual['original_target'])
    
    # Combine all factual data
    all_factual_features = pd.concat(factual_features_list).reset_index(drop=True)
    all_factual_targets = pd.Series(factual_targets)
    
    # Train classifier
    if estimator == RandomForestClassifier:
        clf = estimator(random_state=random_state, n_estimators=200, max_depth=10)
    elif estimator == LogisticRegression:
        clf = estimator(random_state=random_state, max_iter=5000, solver='saga')
    else:
        from sklearn.base import clone
        clf = clone(estimator)
    
    clf.fit(all_factual_features.astype(float), all_factual_targets)
    print(f"✅ Trained classifier on {len(all_factual_features)} factual samples")
    
    return clf


def analyze_group_fairness(group_definition, group_name, all_individual_predictions, df_sample, 
                          protected_attrs, classifier, min_samples, causal_model=None, 
                          categorical_info=None, X_cols=None, dag=None, target=None, calculate_pse=False,
                          pse_sample_size=None, max_paths=None, random_state=42,
                          cached_pse_paths=None, precomputed_pse=None):
    """Analyze fairness for a specific disadvantaged group."""
    
    # Filter individuals belonging to this group
    if group_definition:  # If not empty (i.e., not "all_individuals")
        group_indices = []
        for individual in all_individual_predictions:
            idx = individual['original_index']
            individual_data = df_sample.loc[idx]
            
            # Check if individual belongs to this group
            belongs_to_group = all(individual_data[attr] == val for attr, val in group_definition.items() if attr in df_sample.columns)
            if belongs_to_group:
                group_indices.append(individual)
        
        filtered_individuals = group_indices
    else:
        # All individuals (no filtering)
        filtered_individuals = all_individual_predictions
    
    print(f"  Group '{group_name}' has {len(filtered_individuals)} individuals")
    
    if len(filtered_individuals) < min_samples:
        print(f"  ⚠️  Too few samples ({len(filtered_individuals)} < {min_samples}), skipping...")
        return {
            'fairness_score': np.nan,
            'error': f'Insufficient samples ({len(filtered_individuals)} < {min_samples})',
            'samples_in_group': len(filtered_individuals)
        }
    
    # Analyze fairness for each protected attribute within this group
    attr_fairness_results = {}
    all_attr_scores = []
    
    # NEW: Individual counterfactual fairness metrics across all attributes
    all_macd_scores = []
    all_max_gap_scores = []
    
    for attr in protected_attrs:
        print(f"    Analyzing {attr} for group {group_name}...")
        
        attr_result = analyze_attribute_fairness_for_group(
            attr, filtered_individuals, classifier, group_name
        )
        
        attr_fairness_results[attr] = attr_result
        
        if 'fairness_score' in attr_result and not pd.isna(attr_result['fairness_score']):
            all_attr_scores.append(abs(attr_result['fairness_score']))
            
            # NEW: Collect individual fairness metrics
            if 'macd' in attr_result and not pd.isna(attr_result['macd']):
                all_macd_scores.append(attr_result['macd'])
            if 'max_counterfactual_gap' in attr_result and not pd.isna(attr_result['max_counterfactual_gap']):
                all_max_gap_scores.append(attr_result['max_counterfactual_gap'])
            
            print(f"      {attr}: fairness score = {attr_result['fairness_score']:.6f}")
            print(f"        Mean observed: {attr_result.get('mean_observed', 'N/A'):.6f}")
            print(f"        Mean counterfactual: {attr_result.get('mean_counterfactual', 'N/A'):.6f}")
            print(f"        Binary fairness score: {attr_result.get('binary_fairness_score', 'N/A'):.6f}")
            # NEW: Print individual fairness metrics
            print(f"        MACD: {attr_result.get('macd', 'N/A'):.6f}")
            print(f"        Max Counterfactual Gap: {attr_result.get('max_counterfactual_gap', 'N/A'):.6f}")
        else:
            error_msg = attr_result.get('error', 'No valid fairness comparison generated') if isinstance(attr_result, dict) else 'Invalid attribute result payload'
            print(f"      {attr}: skipped ({error_msg})")
    
    # NEW: Calculate Interventional Fairness for this group
    interventional_results = {}
    if causal_model and categorical_info and X_cols:
        print(f"    Calculating Interventional Fairness for group {group_name}...")
        interventional_results = calculate_interventional_fairness(
            filtered_individuals, classifier, protected_attrs, causal_model, 
            categorical_info, X_cols, group_definition=group_definition
        )
    
    # NEW: Calculate Natural Indirect Effect (NIE) for this group
    nie_results = {}
    if causal_model and categorical_info and X_cols:
        print(f"    Calculating Natural Indirect Effect (NIE) for group {group_name}...")
        for attr in protected_attrs:
            try:
                # Get DAG edges for mediator inference
                dag_edges = dag if isinstance(dag, list) else list(dag.edges()) if hasattr(dag, 'edges') else list(dag) if dag else None
                
                nie_attr_result = calculate_natural_indirect_effect(
                    filtered_individuals, classifier, attr, causal_model,
                    categorical_info, X_cols, mediator_vars=None, dag_edges=dag_edges,
                    target_var=target  # Pass the actual target variable name
                )
                nie_results[attr] = nie_attr_result
                
                if 'nie_score' in nie_attr_result and not pd.isna(nie_attr_result['nie_score']):
                    print(f"      {attr} NIE score: {nie_attr_result['nie_score']:.6f}")
                    if 'nde_score' in nie_attr_result and not pd.isna(nie_attr_result['nde_score']):
                        print(f"      {attr} NDE score: {nie_attr_result['nde_score']:.6f}")
                    if 'total_effect' in nie_attr_result and not pd.isna(nie_attr_result['total_effect']):
                        print(f"      {attr} TE score: {nie_attr_result['total_effect']:.6f}")
                
            except Exception as e:
                print(f"      Error calculating NIE for {attr}: {str(e)}")
                nie_results[attr] = {'nie_score': np.nan, 'error': str(e)}
    
    # NEW: Calculate Path-Specific Effect (PSE) for this group (optional)
    pse_results = {}
    if calculate_pse and causal_model and categorical_info and X_cols:
        print(f"    Calculating Path-Specific Effect (PSE) for group {group_name}...")
        for attr in protected_attrs:
            try:
                if precomputed_pse and attr in precomputed_pse:
                    pse_attr_result = aggregate_precomputed_pse_for_group(
                        precomputed_pse[attr], filtered_individuals
                    )
                    pse_results[attr] = pse_attr_result

                    if isinstance(pse_attr_result, dict):
                        for path_name, path_result in pse_attr_result.items():
                            if isinstance(path_result, dict) and 'pse_score' in path_result:
                                if not pd.isna(path_result['pse_score']):
                                    print(f"      {attr} {path_name} PSE score: {path_result['pse_score']:.6f} (precomputed)")
                    continue

                # Reuse pre-computed path sets where available.
                specific_paths = None
                if cached_pse_paths and attr in cached_pse_paths:
                    specific_paths = cached_pse_paths[attr]

                dag_edges = None
                if specific_paths is None:
                    dag_edges = dag if isinstance(dag, list) else list(dag.edges()) if hasattr(dag, 'edges') else list(dag) if dag else None
                
                pse_attr_result = calculate_path_specific_effect(
                    filtered_individuals, classifier, attr, causal_model,
                    categorical_info, X_cols, specific_paths=specific_paths, dag_edges=dag_edges,
                    target_var=target,
                    pse_sample_size=pse_sample_size,
                    max_paths=max_paths,
                    random_state=random_state
                )
                pse_results[attr] = pse_attr_result
                
                # Print PSE results for each path
                if isinstance(pse_attr_result, dict):
                    for path_name, path_result in pse_attr_result.items():
                        if isinstance(path_result, dict) and 'pse_score' in path_result:
                            if not pd.isna(path_result['pse_score']):
                                print(f"      {attr} {path_name} PSE score: {path_result['pse_score']:.6f}")
                
            except Exception as e:
                print(f"      Error calculating PSE for {attr}: {str(e)}")
                pse_results[attr] = {'pse_score': np.nan, 'error': str(e)}
    elif not calculate_pse:
        print(f"    Skipping Path-Specific Effect (PSE) calculation (disabled) for group {group_name}...")
    else:
        print(f"    Skipping Path-Specific Effect (PSE) calculation (missing requirements) for group {group_name}...")
    
    # NEW: Calculate Individual Treatment Effect (ITE) for this group
    ite_results = {}
    if filtered_individuals:
        print(f"    Calculating Individual Treatment Effect (ITE) for group {group_name}...")
        for attr in protected_attrs:
            try:
                ite_attr_result = calculate_individual_treatment_effect(
                    filtered_individuals, classifier, attr, df_sample
                )
                ite_results[attr] = ite_attr_result
                
                if 'ite_mean' in ite_attr_result and not pd.isna(ite_attr_result['ite_mean']):
                    print(f"      {attr} ITE mean: {ite_attr_result['ite_mean']:.6f}")
                    
                    # Fix formatting for potentially NaN values
                    ite_std = ite_attr_result.get('ite_std', np.nan)
                    if not pd.isna(ite_std):
                        print(f"      {attr} ITE std: {ite_std:.6f}")
                    else:
                        print(f"      {attr} ITE std: N/A")
                    
                    ite_min = ite_attr_result.get('ite_min', np.nan)
                    ite_max = ite_attr_result.get('ite_max', np.nan)
                    min_str = f"{ite_min:.6f}" if not pd.isna(ite_min) else "N/A"
                    max_str = f"{ite_max:.6f}" if not pd.isna(ite_max) else "N/A"
                    print(f"      {attr} ITE range: [{min_str}, {max_str}]")
                    
                    pct_positive = ite_attr_result.get('pct_positive_effects', np.nan)
                    if not pd.isna(pct_positive):
                        print(f"      {attr} % positive effects: {pct_positive:.1f}%")
                    else:
                        print(f"      {attr} % positive effects: N/A")
                
                # Optional: Analyze ITE heterogeneity if we have demographic data
                if df_sample is not None and len(df_sample) > 0:
                    try:
                        heterogeneity_result = analyze_ite_heterogeneity(
                            ite_attr_result, filtered_individuals, df_sample
                        )
                        ite_attr_result['heterogeneity_analysis'] = heterogeneity_result
                        
                        # Print brief heterogeneity summary
                        if heterogeneity_result and 'error' not in heterogeneity_result:
                            n_vars_analyzed = len([k for k, v in heterogeneity_result.items() 
                                                 if isinstance(v, dict) and 'heterogeneity_ratio' in v])
                            if n_vars_analyzed > 0:
                                print(f"      {attr} heterogeneity analysis: {n_vars_analyzed} demographic variables analyzed")
                                
                    except Exception as het_e:
                        print(f"      Warning: ITE heterogeneity analysis failed for {attr}: {str(het_e)}")
                
            except Exception as e:
                print(f"      Error calculating ITE for {attr}: {str(e)}")
                ite_results[attr] = {'ite_mean': np.nan, 'error': str(e)}
    
    # Compile group results
    if all_attr_scores:
        group_result = {
            'group_definition': group_definition,
            'samples_in_group': len(filtered_individuals),
            'attributes_analyzed': attr_fairness_results,
            'max_fairness_score': max(all_attr_scores),  # Worst across all attributes
            'mean_fairness_score': np.mean(all_attr_scores),  # Average across attributes
            'successful_attributes': len(all_attr_scores),
            # NEW: Individual counterfactual fairness metrics
            'mean_macd': np.mean(all_macd_scores) if all_macd_scores else np.nan,
            'max_macd': np.max(all_macd_scores) if all_macd_scores else np.nan,
            'mean_max_gap': np.mean(all_max_gap_scores) if all_max_gap_scores else np.nan,
            'overall_max_gap': np.max(all_max_gap_scores) if all_max_gap_scores else np.nan,
            # NEW: Interventional fairness results
            'interventional_fairness': interventional_results,
            # NEW: Natural Indirect Effect results
            'nie_results': nie_results,
            # NEW: Path-Specific Effect results
            'pse_results': pse_results,
            # NEW: Individual Treatment Effect results
            'ite_results': ite_results
        }
        print(f"  ✅ Group '{group_name}': max fairness score = {max(all_attr_scores):.6f}")
        # NEW: Print individual fairness summary
        if all_macd_scores:
            print(f"      Mean MACD: {np.mean(all_macd_scores):.6f}")
        if all_max_gap_scores:
            print(f"      Overall Max Gap: {np.max(all_max_gap_scores):.6f}")
        # NEW: Print NIE and PSE summary
        if nie_results:
            avg_nie_scores_signed = []
            for attr, nie_result in nie_results.items():
                if isinstance(nie_result, dict) and 'nie_score' in nie_result:
                    if not pd.isna(nie_result['nie_score']):
                        avg_nie_scores_signed.append(nie_result['nie_score'])
            if avg_nie_scores_signed:
                print(f"      Average NIE Score (signed): {np.mean(avg_nie_scores_signed):.6f}")
                print(f"      Average NIE Score (abs): {np.mean(np.abs(avg_nie_scores_signed)):.6f}")
        if pse_results:
            avg_pse_scores_signed = []
            for attr, attr_pse_results in pse_results.items():
                if isinstance(attr_pse_results, dict):
                    for path_name, pse_result in attr_pse_results.items():
                        if isinstance(pse_result, dict) and 'pse_score' in pse_result:
                            if not pd.isna(pse_result['pse_score']):
                                avg_pse_scores_signed.append(pse_result['pse_score'])
            if avg_pse_scores_signed:
                print(f"      Average PSE Score (signed): {np.mean(avg_pse_scores_signed):.6f}")
                print(f"      Average PSE Score (abs): {np.mean(np.abs(avg_pse_scores_signed)):.6f}")
        # NEW: Print ITE summary
        if ite_results:
            avg_ite_means_signed = []
            avg_ite_stds = []
            for attr, ite_result in ite_results.items():
                if isinstance(ite_result, dict) and 'ite_mean' in ite_result:
                    if not pd.isna(ite_result['ite_mean']):
                        avg_ite_means_signed.append(ite_result['ite_mean'])
                    if 'ite_std' in ite_result and not pd.isna(ite_result['ite_std']):
                        avg_ite_stds.append(ite_result['ite_std'])
            if avg_ite_means_signed:
                print(f"      Average ITE Mean (signed): {np.mean(avg_ite_means_signed):.6f}")
                print(f"      Average ITE Mean (abs): {np.mean(np.abs(avg_ite_means_signed)):.6f}")
            if avg_ite_stds:
                print(f"      Average ITE Heterogeneity (Std): {np.mean(avg_ite_stds):.6f}")
    else:
        group_result = {
            'group_definition': group_definition,
            'samples_in_group': len(filtered_individuals),
            'attributes_analyzed': attr_fairness_results,
            'max_fairness_score': np.nan,
            'error': 'No successful attribute analyses',
            # NEW: Default values for new metrics
            'mean_macd': np.nan,
            'max_macd': np.nan,
            'mean_max_gap': np.nan,
            'overall_max_gap': np.nan,
            'interventional_fairness': {},
            # NEW: Default NIE and PSE results
            'nie_results': nie_results if 'nie_results' in locals() else {},
            'pse_results': pse_results if 'pse_results' in locals() else {},
            # NEW: Default ITE results
            'ite_results': ite_results if 'ite_results' in locals() else {}
        }
        print(f"  ❌ Group '{group_name}': no successful analyses")
    
    return group_result



def analyze_attribute_fairness_for_group(attr, filtered_individuals, classifier, group_name):
    """Analyze fairness for a specific attribute within a specific group."""
    
    # Collect predictions for this attribute
    factual_preds = []
    counterfactual_preds = {}
    
    for individual in filtered_individuals:
        if attr not in individual['predictions_by_attr']:
            continue
            
        attr_predictions = individual['predictions_by_attr'][attr]
        
        # Factual prediction
        factual_features = attr_predictions['factual']
        factual_pred = classifier.predict_proba(factual_features.astype(float))[0, 1]
        factual_preds.append(factual_pred)
        
        # Counterfactual predictions
        for cf_key, cf_features in attr_predictions.items():
            if cf_key.startswith('cf_'):
                cf_value = cf_key.replace('cf_', '')
                if cf_value not in counterfactual_preds:
                    counterfactual_preds[cf_value] = []
                
                cf_pred = classifier.predict_proba(cf_features.astype(float))[0, 1]
                counterfactual_preds[cf_value].append(cf_pred)
    
    if not factual_preds:
        return {
            'fairness_score': np.nan,
            'error': f'No predictions generated for {attr}'
        }
    
    # Calculate fairness scores for each counterfactual value
    pairwise_results = {}
    
    # NEW: Individual counterfactual differences for MACD and Max Gap
    all_individual_differences = []
    
    for cf_value, cf_preds in counterfactual_preds.items():
        if len(cf_preds) == len(factual_preds):
            mean_factual = np.mean(factual_preds)
            mean_cf = np.mean(cf_preds)
            fairness_score = mean_factual - mean_cf
            
            # Add binary predictions
            binary_factual = [1 if p >= 0.5 else 0 for p in factual_preds]
            binary_cf = [1 if p >= 0.5 else 0 for p in cf_preds]
            binary_mean_factual = np.mean(binary_factual)
            binary_mean_cf = np.mean(binary_cf)
            binary_fairness_score = binary_mean_factual - binary_mean_cf
            
            # NEW: Calculate individual-level differences
            individual_differences = [abs(f - c) for f, c in zip(factual_preds, cf_preds)]
            all_individual_differences.extend(individual_differences)
            
            # NEW: Individual counterfactual fairness metrics
            macd = np.mean(individual_differences)  # Mean Absolute Counterfactual Difference
            max_gap = np.max(individual_differences)  # Maximum Counterfactual Gap
            
            pairwise_results[f'factual_vs_{cf_value}'] = {
                'mean_factual': mean_factual,
                'mean_counterfactual': mean_cf,
                'fairness_score': fairness_score,
                'binary_mean_factual': binary_mean_factual,
                'binary_mean_counterfactual': binary_mean_cf,
                'binary_fairness_score': binary_fairness_score,
                'samples_analyzed': len(factual_preds),
                # NEW: Individual counterfactual fairness metrics
                'macd': macd,
                'max_counterfactual_gap': max_gap,
                'individual_differences': individual_differences
            }

    # Instead of taking max, return the primary comparison (usually the first one)
    if pairwise_results:
        # Get the primary comparison result
        primary_comparison = list(pairwise_results.values())[0]
        primary_key = list(pairwise_results.keys())[0]
        
        # NEW: Overall individual fairness metrics across all comparisons
        overall_macd = np.mean(all_individual_differences) if all_individual_differences else np.nan
        overall_max_gap = np.max(all_individual_differences) if all_individual_differences else np.nan
        
        return {
            'fairness_score': primary_comparison['fairness_score'],
            'mean_observed': primary_comparison['mean_factual'],
            'mean_counterfactual': primary_comparison['mean_counterfactual'],
            'binary_mean_observed': primary_comparison['binary_mean_factual'],
            'binary_mean_counterfactual': primary_comparison['binary_mean_counterfactual'],
            'binary_fairness_score': primary_comparison['binary_fairness_score'],
            'primary_comparison': primary_key,
            'all_comparisons': pairwise_results,
            'samples_analyzed': primary_comparison['samples_analyzed'],
            # NEW: Individual counterfactual fairness metrics
            'macd': primary_comparison['macd'],
            'max_counterfactual_gap': primary_comparison['max_counterfactual_gap'],
            'overall_macd': overall_macd,
            'overall_max_gap': overall_max_gap
        }
    else:
        return {
            'fairness_score': np.nan,
            'error': f'No valid comparisons for {attr}'
        }

def calculate_interventional_fairness(all_individual_predictions, classifier, protected_attrs, causal_model, 
                                     categorical_info, X_cols, group_definition=None, k_values=[1, 2]):
    """
    Calculate Interventional (K-) Fairness metrics.
    
    Args:
        all_individual_predictions: List of individual prediction data
        classifier: Trained classifier
        protected_attrs: List of protected attribute names
        causal_model: Fitted causal model
        categorical_info: Categorical variable mapping info
        X_cols: Feature column names
        group_definition: Dict defining the group (e.g., {'sex': 1})
        k_values: List of k values to test for k-fairness (default: [1, 2])
    
    Returns:
        dict: Interventional fairness results
    """
    
    # 🟡 CONFIGURABLE: Make violation threshold adaptable in future versions
    violation_threshold = 0.1  # Default threshold for counting violations
    
    # NEW: Determine which protected attributes are "free" to intervene on
    if group_definition:
        # Attributes that are fixed by the group definition cannot be intervened on
        fixed_attrs = set(group_definition.keys())
        free_attrs = [attr for attr in protected_attrs if attr not in fixed_attrs]
    else:
        # If no group definition, all protected attributes are free
        free_attrs = protected_attrs.copy()
    
    print(f"\nCalculating Interventional (K-) Fairness:")
    print(f"  Protected attributes: {protected_attrs}")
    if group_definition:
        print(f"  Fixed by group definition: {list(group_definition.keys())}")
    print(f"  Free to intervene on: {free_attrs}")
    
    if not free_attrs:
        print("  ⚠️  No free attributes to intervene on - skipping interventional fairness")
        return {}
    
    # NEW: Adjust k_values based on available free attributes
    max_possible_k = len(free_attrs)
    valid_k_values = [k for k in k_values if k <= max_possible_k]
    
    if len(valid_k_values) != len(k_values):
        print(f"  ⚠️  Adjusted K values from {k_values} to {valid_k_values} (max possible K = {max_possible_k})")
    
    if not valid_k_values:
        print("  ⚠️  No valid K values - skipping interventional fairness")
        return {}
    
    print(f"  Testing K values: {valid_k_values}")
    
    interventional_results = {}
    
    for k in valid_k_values:
        print(f"  Calculating {k}-fairness...")
        
        k_fairness_violations = []
        total_comparisons = 0
        
        # For each individual
        for individual_idx, individual in enumerate(all_individual_predictions):
            try:
                # Get factual prediction for this individual
                first_attr = list(individual['predictions_by_attr'].keys())[0]
                factual_features = individual['predictions_by_attr'][first_attr]['factual']
                factual_pred = classifier.predict_proba(factual_features.astype(float))[0, 1]
                
                # Generate k random interventions on FREE protected attributes only
                for intervention_attempt in range(k):
                    if not free_attrs:
                        break
                        
                    # NEW: Randomly select from FREE attributes only
                    attr_to_intervene = np.random.choice(free_attrs)
                    
                    # Get possible values for this attribute
                    if attr_to_intervene in categorical_info:
                        possible_values = list(categorical_info[attr_to_intervene]['string_to_int'].keys())
                        # Remove current value (ensure string comparison)
                        current_value = str(individual['factual_values'][attr_to_intervene])
                        possible_values = [v for v in possible_values if v != current_value]
                        
                        if possible_values:
                            # Randomly select a counterfactual value
                            cf_value = np.random.choice(possible_values)
                            
                            # Generate counterfactual prediction
                            cf_key = f'cf_{cf_value}'
                            if cf_key in individual['predictions_by_attr'][attr_to_intervene]:
                                cf_features = individual['predictions_by_attr'][attr_to_intervene][cf_key]
                                cf_pred = classifier.predict_proba(cf_features.astype(float))[0, 1]
                                
                                # Calculate difference
                                difference = abs(factual_pred - cf_pred)
                                k_fairness_violations.append(difference)
                                total_comparisons += 1
                            
            except Exception as e:
                print(f"    Error processing individual for {k}-fairness: {str(e)}")
                continue
        
        # Calculate k-fairness metrics
        if k_fairness_violations:
            mean_violation = np.mean(k_fairness_violations)
            max_violation = np.max(k_fairness_violations)
            # 🟡 IMPROVED: Use configurable threshold instead of hard-coded 0.1
            violation_rate = np.mean([v > violation_threshold for v in k_fairness_violations])  # Rate of violations > threshold
            
            interventional_results[f'{k}_fairness'] = {
                'mean_violation': mean_violation,
                'max_violation': max_violation,
                'violation_rate': violation_rate,
                'total_comparisons': total_comparisons,
                'violations': k_fairness_violations,
                'free_attributes_used': free_attrs.copy(),  # Track which attributes were available
                'k_value': k
            }
            
            print(f"    {k}-fairness mean violation: {mean_violation:.6f}")
            print(f"    {k}-fairness max violation: {max_violation:.6f}")
            print(f"    {k}-fairness violation rate (>{violation_threshold}): {violation_rate:.6f}")
        else:
            print(f"    ⚠️  No {k}-fairness violations found (total comparisons: {total_comparisons})")
            interventional_results[f'{k}_fairness'] = {
                'mean_violation': np.nan,
                'max_violation': np.nan,
                'violation_rate': np.nan,
                'total_comparisons': total_comparisons,
                'error': 'No valid interventional comparisons',
                'free_attributes_used': free_attrs.copy(),
                'k_value': k
            }
    
    return interventional_results

def compile_overall_results(group_results, all_fairness_scores, protected_attrs):
    """Compile overall results across all groups."""
    
    if not group_results:
        return {
            'fairness_score': np.nan,
            'error': 'No group results to compile'
        }
    
    # Extract successful groups - handle both dict and other formats
    if isinstance(group_results, dict):
        successful_groups = []
        all_calculated_fairness_scores = []
        
        for group_name, group_result in group_results.items():
            print(f"  Processing group: {group_name}")
            print(f"    Group result keys: {list(group_result.keys()) if isinstance(group_result, dict) else 'Not a dict'}")
            
            if isinstance(group_result, dict) and 'max_fairness_score' in group_result:
                fairness_score = group_result['max_fairness_score']
                print(f"    Fairness score: {fairness_score}")
                
                if not pd.isna(fairness_score):
                    successful_groups.append(group_result)
                    all_calculated_fairness_scores.append(abs(fairness_score))
                    print(f"    ✅ Added to successful groups")
                else:
                    print(f"    ❌ Fairness score is NaN")
            else:
                print(f"    ❌ No max_fairness_score found")
    else:
        print(f"  ❌ group_results is not a dictionary")
        return {
            'fairness_score': np.nan,
            'error': 'Invalid group_results format'
        }
    
    print(f"  Found {len(successful_groups)} successful groups")
    print(f"  Calculated fairness scores: {all_calculated_fairness_scores}")
    
    # Use the calculated scores if all_fairness_scores is empty
    fairness_scores_to_use = all_fairness_scores if all_fairness_scores else all_calculated_fairness_scores
    
    if not successful_groups or not fairness_scores_to_use:
        print(f"  ❌ No successful analyses found")
        return {
            'fairness_score': np.nan,
            'error': 'No successful group analyses',
            'group_results': group_results,
            'groups_analyzed': len(group_results),
            'successful_analyses': 0,
            'analysis_type': 'multiple_group_sequential_analysis'
        }
    
    # Calculate overall metrics
    max_fairness_scores = [abs(r['max_fairness_score']) for r in successful_groups]
    # 🔴 CRITICAL FIX: Take absolute values to prevent cancellation of opposite-direction bias
    mean_fairness_scores = [abs(r['mean_fairness_score']) for r in successful_groups 
                           if 'mean_fairness_score' in r and not pd.isna(r['mean_fairness_score'])]
    
    worst_fairness_score = max(max_fairness_scores)
    worst_group_idx = max_fairness_scores.index(worst_fairness_score)
    
    # Find worst group name
    worst_group = None
    for i, (group_name, group_result) in enumerate(group_results.items()):
        if isinstance(group_result, dict) and 'max_fairness_score' in group_result:
            if not pd.isna(group_result['max_fairness_score']) and abs(group_result['max_fairness_score']) == worst_fairness_score:
                worst_group = group_name
                break
    
    # Individual counterfactual fairness metrics
    all_macd_scores = []
    all_max_gap_scores = []
    
    for group_result in successful_groups:
        if 'mean_macd' in group_result and not pd.isna(group_result.get('mean_macd')):
            all_macd_scores.append(group_result['mean_macd'])
        if 'overall_max_gap' in group_result and not pd.isna(group_result.get('overall_max_gap')):
            all_max_gap_scores.append(group_result['overall_max_gap'])
    
    # Interventional fairness metrics
    all_1_fairness_violations = []
    all_2_fairness_violations = []
    
    for group_result in successful_groups:
        if 'interventional_fairness' in group_result:
            interventional = group_result['interventional_fairness']
            if isinstance(interventional, dict):
                if '1_fairness' in interventional:
                    result = interventional['1_fairness']
                    if isinstance(result, dict) and 'mean_violation' in result and not pd.isna(result['mean_violation']):
                        all_1_fairness_violations.append(result['mean_violation'])
                
                if '2_fairness' in interventional:
                    result = interventional['2_fairness']
                    if isinstance(result, dict) and 'mean_violation' in result and not pd.isna(result['mean_violation']):
                        all_2_fairness_violations.append(result['mean_violation'])
    
    # NEW: Natural Indirect Effect (NIE) metrics
    all_nie_scores = []
    all_nde_scores = []
    all_te_scores = []
    
    for group_result in successful_groups:
        if 'nie_results' in group_result and isinstance(group_result['nie_results'], dict):
            for attr, nie_result in group_result['nie_results'].items():
                if isinstance(nie_result, dict):
                    if 'nie_score' in nie_result and not pd.isna(nie_result['nie_score']):
                        all_nie_scores.append(nie_result['nie_score'])
                    if 'nde_score' in nie_result and not pd.isna(nie_result['nde_score']):
                        all_nde_scores.append(nie_result['nde_score'])
                    if 'total_effect' in nie_result and not pd.isna(nie_result['total_effect']):
                        all_te_scores.append(nie_result['total_effect'])
    
    # NEW: Path-Specific Effect (PSE) metrics
    all_pse_scores = []
    pse_path_results = {}
    
    for group_result in successful_groups:
        if 'pse_results' in group_result and isinstance(group_result['pse_results'], dict):
            for attr, attr_pse_results in group_result['pse_results'].items():
                if isinstance(attr_pse_results, dict):
                    for path_name, pse_result in attr_pse_results.items():
                        if isinstance(pse_result, dict) and 'pse_score' in pse_result:
                            if not pd.isna(pse_result['pse_score']):
                                all_pse_scores.append(pse_result['pse_score'])
                                # Store path-specific results
                                if path_name not in pse_path_results:
                                    pse_path_results[path_name] = []
                                pse_path_results[path_name].append(pse_result['pse_score'])
    
    # NEW: Stratified ITE aggregation following recommended approach
    all_individual_ite_values = []  # THE KEY COLLECTION for overall_ite_mean
    all_group_ite_disparities = []
    all_beneficial_rate_disparities = []
    all_harmful_rate_disparities = []
    ite_by_attribute = {}

    for group_result in successful_groups:
        if 'ite_results' in group_result and isinstance(group_result['ite_results'], dict):
            for attr, ite_result in group_result['ite_results'].items():
                if isinstance(ite_result, dict):
                    # Collect ALL individual ITE values for proper overall calculation
                    if 'raw_ite_values' in ite_result and ite_result['raw_ite_values']:
                        all_individual_ite_values.extend(ite_result['raw_ite_values'])
                    
                    # Collect fairness metrics if available
                    if 'fairness_metrics' in ite_result and ite_result['fairness_metrics']:
                        fairness = ite_result['fairness_metrics']
                        if 'ite_disparity' in fairness and not pd.isna(fairness['ite_disparity']):
                            all_group_ite_disparities.append(fairness['ite_disparity'])
                        if 'beneficial_rate_disparity' in fairness and not pd.isna(fairness['beneficial_rate_disparity']):
                            all_beneficial_rate_disparities.append(fairness['beneficial_rate_disparity'])
                        if 'harmful_rate_disparity' in fairness and not pd.isna(fairness['harmful_rate_disparity']):
                            all_harmful_rate_disparities.append(fairness['harmful_rate_disparity'])
                    
                    # Store attribute-specific results
                    if attr not in ite_by_attribute:
                        ite_by_attribute[attr] = {
                            'effect_sizes': [], 'disparities': [], 'group_means': []
                        }
                    
                    if 'ite_mean' in ite_result and not pd.isna(ite_result['ite_mean']):
                        ite_by_attribute[attr]['effect_sizes'].append(ite_result['ite_mean'])
                    
                    # Collect group-level means for attribute analysis
                    if 'group_results' in ite_result:
                        for group_name, group_stats in ite_result['group_results'].items():
                            if 'mean_ite' in group_stats:
                                ite_by_attribute[attr]['group_means'].append(group_stats['mean_ite'])
    
    overall_results = {
        'fairness_score': worst_fairness_score,
        'mean_fairness_score': np.mean(mean_fairness_scores) if mean_fairness_scores else np.nan,
        'worst_group': worst_group,
        'group_results': group_results,
        'groups_analyzed': len(group_results),
        'successful_analyses': len(successful_groups),
        'analysis_type': 'multiple_group_sequential_analysis',
        # Individual counterfactual fairness metrics
        'overall_mean_macd': np.mean(all_macd_scores) if all_macd_scores else np.nan,
        'overall_max_macd': np.max(all_macd_scores) if all_macd_scores else np.nan,
        'overall_mean_max_gap': np.mean(all_max_gap_scores) if all_max_gap_scores else np.nan,
        'overall_worst_max_gap': np.max(all_max_gap_scores) if all_max_gap_scores else np.nan,
        # Interventional fairness metrics
        'overall_1_fairness_mean': np.mean(all_1_fairness_violations) if all_1_fairness_violations else np.nan,
        'overall_1_fairness_max': np.max(all_1_fairness_violations) if all_1_fairness_violations else np.nan,
        'overall_2_fairness_mean': np.mean(all_2_fairness_violations) if all_2_fairness_violations else np.nan,
        'overall_2_fairness_max': np.max(all_2_fairness_violations) if all_2_fairness_violations else np.nan,
        # Natural Indirect Effect (NIE), Natural Direct Effect (NDE/DE), Total Effect (TE)
        'overall_nie_mean': np.mean(all_nie_scores) if all_nie_scores else np.nan,
        'overall_nie_max': np.max(all_nie_scores) if all_nie_scores else np.nan,
        'overall_nie_min': np.min(all_nie_scores) if all_nie_scores else np.nan,
        'overall_nie_std': np.std(all_nie_scores) if all_nie_scores else np.nan,
        'overall_nie_mean_abs': np.mean(np.abs(all_nie_scores)) if all_nie_scores else np.nan,
        'overall_nde_mean': np.mean(all_nde_scores) if all_nde_scores else np.nan,
        'overall_nde_max': np.max(all_nde_scores) if all_nde_scores else np.nan,
        'overall_nde_min': np.min(all_nde_scores) if all_nde_scores else np.nan,
        'overall_nde_std': np.std(all_nde_scores) if all_nde_scores else np.nan,
        'overall_nde_mean_abs': np.mean(np.abs(all_nde_scores)) if all_nde_scores else np.nan,
        # DE alias for users/reports that use DE terminology.
        'overall_de_mean': np.mean(all_nde_scores) if all_nde_scores else np.nan,
        'overall_de_max': np.max(all_nde_scores) if all_nde_scores else np.nan,
        'overall_de_min': np.min(all_nde_scores) if all_nde_scores else np.nan,
        'overall_de_std': np.std(all_nde_scores) if all_nde_scores else np.nan,
        'overall_te_mean': np.mean(all_te_scores) if all_te_scores else np.nan,
        'overall_te_max': np.max(all_te_scores) if all_te_scores else np.nan,
        'overall_te_min': np.min(all_te_scores) if all_te_scores else np.nan,
        'overall_te_std': np.std(all_te_scores) if all_te_scores else np.nan,
        # NEW: Path-Specific Effect (PSE) metrics
        'overall_pse_mean': np.mean(all_pse_scores) if all_pse_scores else np.nan,
        'overall_pse_max': np.max(all_pse_scores) if all_pse_scores else np.nan,
        'overall_pse_min': np.min(all_pse_scores) if all_pse_scores else np.nan,
        'overall_pse_std': np.std(all_pse_scores) if all_pse_scores else np.nan,
        'overall_pse_mean_abs': np.mean(np.abs(all_pse_scores)) if all_pse_scores else np.nan,
        'pse_by_path': {path: {'mean': np.mean(scores), 'max': np.max(scores), 'min': np.min(scores)} 
                       for path, scores in pse_path_results.items() if scores},
        'n_nie_calculations': len(all_nie_scores),
        'n_pse_calculations': len(all_pse_scores),
    }

    # Calculate the RECOMMENDED overall_ite_mean using ALL individual values
    if all_individual_ite_values:
        signed_ite_values = np.array(all_individual_ite_values)
        abs_ite_values = np.abs(signed_ite_values)
        overall_ite_mean_signed = np.mean(signed_ite_values)
        overall_ite_mean_abs = np.mean(abs_ite_values)
        
        # Additional overall metrics
        overall_results.update({
            # MAIN METRIC: Recommended aggregation approach
            'overall_ite_mean': overall_ite_mean_signed,
            'overall_ite_mean_signed': overall_ite_mean_signed,
            'overall_ite_mean_abs': overall_ite_mean_abs,
            'overall_ite_median': np.median(signed_ite_values),
            'overall_ite_std': np.std(signed_ite_values),
            'overall_ite_max': np.max(signed_ite_values),
            'overall_ite_min': np.min(signed_ite_values),
            'overall_ite_median_abs': np.median(abs_ite_values),
            'overall_ite_std_abs': np.std(abs_ite_values),
            'overall_ite_max_abs': np.max(abs_ite_values),
            # Backward-compatible aliases used by current reporting/CSV code.
            'overall_ite_median_mean': np.median(signed_ite_values),
            'overall_ite_heterogeneity_mean': np.std(signed_ite_values),
            
            # Fairness-specific metrics
            'overall_ite_disparity_max': max(all_group_ite_disparities) if all_group_ite_disparities else np.nan,
            'overall_ite_disparity_mean': np.mean(all_group_ite_disparities) if all_group_ite_disparities else np.nan,
            'overall_beneficial_disparity_max': max(all_beneficial_rate_disparities) if all_beneficial_rate_disparities else np.nan,
            'overall_harmful_disparity_max': max(all_harmful_rate_disparities) if all_harmful_rate_disparities else np.nan,
            
            # Direction analysis (preserving signs)
            'overall_pct_positive_effects': np.mean([ite > 0 for ite in all_individual_ite_values]) * 100,
            'overall_pct_negative_effects': np.mean([ite < 0 for ite in all_individual_ite_values]) * 100,
            
            # Metadata
            'n_individual_ite_values': len(all_individual_ite_values),
            'n_ite_calculations': len(all_individual_ite_values),  # For backward compatibility
            
            # Detailed by-attribute analysis
            'ite_by_attribute': {
                attr: {
                    'mean_effect_size': np.mean(stats['effect_sizes']) if stats['effect_sizes'] else np.nan,
                    'between_group_variance': np.var(stats['group_means']) if len(stats['group_means']) > 1 else np.nan,
                    'group_range': max(stats['group_means']) - min(stats['group_means']) if len(stats['group_means']) > 1 else np.nan,
                    'n_groups': len(stats['group_means']),
                    'group_means': stats['group_means']
                }
                for attr, stats in ite_by_attribute.items()
            }
        })
        
        print(f"✅ Recommended ITE aggregation completed:")
        print(f"   overall_ite_mean (signed): {overall_ite_mean_signed:.6f}")
        print(f"   overall_ite_mean_abs: {overall_ite_mean_abs:.6f}")
        print(f"   Based on {len(all_individual_ite_values)} individual ITE values")
        print(f"   Across {len(ite_by_attribute)} protected attributes")
        if all_group_ite_disparities:
            print(f"   Max group disparity: {max(all_group_ite_disparities):.6f}")
    else:
        overall_results.update({
            'overall_ite_mean': np.nan,
            'error': 'No individual ITE values collected for overall calculation'
        })
    
    print(f"  ✅ Successfully compiled overall results:")
    print(f"    Worst fairness score: {worst_fairness_score}")
    print(f"    Successful analyses: {len(successful_groups)}")
    print(f"    Worst group: {worst_group}")
    # NEW: Print NIE and PSE summary
    if all_nie_scores:
        print(f"    NIE calculations: {len(all_nie_scores)}, Mean: {np.mean(all_nie_scores):.6f}, Max: {np.max(all_nie_scores):.6f}")
    if all_nde_scores:
        print(f"    NDE calculations: {len(all_nde_scores)}, Mean: {np.mean(all_nde_scores):.6f}")
    if all_te_scores:
        print(f"    TE calculations: {len(all_te_scores)}, Mean: {np.mean(all_te_scores):.6f}, Max: {np.max(all_te_scores):.6f}")
    if all_pse_scores:
        print(f"    PSE calculations: {len(all_pse_scores)}, Mean: {np.mean(all_pse_scores):.6f}, Max: {np.max(all_pse_scores):.6f}")
    if pse_path_results:
        print(f"    PSE paths analyzed: {len(pse_path_results)}")
    # NEW: Print ITE summary using recommended approach
    if all_individual_ite_values:
        overall_ite_mean_for_print = np.mean(all_individual_ite_values)
        overall_ite_mean_abs_for_print = np.mean(np.abs(all_individual_ite_values))
        print(f"    ITE calculations: {len(all_individual_ite_values)} individual values, Mean (signed): {overall_ite_mean_for_print:.6f}, Mean abs: {overall_ite_mean_abs_for_print:.6f}")
        if all_group_ite_disparities:
            print(f"    ITE group disparities: Mean = {np.mean(all_group_ite_disparities):.6f}, Max = {max(all_group_ite_disparities):.6f}")
        if all_beneficial_rate_disparities:
            print(f"    Beneficial rate disparities: Mean = {np.mean(all_beneficial_rate_disparities):.6f}, Max = {max(all_beneficial_rate_disparities):.6f}")
    # Print attribute-specific ITE summary
    if ite_by_attribute:
        print(f"    ITE by protected attribute:")
        for attr, stats in ite_by_attribute.items():
            if stats['effect_sizes']:
                print(f"      {attr}: Effect={np.mean(stats['effect_sizes']):.4f}±{np.std(stats['effect_sizes']):.4f}, "
                     f"N_groups={len(stats['group_means'])}, Group_range={max(stats['group_means']) - min(stats['group_means']):.4f}" if len(stats['group_means']) > 1 else f"      {attr}: Effect={np.mean(stats['effect_sizes']):.4f}, N_groups={len(stats['group_means'])}")
    
    return overall_results

def flatten_fairness_results_for_csv(fairness_results, base_result):
    """
    Convert nested fairness results into multiple flat rows for CSV output.
    Each row represents one group-attribute combination with all metrics.
    
    Args:
        fairness_results: The nested fairness results from evaluate_fairness_on_data
        base_result: The base DAG result (dag_id, edges, etc.)
    
    Returns:
        list: List of flattened dictionaries for DataFrame conversion
    """
    flattened_rows = []
    
    # Check if we have group results
    if 'group_results' not in fairness_results:
        # If no group results, create a single row with overall results only
        row = base_result.copy()
        row.update({
            'group_name': 'overall',
            'group_definition': 'all_individuals',
            'attribute_name': 'overall',
            'samples_in_group': fairness_results.get('samples_analyzed', np.nan),
            'fairness_score': fairness_results.get('fairness_score', np.nan),
            'mean_observed': fairness_results.get('mean_observed', np.nan),
            'mean_counterfactual': fairness_results.get('mean_counterfactual', np.nan),
            'binary_fairness_score': fairness_results.get('binary_fairness_score', np.nan),
            'overall_fairness_score': fairness_results.get('fairness_score', np.nan),
            'analysis_error': fairness_results.get('error', '')
        })
        # Add overall-level ITE metrics
        if 'overall_ite_mean' in fairness_results:
            row.update({
                'overall_ite_mean': fairness_results.get('overall_ite_mean', np.nan),
                'overall_ite_mean_signed': fairness_results.get('overall_ite_mean_signed', fairness_results.get('overall_ite_mean', np.nan)),
                'overall_ite_mean_abs': fairness_results.get('overall_ite_mean_abs', np.nan),
                'overall_ite_median_mean': fairness_results.get('overall_ite_median_mean', fairness_results.get('overall_ite_median', np.nan)),
                'overall_ite_heterogeneity_mean': fairness_results.get('overall_ite_heterogeneity_mean', fairness_results.get('overall_ite_std', np.nan)),
                'overall_pct_positive_effects': fairness_results.get('overall_pct_positive_effects', np.nan),
                'overall_pct_negative_effects': fairness_results.get('overall_pct_negative_effects', np.nan),
                'n_ite_calculations': fairness_results.get('n_ite_calculations', 0),
            })
        flattened_rows.append(row)
        return flattened_rows
    
    group_results = fairness_results['group_results']
    overall_results = fairness_results
    
    # IMPROVED: Add overall-level ITE metrics to base result for all rows
    base_overall_metrics = {}
    if 'overall_ite_mean' in overall_results:
        base_overall_metrics.update({
            'overall_ite_mean': overall_results.get('overall_ite_mean', np.nan),
            'overall_ite_mean_signed': overall_results.get('overall_ite_mean_signed', overall_results.get('overall_ite_mean', np.nan)),
            'overall_ite_mean_abs': overall_results.get('overall_ite_mean_abs', np.nan),
            'overall_ite_median_mean': overall_results.get('overall_ite_median_mean', overall_results.get('overall_ite_median', np.nan)),
            'overall_ite_heterogeneity_mean': overall_results.get('overall_ite_heterogeneity_mean', overall_results.get('overall_ite_std', np.nan)),
            'overall_pct_positive_effects': overall_results.get('overall_pct_positive_effects', np.nan),
            'overall_pct_negative_effects': overall_results.get('overall_pct_negative_effects', np.nan),
            'n_ite_calculations': overall_results.get('n_ite_calculations', 0),
        })
    
    # Add attribute-specific overall ITE metrics for detailed analysis
    ite_by_attr = overall_results.get('ite_by_attribute', {})
    for attr, metrics in ite_by_attr.items():
        prefix = f"overall_{attr}_ite"
        base_overall_metrics.update({
            f'{prefix}_mean_effect': metrics.get('mean_effect', np.nan),
            f'{prefix}_effect_size': metrics.get('effect_size_mean', np.nan),
            f'{prefix}_effect_size_std': metrics.get('effect_size_std', np.nan),
            f'{prefix}_heterogeneity': metrics.get('heterogeneity_mean', np.nan),
            f'{prefix}_pct_positive': metrics.get('pct_positive_mean', np.nan),
            f'{prefix}_pct_negative': metrics.get('pct_negative_mean', np.nan),
            f'{prefix}_n_measurements': metrics.get('n_measurements', 0),
            f'{prefix}_95_ci_lower': metrics.get('effect_95_ci_lower', np.nan),
            f'{prefix}_95_ci_upper': metrics.get('effect_95_ci_upper', np.nan),
            f'{prefix}_significant': metrics.get('effect_significant', False),
        })
    
    # Process each group
    for group_name, group_result in group_results.items():
        if not isinstance(group_result, dict):
            continue
            
        # Get group-level information
        group_definition = group_result.get('group_definition', {})
        group_def_str = str(group_definition) if group_definition else 'all_individuals'
        samples_in_group = group_result.get('samples_in_group', np.nan)
        group_max_fairness = group_result.get('max_fairness_score', np.nan)
        group_mean_fairness = group_result.get('mean_fairness_score', np.nan)
        
        # Group-level individual fairness metrics
        group_mean_macd = group_result.get('mean_macd', np.nan)
        group_max_macd = group_result.get('max_macd', np.nan)
        group_mean_max_gap = group_result.get('mean_max_gap', np.nan)
        group_overall_max_gap = group_result.get('overall_max_gap', np.nan)
        
        # Group-level interventional fairness
        interventional_fairness = group_result.get('interventional_fairness', {})
        group_1_fairness_mean = np.nan
        group_1_fairness_max = np.nan
        group_1_fairness_violation_rate = np.nan
        group_2_fairness_mean = np.nan
        group_2_fairness_max = np.nan
        group_2_fairness_violation_rate = np.nan
        
        if interventional_fairness:
            if '1_fairness' in interventional_fairness and isinstance(interventional_fairness['1_fairness'], dict):
                fairness_1 = interventional_fairness['1_fairness']
                group_1_fairness_mean = fairness_1.get('mean_violation', np.nan)
                group_1_fairness_max = fairness_1.get('max_violation', np.nan)
                group_1_fairness_violation_rate = fairness_1.get('violation_rate', np.nan)
            
            if '2_fairness' in interventional_fairness and isinstance(interventional_fairness['2_fairness'], dict):
                fairness_2 = interventional_fairness['2_fairness']
                group_2_fairness_mean = fairness_2.get('mean_violation', np.nan)
                group_2_fairness_max = fairness_2.get('max_violation', np.nan)
                group_2_fairness_violation_rate = fairness_2.get('violation_rate', np.nan)
        
        # NEW: Group-level NIE results
        nie_results = group_result.get('nie_results', {})
        group_nie_metrics = {}
        for attr, nie_result in nie_results.items():
            if isinstance(nie_result, dict):
                group_nie_metrics[f'group_{attr}_nie_score'] = nie_result.get('nie_score', np.nan)
                group_nie_metrics[f'group_{attr}_nde_score'] = nie_result.get('nde_score', np.nan)
                group_nie_metrics[f'group_{attr}_total_effect'] = nie_result.get('total_effect', np.nan)
                group_nie_metrics[f'group_{attr}_nie_n_individuals'] = nie_result.get('n_individuals', np.nan)
        
        # NEW: Group-level PSE results
        pse_results = group_result.get('pse_results', {})
        group_pse_metrics = {}
        for attr, attr_pse_results in pse_results.items():
            if isinstance(attr_pse_results, dict):
                for path_name, pse_result in attr_pse_results.items():
                    if isinstance(pse_result, dict):
                        prefix = f'group_{attr}_{path_name}'
                        group_pse_metrics[f'{prefix}_pse_score'] = pse_result.get('pse_score', np.nan)
                        group_pse_metrics[f'{prefix}_total_effect'] = pse_result.get('total_effect', np.nan)
                        group_pse_metrics[f'{prefix}_effect_without_path'] = pse_result.get('effect_without_path', np.nan)
                        group_pse_metrics[f'{prefix}_pse_n_individuals'] = pse_result.get('n_individuals', np.nan)
        
        # NEW: Group-level ITE results
        ite_results = group_result.get('ite_results', {})
        group_ite_metrics = {}
        for attr, ite_result in ite_results.items():
            if isinstance(ite_result, dict):
                prefix = f'group_{attr}_ite'
                group_ite_metrics[f'{prefix}_mean'] = ite_result.get('ite_mean', np.nan)
                group_ite_metrics[f'{prefix}_median'] = ite_result.get('ite_median', np.nan)
                group_ite_metrics[f'{prefix}_std'] = ite_result.get('ite_std', np.nan)
                group_ite_metrics[f'{prefix}_min'] = ite_result.get('ite_min', np.nan)
                group_ite_metrics[f'{prefix}_max'] = ite_result.get('ite_max', np.nan)
                group_ite_metrics[f'{prefix}_heterogeneity'] = ite_result.get('effect_heterogeneity', np.nan)
                group_ite_metrics[f'{prefix}_pct_positive'] = ite_result.get('pct_positive_effects', np.nan)
                group_ite_metrics[f'{prefix}_pct_negative'] = ite_result.get('pct_negative_effects', np.nan)
                group_ite_metrics[f'{prefix}_n_individuals'] = ite_result.get('n_individuals', np.nan)
        
        # Process each attribute within the group
        attributes_analyzed = group_result.get('attributes_analyzed', {})
        
        if attributes_analyzed:
            for attr_name, attr_result in attributes_analyzed.items():
                if not isinstance(attr_result, dict):
                    continue
                
                # Create a row for this group-attribute combination
                row = base_result.copy()
                # Add overall-level metrics to every row
                row.update(base_overall_metrics)
                
                # Group information
                row.update({
                    'group_name': group_name,
                    'group_definition': group_def_str,
                    'samples_in_group': samples_in_group,
                    'group_max_fairness_score': group_max_fairness,
                    'group_mean_fairness_score': group_mean_fairness,
                    
                    # Group-level individual fairness metrics
                    'group_mean_macd': group_mean_macd,
                    'group_max_macd': group_max_macd,
                    'group_mean_max_gap': group_mean_max_gap,
                    'group_overall_max_gap': group_overall_max_gap,
                    
                    # Group-level interventional fairness
                    'group_1_fairness_mean_violation': group_1_fairness_mean,
                    'group_1_fairness_max_violation': group_1_fairness_max,
                    'group_1_fairness_violation_rate': group_1_fairness_violation_rate,
                    'group_2_fairness_mean_violation': group_2_fairness_mean,
                    'group_2_fairness_max_violation': group_2_fairness_max,
                    'group_2_fairness_violation_rate': group_2_fairness_violation_rate,
                })
                
                # NEW: Add group-level NIE, PSE and ITE metrics
                row.update(group_nie_metrics)
                row.update(group_pse_metrics)
                row.update(group_ite_metrics)
                
                # Attribute information
                row.update({
                    'attribute_name': attr_name,
                    'attribute_has_error': 'error' in attr_result,
                    'attribute_error_message': attr_result.get('error', ''),
                })
                
                # Attribute-level counterfactual fairness metrics
                if 'fairness_score' in attr_result and not pd.isna(attr_result['fairness_score']):
                    row.update({
                        # Main counterfactual fairness metrics
                        'fairness_score': attr_result['fairness_score'],
                        'mean_observed': attr_result.get('mean_observed', np.nan),
                        'mean_counterfactual': attr_result.get('mean_counterfactual', np.nan),
                        'binary_mean_observed': attr_result.get('binary_mean_observed', np.nan),
                        'binary_mean_counterfactual': attr_result.get('binary_mean_counterfactual', np.nan),
                        'binary_fairness_score': attr_result.get('binary_fairness_score', np.nan),
                        'samples_analyzed': attr_result.get('samples_analyzed', np.nan),
                        'primary_comparison': attr_result.get('primary_comparison', ''),
                        
                        # Individual counterfactual fairness metrics
                        'macd': attr_result.get('macd', np.nan),
                        'max_counterfactual_gap': attr_result.get('max_counterfactual_gap', np.nan),
                        'overall_macd': attr_result.get('overall_macd', np.nan),
                        'overall_max_gap': attr_result.get('overall_max_gap', np.nan),
                    })
                else:
                    # Attribute failed - fill with NaNs
                    row.update({
                        'fairness_score': np.nan,
                        'mean_observed': np.nan,
                        'mean_counterfactual': np.nan,
                        'binary_mean_observed': np.nan,
                        'binary_mean_counterfactual': np.nan,
                        'binary_fairness_score': np.nan,
                        'samples_analyzed': np.nan,
                        'primary_comparison': '',
                        'macd': np.nan,
                        'max_counterfactual_gap': np.nan,
                        'overall_macd': np.nan,
                        'overall_max_gap': np.nan,
                    })
                
                # Add all comparisons as additional columns (optional - can be quite wide)
                all_comparisons = attr_result.get('all_comparisons', {})
                for comparison_name, comparison_data in all_comparisons.items():
                    if isinstance(comparison_data, dict):
                        prefix = f"comparison_{comparison_name}"
                        row.update({
                            f'{prefix}_fairness_score': comparison_data.get('fairness_score', np.nan),
                            f'{prefix}_mean_factual': comparison_data.get('mean_factual', np.nan),
                            f'{prefix}_mean_counterfactual': comparison_data.get('mean_counterfactual', np.nan),
                            f'{prefix}_binary_fairness_score': comparison_data.get('binary_fairness_score', np.nan),
                        })
                
                flattened_rows.append(row)
        else:
            # Group has no successful attribute analyses - create one row for the group
            row = base_result.copy()
            row.update({
                'group_name': group_name,
                'group_definition': group_def_str,
                'samples_in_group': samples_in_group,
                'group_max_fairness_score': group_max_fairness,
                'group_mean_fairness_score': group_mean_fairness,
                'attribute_name': 'none',
                'attribute_has_error': True,
                'attribute_error_message': group_result.get('error', 'No successful attribute analyses'),
                'fairness_score': np.nan,
                # Fill other metrics with NaN
                'mean_observed': np.nan,
                'mean_counterfactual': np.nan,
                'binary_fairness_score': np.nan,
                'macd': np.nan,
                'max_counterfactual_gap': np.nan,
            })
            flattened_rows.append(row)
    
    # Add overall results as additional columns to all rows
    if flattened_rows:
        overall_metrics = {
            'overall_worst_fairness_score': overall_results.get('fairness_score', np.nan),
            'overall_mean_fairness_score': overall_results.get('mean_fairness_score', np.nan),
            'overall_worst_group': overall_results.get('worst_group', ''),
            'overall_successful_analyses': overall_results.get('successful_analyses', 0),
            'overall_groups_analyzed': overall_results.get('groups_analyzed', 0),
            
            # Overall individual fairness metrics
            'overall_mean_macd': overall_results.get('overall_mean_macd', np.nan),
            'overall_max_macd': overall_results.get('overall_max_macd', np.nan),
            'overall_mean_max_gap': overall_results.get('overall_mean_max_gap', np.nan),
            'overall_worst_max_gap': overall_results.get('overall_worst_max_gap', np.nan),
            
            # Overall interventional fairness metrics
            'overall_1_fairness_mean': overall_results.get('overall_1_fairness_mean', np.nan),
            'overall_1_fairness_max': overall_results.get('overall_1_fairness_max', np.nan),
            'overall_2_fairness_mean': overall_results.get('overall_2_fairness_mean', np.nan),
            'overall_2_fairness_max': overall_results.get('overall_2_fairness_max', np.nan),
            
            # NEW: Overall NIE metrics
            'overall_nie_mean': overall_results.get('overall_nie_mean', np.nan),
            'overall_nie_mean_abs': overall_results.get('overall_nie_mean_abs', np.nan),
            'overall_nie_max': overall_results.get('overall_nie_max', np.nan),
            'overall_nie_min': overall_results.get('overall_nie_min', np.nan),
            'overall_nie_std': overall_results.get('overall_nie_std', np.nan),
            'overall_nde_mean': overall_results.get('overall_nde_mean', np.nan),
            'overall_nde_max': overall_results.get('overall_nde_max', np.nan),
            'overall_nde_min': overall_results.get('overall_nde_min', np.nan),
            'overall_nde_std': overall_results.get('overall_nde_std', np.nan),
            'overall_nde_mean_abs': overall_results.get('overall_nde_mean_abs', np.nan),
            'overall_de_mean': overall_results.get('overall_de_mean', np.nan),
            'overall_de_max': overall_results.get('overall_de_max', np.nan),
            'overall_de_min': overall_results.get('overall_de_min', np.nan),
            'overall_de_std': overall_results.get('overall_de_std', np.nan),
            'overall_te_mean': overall_results.get('overall_te_mean', np.nan),
            'overall_te_max': overall_results.get('overall_te_max', np.nan),
            'overall_te_min': overall_results.get('overall_te_min', np.nan),
            'overall_te_std': overall_results.get('overall_te_std', np.nan),
            'n_nie_calculations': overall_results.get('n_nie_calculations', 0),
            
            # NEW: Overall PSE metrics
            'overall_pse_mean': overall_results.get('overall_pse_mean', np.nan),
            'overall_pse_mean_abs': overall_results.get('overall_pse_mean_abs', np.nan),
            'overall_pse_max': overall_results.get('overall_pse_max', np.nan),
            'overall_pse_min': overall_results.get('overall_pse_min', np.nan),
            'overall_pse_std': overall_results.get('overall_pse_std', np.nan),
            'n_pse_calculations': overall_results.get('n_pse_calculations', 0),
        }
        
        for row in flattened_rows:
            row.update(overall_metrics)
    
    return flattened_rows

def print_fairness_summary(group_results, overall_results):
    """Print a comprehensive summary of fairness results."""
    
    print("\nDETAILED RESULTS BY GROUP:")
    print("-" * 60)
    
    for group_name, group_result in group_results.items():
        print(f"\n📊 {group_name.upper()}:")
        print(f"   Definition: {group_result.get('group_definition', {})}")
        print(f"   Samples: {group_result.get('samples_in_group', 0)}")
        
        if 'max_fairness_score' in group_result and not pd.isna(group_result['max_fairness_score']):
            print(f"   Max Fairness Score: {group_result['max_fairness_score']:.6f}")
            print(f"   Mean Fairness Score: {group_result.get('mean_fairness_score', 'N/A'):.6f}")
            
            # Individual fairness metrics
            if 'mean_macd' in group_result and not pd.isna(group_result.get('mean_macd')):
                print(f"   Mean MACD: {group_result['mean_macd']:.6f}")
            if 'overall_max_gap' in group_result and not pd.isna(group_result.get('overall_max_gap')):
                print(f"   Overall Max Gap: {group_result['overall_max_gap']:.6f}")
            
            # Show attribute breakdown
            if 'attributes_analyzed' in group_result:
                print(f"   Attribute Breakdown:")
                for attr, attr_result in group_result['attributes_analyzed'].items():
                    if 'fairness_score' in attr_result and not pd.isna(attr_result['fairness_score']):
                        print(f"     {attr}:")
                        print(f"       Fairness score: {attr_result['fairness_score']:.6f}")
                        print(f"       Mean observed: {attr_result.get('mean_observed', 'N/A'):.6f}")
                        print(f"       Mean counterfactual: {attr_result.get('mean_counterfactual', 'N/A'):.6f}")
                        print(f"       Binary mean observed: {attr_result.get('binary_mean_observed', 'N/A'):.6f}")
                        print(f"       Binary mean counterfactual: {attr_result.get('binary_mean_counterfactual', 'N/A'):.6f}")
                        print(f"       Binary fairness score: {attr_result.get('binary_fairness_score', 'N/A'):.6f}")
                        # Individual fairness metrics
                        if 'macd' in attr_result and not pd.isna(attr_result.get('macd')):
                            print(f"       MACD: {attr_result['macd']:.6f}")
                        if 'max_counterfactual_gap' in attr_result and not pd.isna(attr_result.get('max_counterfactual_gap')):
                            print(f"       Max Counterfactual Gap: {attr_result['max_counterfactual_gap']:.6f}")
                    else:
                        print(f"     {attr}: ERROR")
            
            # Interventional fairness results
            if 'interventional_fairness' in group_result and group_result['interventional_fairness']:
                print(f"   Interventional Fairness Results:")
                for k_fairness, results in group_result['interventional_fairness'].items():
                    if isinstance(results, dict) and 'mean_violation' in results and not pd.isna(results['mean_violation']):
                        print(f"     {k_fairness} mean violation: {results['mean_violation']:.6f}")
                        print(f"     {k_fairness} max violation: {results['max_violation']:.6f}")
                        print(f"     {k_fairness} violation rate: {results['violation_rate']:.6f}")
            
            # NEW: NIE results
            if 'nie_results' in group_result and group_result['nie_results']:
                print(f"   Natural Indirect Effect (NIE) Results:")
                for attr, nie_result in group_result['nie_results'].items():
                    if isinstance(nie_result, dict) and 'nie_score' in nie_result:
                        if not pd.isna(nie_result['nie_score']):
                            print(f"     {attr}:")
                            print(f"       NIE Score: {nie_result['nie_score']:.6f}")
                            if 'nde_score' in nie_result and not pd.isna(nie_result['nde_score']):
                                print(f"       NDE Score: {nie_result['nde_score']:.6f}")
                            if 'total_effect' in nie_result and not pd.isna(nie_result['total_effect']):
                                print(f"       Total Effect: {nie_result['total_effect']:.6f}")
                            if 'n_individuals' in nie_result:
                                print(f"       Individuals: {nie_result['n_individuals']}")
                        else:
                            print(f"     {attr}: No valid NIE calculation")
                            if 'error' in nie_result:
                                print(f"       Error: {nie_result['error']}")
            
            # NEW: PSE results
            if 'pse_results' in group_result and group_result['pse_results']:
                print(f"   Path-Specific Effect (PSE) Results:")
                for attr, attr_pse_results in group_result['pse_results'].items():
                    if isinstance(attr_pse_results, dict):
                        print(f"     {attr}:")
                        for path_name, pse_result in attr_pse_results.items():
                            if isinstance(pse_result, dict) and 'pse_score' in pse_result:
                                if not pd.isna(pse_result['pse_score']):
                                    print(f"       {path_name}:")
                                    print(f"         PSE Score: {pse_result['pse_score']:.6f}")
                                    if 'total_effect' in pse_result and not pd.isna(pse_result['total_effect']):
                                        print(f"         Total Effect: {pse_result['total_effect']:.6f}")
                                    if 'effect_without_path' in pse_result and not pd.isna(pse_result['effect_without_path']):
                                        print(f"         Effect without Path: {pse_result['effect_without_path']:.6f}")
                                    if 'n_individuals' in pse_result:
                                        print(f"         Individuals: {pse_result['n_individuals']}")
                                else:
                                    print(f"       {path_name}: No valid PSE calculation")
                                    if 'error' in pse_result:
                                        print(f"         Error: {pse_result['error']}")
            # NEW: ITE results
            if 'ite_results' in group_result and group_result['ite_results']:
                print(f"   Individual Treatment Effect (ITE) Results:")
                for attr, ite_result in group_result['ite_results'].items():
                    if isinstance(ite_result, dict) and 'ite_mean' in ite_result:
                        if not pd.isna(ite_result['ite_mean']):
                            print(f"     {attr}:")
                            print(f"       ITE Mean: {ite_result['ite_mean']:.6f}")
                            if 'ite_median' in ite_result and not pd.isna(ite_result['ite_median']):
                                print(f"       ITE Median: {ite_result['ite_median']:.6f}")
                            if 'ite_std' in ite_result and not pd.isna(ite_result['ite_std']):
                                print(f"       ITE Std Dev: {ite_result['ite_std']:.6f}")
                            if 'effect_heterogeneity' in ite_result and not pd.isna(ite_result['effect_heterogeneity']):
                                print(f"       Effect Heterogeneity: {ite_result['effect_heterogeneity']:.6f}")
                            if 'pct_positive_effects' in ite_result and not pd.isna(ite_result['pct_positive_effects']):
                                print(f"       Positive Effects: {ite_result['pct_positive_effects']:.1f}%")
                            if 'pct_negative_effects' in ite_result and not pd.isna(ite_result['pct_negative_effects']):
                                print(f"       Negative Effects: {ite_result['pct_negative_effects']:.1f}%")
                            if 'n_individuals' in ite_result:
                                print(f"       Individuals: {ite_result['n_individuals']}")
                        else:
                            print(f"     {attr}: No valid ITE calculation")
                            if 'error' in ite_result:
                                print(f"       Error: {ite_result['error']}")
                        
        else:
            error_msg = group_result.get('error', 'Unknown error')
            print(f"   ❌ ERROR: {error_msg}")
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY:")
    
    if 'fairness_score' in overall_results and not pd.isna(overall_results['fairness_score']):
        print(f"Worst Fairness Score: {overall_results['fairness_score']:.6f}")
        print(f"Mean Fairness Score: {overall_results.get('mean_fairness_score', 'N/A'):.6f}")
        print(f"Worst Group: {overall_results.get('worst_group', 'N/A')}")
        print(f"Successful Analyses: {overall_results.get('successful_analyses', 0)}")
        print(f"Groups Analyzed: {overall_results.get('groups_analyzed', 0)}")
        
        # Individual fairness metrics
        if 'overall_mean_macd' in overall_results and not pd.isna(overall_results.get('overall_mean_macd')):
            print(f"Overall Mean MACD: {overall_results['overall_mean_macd']:.6f}")
        if 'overall_worst_max_gap' in overall_results and not pd.isna(overall_results.get('overall_worst_max_gap')):
            print(f"Overall Worst Max Gap: {overall_results['overall_worst_max_gap']:.6f}")
        
        # Interventional fairness metrics
        if 'overall_1_fairness_mean' in overall_results and not pd.isna(overall_results.get('overall_1_fairness_mean')):
            print(f"Overall 1-Fairness Mean: {overall_results['overall_1_fairness_mean']:.6f}")
            print(f"Overall 1-Fairness Max: {overall_results.get('overall_1_fairness_max', 'N/A'):.6f}")
        
        if 'overall_2_fairness_mean' in overall_results and not pd.isna(overall_results.get('overall_2_fairness_mean')):
            print(f"Overall 2-Fairness Mean: {overall_results['overall_2_fairness_mean']:.6f}")
            print(f"Overall 2-Fairness Max: {overall_results.get('overall_2_fairness_max', 'N/A'):.6f}")
        
        # NEW: NIE and PSE summary
        if 'overall_nie_mean' in overall_results and not pd.isna(overall_results.get('overall_nie_mean')):
            print(f"Overall NIE Mean: {overall_results['overall_nie_mean']:.6f}")
            if not pd.isna(overall_results.get('overall_nie_mean_abs', np.nan)):
                print(f"Overall NIE Mean Abs: {overall_results['overall_nie_mean_abs']:.6f}")
            print(f"Overall NIE Max: {overall_results.get('overall_nie_max', 'N/A'):.6f}")
            print(f"NIE Calculations: {overall_results.get('n_nie_calculations', 0)}")
        
        if 'overall_nde_mean' in overall_results and not pd.isna(overall_results.get('overall_nde_mean')):
            print(f"Overall NDE Mean: {overall_results['overall_nde_mean']:.6f}")
            if not pd.isna(overall_results.get('overall_de_mean', np.nan)):
                print(f"Overall DE Mean (alias): {overall_results['overall_de_mean']:.6f}")

        if 'overall_te_mean' in overall_results and not pd.isna(overall_results.get('overall_te_mean')):
            print(f"Overall TE Mean: {overall_results['overall_te_mean']:.6f}")
        
        if 'overall_pse_mean' in overall_results and not pd.isna(overall_results.get('overall_pse_mean')):
            print(f"Overall PSE Mean: {overall_results['overall_pse_mean']:.6f}")
            if not pd.isna(overall_results.get('overall_pse_mean_abs', np.nan)):
                print(f"Overall PSE Mean Abs: {overall_results['overall_pse_mean_abs']:.6f}")
            print(f"Overall PSE Max: {overall_results.get('overall_pse_max', 'N/A'):.6f}")
            print(f"PSE Calculations: {overall_results.get('n_pse_calculations', 0)}")
        
        # NEW: Overall ITE results
        if 'overall_ite_mean' in overall_results and not pd.isna(overall_results.get('overall_ite_mean')):
            print(f"Overall ITE Mean (signed): {overall_results['overall_ite_mean']:.6f}")
            ite_abs_mean = overall_results.get('overall_ite_mean_abs', np.nan)
            if not pd.isna(ite_abs_mean):
                print(f"Overall ITE Mean Abs: {ite_abs_mean:.6f}")
            # Fix key name mismatch and formatting
            ite_median = overall_results.get('overall_ite_median_mean', overall_results.get('overall_ite_median', np.nan))
            if not pd.isna(ite_median):
                print(f"Overall ITE Median: {ite_median:.6f}")
            else:
                print(f"Overall ITE Median: N/A")
            
            ite_heterogeneity = overall_results.get('overall_ite_heterogeneity_mean', overall_results.get('overall_ite_std', np.nan))
            if not pd.isna(ite_heterogeneity):
                print(f"Overall ITE Heterogeneity: {ite_heterogeneity:.6f}")
            else:
                print(f"Overall ITE Heterogeneity: N/A")
            
            print(f"ITE Calculations: {overall_results.get('n_ite_calculations', 0)}")
            
            # Fix percentage formatting
            pct_positive = overall_results.get('overall_pct_positive_effects', np.nan)
            pct_negative = overall_results.get('overall_pct_negative_effects', np.nan)
            if not pd.isna(pct_positive):
                print(f"Overall Positive Effects: {pct_positive:.1f}%")
            else:
                print(f"Overall Positive Effects: N/A")
            if not pd.isna(pct_negative):
                print(f"Overall Negative Effects: {pct_negative:.1f}%")
            else:
                print(f"Overall Negative Effects: N/A")
            
            # IMPROVED: Detailed ITE results by protected attribute for ML paper reporting
            ite_by_attr = overall_results.get('ite_by_attribute', {})
            if ite_by_attr:
                print(f"\n=== Individual Treatment Effects by Protected Attribute ===")
                print(f"{'Attribute':<15} {'Mean ITE':<10} {'Effect Size':<12} {'95% CI':<20} {'Hetero.':<8} {'Pos%':<6} {'N':<4}")
                print(f"{'-'*80}")
                
                for attr, metrics in ite_by_attr.items():
                    mean_ite = metrics.get('mean_effect', np.nan)
                    effect_size = metrics.get('effect_size_mean', np.nan)
                    effect_std = metrics.get('effect_size_std', np.nan)
                    ci_lower = metrics.get('effect_95_ci_lower', np.nan)
                    ci_upper = metrics.get('effect_95_ci_upper', np.nan)
                    heterogeneity = metrics.get('heterogeneity_mean', np.nan)
                    pct_pos = metrics.get('pct_positive_mean', np.nan)
                    n_measurements = metrics.get('n_measurements', 0)
                    significant = metrics.get('effect_significant', False)
                    
                    # Format values with appropriate precision
                    mean_str = f"{mean_ite:.4f}" if not pd.isna(mean_ite) else "N/A"
                    effect_str = f"{effect_size:.4f}" if not pd.isna(effect_size) else "N/A"
                    if not pd.isna(effect_std) and effect_std > 0:
                        effect_str += f"±{effect_std:.3f}"
                    
                    if not pd.isna(ci_lower) and not pd.isna(ci_upper):
                        ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                        if significant:
                            ci_str += "*"  # Mark significant effects
                    else:
                        ci_str = "N/A"
                    
                    hetero_str = f"{heterogeneity:.3f}" if not pd.isna(heterogeneity) else "N/A"
                    pos_str = f"{pct_pos:.1f}" if not pd.isna(pct_pos) else "N/A"
                    
                    print(f"{attr:<15} {mean_str:<10} {effect_str:<12} {ci_str:<20} {hetero_str:<8} {pos_str:<6} {n_measurements:<4}")
                
                print(f"* indicates statistically significant effect at α=0.05 (approximate)")
                print(f"ITE = Individual Treatment Effect; Hetero. = Heterogeneity; Pos% = % Positive Effects")
        
        # Path-specific PSE results
        pse_by_path = overall_results.get('pse_by_path', {})
        if pse_by_path:
            print(f"PSE by Path:")
            for path_name, path_metrics in pse_by_path.items():
                mean_val = path_metrics.get('mean', np.nan)
                max_val = path_metrics.get('max', np.nan)
                mean_str = f"{mean_val:.6f}" if not pd.isna(mean_val) else "N/A"
                max_str = f"{max_val:.6f}" if not pd.isna(max_val) else "N/A"
                print(f"  {path_name}: Mean={mean_str}, Max={max_str}")
                
    else:
        print("❌ No successful fairness analyses completed")
        if 'error' in overall_results:
            print(f"Error: {overall_results['error']}")
    print(f"{'='*60}")


def evaluate_synthetic_data_ml(synthetic_data, 
                               data_type, 
                               dag_id, 
                               protected_attrs, 
                               disadvantage_group, 
                               target, 
                               X_cols, 
                               dag,
                               ml_results_dir,
                               cutoff_threshold=0.5):
    """
    Process synthetic data for ML evaluation and save results.
    
    Args:
        synthetic_data: DataFrame with synthetic data
        data_type: Type of synthetic data ('standard', 'double_size', etc.)
        dag_id: ID of the DAG
        protected_attrs: List of protected attributes
        disadvantage_group: Dictionary defining disadvantaged group
        target: Target variable name
        X_cols: List of feature column names
        dag: DAG edges
        ml_results_dir: Directory to save ML results
        cutoff_threshold: Threshold for XGBoost model predictions (default: 0.5)
        
    Returns:
        list: ML evaluation results
    """
    try:
        print(f"Running ML evaluation on {data_type} synthetic data...")
        
        # Check if we have enough data and classes
        if len(synthetic_data) < 10:
            print(f"Warning: Not enough synthetic data samples ({len(synthetic_data)})")
            return []
        
        target_dist = synthetic_data[target].value_counts()
        if len(target_dist) < 2:
            print(f"Warning: Only one class in synthetic {data_type} data")
            return []
        
        print(f"Target distribution: {target_dist.to_dict()}")

        # ===== SMART DATA TYPE CHECK - ONLY CONVERT IF NEEDED =====
        synthetic_data_processed = synthetic_data.copy()
        conversion_needed = False
        
        # Check if conversion is actually needed
        print("Checking data types for ML compatibility...")
        non_numeric_cols = []
        
        for col in synthetic_data_processed.columns:
            if not pd.api.types.is_numeric_dtype(synthetic_data_processed[col]):
                # Check if it contains string representations of numbers
                sample_values = synthetic_data_processed[col].dropna().head(10)
                if len(sample_values) > 0:
                    # Test if values look like they should be numeric
                    string_numeric_pattern = all(
                        str(val).replace('.', '').replace('-', '').isdigit() 
                        for val in sample_values
                    )
                    if string_numeric_pattern:
                        non_numeric_cols.append(col)
                        conversion_needed = True
        
        if conversion_needed:
            print(f"⚠️  Conversion needed for columns: {non_numeric_cols}")
            print("Converting string-numeric data to proper numeric format...")
            
            for col in non_numeric_cols:
                original_dtype = synthetic_data_processed[col].dtype
                try:
                    # Convert to numeric
                    synthetic_data_processed[col] = pd.to_numeric(
                        synthetic_data_processed[col], errors='coerce'
                    )
                    
                    # Handle any NaN values created during conversion
                    if synthetic_data_processed[col].isna().any():
                        # Use the mode of original values, converted to numeric
                        original_mode = synthetic_data[col].mode()
                        if len(original_mode) > 0:
                            mode_val = pd.to_numeric(original_mode.iloc[0], errors='coerce')
                            if pd.isna(mode_val):
                                mode_val = 0
                        else:
                            mode_val = 0
                        
                        synthetic_data_processed[col] = synthetic_data_processed[col].fillna(mode_val)
                    
                    # For categorical columns, ensure integer type
                    if col in X_cols:  # Don't force target conversion
                        synthetic_data_processed[col] = synthetic_data_processed[col].astype(int)
                    
                    print(f"  ✅ Converted {col}: {original_dtype} -> {synthetic_data_processed[col].dtype}")
                    
                except Exception as e:
                    print(f"  ⚠️  Could not convert {col}: {e}")
                    # Fallback: leave as-is and let sklearn handle it
                    synthetic_data_processed[col] = synthetic_data[col]
        else:
            print("✅ All data types are already ML-compatible, no conversion needed")
        
        # Final validation - check if any obviously problematic data remains
        final_check_failed = False
        for col in X_cols:
            if synthetic_data_processed[col].dtype == 'object':
                # Check if object column contains mixed types
                sample_vals = synthetic_data_processed[col].dropna().head(5).tolist()
                if any(isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit() for val in sample_vals):
                    print(f"⚠️  Column {col} still contains non-numeric strings: {sample_vals}")
                    final_check_failed = True
        
        if final_check_failed:
            print("⚠️  Some columns may still cause issues, but proceeding with evaluation...")
        
        # ===== END OF SMART CONVERSION SECTION =====

        # Prepare features and target
        X = synthetic_data_processed[X_cols].copy()
        y = synthetic_data_processed[target].copy()
        
        # Handle any missing values
        import numpy as np  # Add this import at the top of the file if not present
        if X.select_dtypes(include=[np.number]).shape[1] > 0:
            X = X.fillna(X.median())
        else:
            X = X.fillna(X.mode().iloc[0] if len(X.mode()) > 0 else 0)
        
        # Enhanced ML evaluation with multiple approaches
        results = []
        
        try:
            print("Running evaluation using existing run_samples function...")
            
            # Split the data for run_samples
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            available_attrs = [a for a in protected_attrs if a in synthetic_data_processed.columns]

            if available_attrs:
                results_df = run_samples(
                    sample_name=f"dag_{dag_id}_{data_type}",
                    df_model=synthetic_data,
                    target_name=target,
                    X_tr=X_train,
                    y_tr=y_train,
                    X_te=X_test,
                    y_te=y_test,
                    protected_attrs=available_attrs,
                    df_original=synthetic_data,
                    random_state=42,
                    cutoff_threshold=cutoff_threshold
                )

                # Convert DataFrame to list of dictionaries
                run_samples_results = results_df.to_dict('records')
                results.extend(run_samples_results)
                print(f"Successfully got {len(run_samples_results)} results from run_samples")

            else:
                print(f"Warning: No protected attributes {protected_attrs} found in data")
                
        except Exception as e:
            print(f"Warning: run_samples evaluation failed: {str(e)}")
            print("Note: This is expected if run_samples has compatibility issues")
            # You could add a simple fallback here if needed
        
        # Add metadata to results
        for result in results:
            result['dag_id'] = dag_id
            result['data_type'] = data_type
            result['dag_edges'] = str(dag)
            result['synthetic_samples'] = len(synthetic_data_processed)
        
        # Save individual results
        if ml_results_dir and results:  # Only save if we have results
            import os  # Add this import at the top if not present
            os.makedirs(ml_results_dir, exist_ok=True)  # Ensure directory exists
            result_file = os.path.join(ml_results_dir, f"{data_type}_dag_{dag_id}_results.csv")
            result_df = pd.DataFrame(results)
            result_df.to_csv(result_file, index=False)
            print(f"Saved {len(results)} results to {result_file}")
        
        return results
        
    except Exception as e:
        print(f"Error in ML evaluation for {data_type} data: {str(e)}")
        import traceback
        traceback.print_exc()  # This helps with debugging
        return []


def log_ml_results_for_dag(results, dag_id, data_type, protected_attrs, target):
    """Log ML results for a specific DAG"""
    if results:
        try:
            print(f"ML Results for DAG {dag_id} ({data_type}):\n")
            for result in results:
                if 'model_type' in result:
                    print(f"  {result['model_type']}: ")
                    print(f"AUROC={result.get('AUROC', 'N/A')}, ")
                    print(f"DPD={result.get('DPD', 'N/A')}, ")
                    print(f"EO={result.get('EO', 'N/A')}\n")
        except Exception as e:
            print(f"Error logging ML results: {str(e)}")


def cpdag_to_dags(cpdag, node_names=None):
    """
    FIXED: Convert a CPDAG (represented as a GeneralGraph from causal_learn) to all possible DAGs.
    
    Parameters:
    -----------
    cpdag : GeneralGraph
        The CPDAG from the GES algorithm
    node_names : list, optional
        A list of node names corresponding to the nodes in the CPDAG
        
    Returns:
    --------
    list
        A list of DAGs, each represented as a list of (source, target) tuples
    """
    import itertools
    
    # Handle different types of input
    if hasattr(cpdag, 'get_nodes'):
        # This is a GeneralGraph from causal-learn
        nodes = cpdag.get_nodes()
        n_nodes = len(nodes)
        
        # Create node name mapping
        if node_names is None:
            node_names = [node.get_name() for node in nodes]
        
        print(f"🔍 Node mapping:")
        for i, name in enumerate(node_names):
            print(f"   {i}: {name}")
        
        # Extract edges from the GeneralGraph
        directed_edges = []
        undirected_edges = []
        
        # Access the graph matrix directly
        graph_matrix = cpdag.graph
        
        print(f"🔍 Analyzing adjacency matrix ({graph_matrix.shape}):")
        
        # Import endpoint values for clarity
        from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint
        TAIL = Endpoint.TAIL.value  # = 1
        ARROW = Endpoint.ARROW.value  # = -1
        NULL = Endpoint.NULL.value  # = 0
        
        print(f"   Endpoint values: TAIL={TAIL}, ARROW={ARROW}, NULL={NULL}")
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes): # <-- FIX IS HERE
                val_ij = graph_matrix[i, j]
                val_ji = graph_matrix[j, i]

                # Check for an edge from i to j: i --> j
                # This means a TAIL at i and an ARROW at j
                if val_ij == TAIL and val_ji == ARROW:
                    directed_edges.append((node_names[i], node_names[j]))
                    print(f"   Directed: {node_names[i]} -> {node_names[j]}")
                
                # Check for an edge from j to i: j --> i
                # This means a TAIL at j and an ARROW at i
                elif val_ij == ARROW and val_ji == TAIL:
                    directed_edges.append((node_names[j], node_names[i]))
                    print(f"   Directed: {node_names[j]} -> {node_names[i]}")
                    
                # Check for an undirected edge: i -- j
                # This means a TAIL at both i and j
                elif val_ij == TAIL and val_ji == TAIL:
                    undirected_edges.append((node_names[i], node_names[j]))
                    print(f"   Undirected: {node_names[i]} -- {node_names[j]}")
        
        print(f"🔍 Extracted edges:")
        print(f"   Directed: {len(directed_edges)}")
        for edge in directed_edges:
            print(f"     {edge[0]} -> {edge[1]}")
        print(f"   Undirected: {len(undirected_edges)}")
        for edge in undirected_edges:
            print(f"     {edge[0]} -- {edge[1]}")
            
    elif hasattr(cpdag, 'pgmpy_dag'):
        # This is a pgmpy wrapper
        from src.faircausal.core.ges_runner_pgmpy import simple_dag_converter
        return simple_dag_converter(cpdag, node_names)
        
    elif hasattr(cpdag, '__len__'):
        # This is a matrix representation
        n_nodes = len(cpdag)
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_nodes)]
        
        directed_edges = []
        undirected_edges = []
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if cpdag[i][j] == 1:  # Directed edge i -> j
                    directed_edges.append((node_names[i], node_names[j]))
                # Add logic for undirected edges if needed
                    
    else:
        raise ValueError(f"Unsupported CPDAG type: {type(cpdag)}")
    
    # Generate all possible orientations of undirected edges
    all_dags = []
    
    if not undirected_edges:
        # No undirected edges - return the single DAG
        print(f"🔍 No undirected edges - returning single DAG with {len(directed_edges)} edges")
        all_dags.append(directed_edges)
    else:
        # Generate all possible orientations
        print(f"🔍 Found {len(undirected_edges)} undirected edges to orient")
        
        # Limit the number of orientations to avoid combinatorial explosion
        max_orientations = 1000
        n_orientations = 2 ** len(undirected_edges)
        
        if n_orientations > max_orientations:
            print(f"⚠️  Too many possible orientations ({n_orientations}). Limiting to {max_orientations}")
            # Sample random orientations instead of trying all
            import random
            random.seed(42)
            
            for _ in range(max_orientations):
                dag_edges = directed_edges.copy()
                for edge in undirected_edges:
                    if random.choice([True, False]):
                        dag_edges.append(edge)
                    else:
                        dag_edges.append((edge[1], edge[0]))  # Reverse direction
                
                # Simple cycle check using NetworkX
                G = nx.DiGraph()
                G.add_edges_from(dag_edges)
                if nx.is_directed_acyclic_graph(G):
                    all_dags.append(dag_edges)
        else:
            # Try all possible orientations
            for directions in itertools.product([0, 1], repeat=len(undirected_edges)):
                dag_edges = directed_edges.copy()
                
                for (u, v), direction in zip(undirected_edges, directions):
                    if direction == 0:
                        dag_edges.append((u, v))
                    else:
                        dag_edges.append((v, u))
                
                # Check for cycles
                G = nx.DiGraph()
                G.add_edges_from(dag_edges)
                if nx.is_directed_acyclic_graph(G):
                    all_dags.append(dag_edges)
    
    print(f"🔍 Generated {len(all_dags)} valid DAGs from CPDAG")
    
    return all_dags

def select_fairest_dag(
    cpdag,
    data: np.ndarray,
    node_names: List[str],
    protected_attr_indices: List[int],
    outcome_index: int,
    fairness_lambda: float = 1.0,
    beta: float = 0.0,
    max_stage2_dags: int = 50,
) -> Dict[str, Any]:
    """
    Stage 2 of the Two-Stage Pipeline: select the fairest DAG from the MEC.

    Optimizes the fairness-adjusted BIC score (Eq. 9 in the paper):
        FS(G) = BIC(G) - lambda * N * Psi_Total(G)

    where Psi_Total(G) = sum_i rho_G(i) * sum_{j in Pa(i)} phi_star(j),
    and rho_G uses graph-aware Markov Blanket membership (Eq. 7-8).

    Parameters
    ----------
    cpdag : GeneralGraph
        The CPDAG output from Stage 1 (GES).
    data : np.ndarray, shape (N, n_nodes)
        The discretized data used for BIC scoring.
    node_names : List[str]
        Names of all nodes in the graph.
    protected_attr_indices : List[int]
        Indices of protected attribute nodes.
    outcome_index : int
        Index of the outcome node.
    fairness_lambda : float
        Regularization weight lambda (same value used in Stage 1).
    beta : float
        Penalty discount for nodes outside MB(Y), in [0, 1]. Default 0.0
        (only edges within MB(Y) are penalised).
    max_stage2_dags : int
        Maximum number of DAGs to evaluate (to avoid combinatorial explosion).

    Returns
    -------
    dict with keys:
        'dag_edges'       : list of (src, tgt) string tuples for the best DAG
        'score'           : fairness-adjusted BIC score FS(G)
        'bic_score'       : raw BIC(G)
        'fairness_penalty': total Psi_Total across the DAG
        'n_dags_evaluated': number of DAGs compared
    """
    from src.faircausal.causal_learn.causallearn.search.ScoreBased.GES_fair import OptimizedBICScore
    from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph
    from src.faircausal.causal_learn.causallearn.graph.GraphNode import GraphNode
    from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint
    from src.faircausal.core.fairness_scoring import (
        compute_proxy_strength_matrix,
        compute_proxy_load,
        compute_phase2_penalty,
    )

    N = data.shape[0]
    n_nodes = len(node_names)
    TAIL_val = Endpoint.TAIL.value
    ARROW_val = Endpoint.ARROW.value

    # Enumerate all DAGs in the MEC
    all_dags = cpdag_to_dags(cpdag, node_names)

    if not all_dags:
        print("Warning: Stage 2 found no valid DAGs in MEC.")
        return {
            "dag_edges": [],
            "score": float("-inf"),
            "bic_score": float("-inf"),
            "fairness_penalty": 0.0,
            "n_dags_evaluated": 0,
        }

    if len(all_dags) <= 1:
        print("Stage 2: Only one DAG in MEC — no selection needed.")
        return {
            "dag_edges": all_dags[0] if all_dags else [],
            "score": float("nan"),
            "bic_score": float("nan"),
            "fairness_penalty": float("nan"),
            "n_dags_evaluated": len(all_dags),
        }

    # Limit evaluation to max_stage2_dags
    if len(all_dags) > max_stage2_dags:
        import random
        random.seed(42)
        all_dags = random.sample(all_dags, max_stage2_dags)
        print(f"Stage 2: Randomly sampled {max_stage2_dags} DAGs for evaluation.")

    # Precompute reusable objects
    bic_scorer = OptimizedBICScore(data)
    proxy_strengths = compute_proxy_strength_matrix(data, node_names, protected_attr_indices)
    proxy_load = compute_proxy_load(proxy_strengths, n_nodes, protected_attr_indices)
    N_float = float(N)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    def _build_general_graph(dag_edges):
        """Convert a list of (src_name, tgt_name) edge tuples to a GeneralGraph."""
        nodes = [GraphNode(name) for name in node_names]
        G = GeneralGraph(nodes)
        for src, tgt in dag_edges:
            si, ti = name_to_idx[src], name_to_idx[tgt]
            G.graph[si, ti] = TAIL_val
            G.graph[ti, si] = ARROW_val
        return G

    def _bic_score(dag_edges):
        """Compute total BIC score for a DAG represented as edge tuples."""
        parents: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
        for src, tgt in dag_edges:
            parents[name_to_idx[tgt]].append(name_to_idx[src])
        return sum(-bic_scorer.score(i, parents[i]) for i in range(n_nodes))

    print(f"Stage 2: Evaluating {len(all_dags)} DAGs for fairness-adjusted BIC ...")

    best_info: Dict[str, Any] = {
        "dag_edges": all_dags[0],
        "score": float("-inf"),
        "bic_score": 0.0,
        "fairness_penalty": 0.0,
        "n_dags_evaluated": len(all_dags),
    }

    for dag_edges in all_dags:
        bic = _bic_score(dag_edges)
        G = _build_general_graph(dag_edges)

        total_penalty = compute_phase2_penalty(
            G.graph,
            data,
            proxy_load,
            outcome_index,
            beta=beta,
        )

        fs = bic - fairness_lambda * N_float * total_penalty

        if fs > best_info["score"]:
            best_info = {
                "dag_edges": dag_edges,
                "score": fs,
                "bic_score": bic,
                "fairness_penalty": total_penalty,
                "n_dags_evaluated": len(all_dags),
            }

    print(
        f"Stage 2: Best DAG — BIC={best_info['bic_score']:.4f}, "
        f"penalty={best_info['fairness_penalty']:.4f}, FS={best_info['score']:.4f}"
    )
    return best_info


def analyze_all_dags_improved_flow(cpdag, 
                                   df, 
                                   estimator, 
                                   protected_attrs, 
                                   X_cols, 
                                   target, 
                                   disadvantage_group, 
                                   var_types, 
                                   node_names=None, 
                                   max_dags=None,
                                   save_model=False, 
                                   model_dir="models",
                                   log_file=None,
                                   generate_synthetic=True,
                                   generate_synthetic_double=False,
                                   generate_synthetic_balanced=False,
                                   generate_synthetic_protected=False,
                                   sample_percentage=1.0,
                                   min_samples=10,
                                   random_state=42,
                                   cutoff_threshold=0.5,
                                   calculate_pse=False,
                                   pse_sample_size=None,
                                   max_paths=None):
    """
    FIXED: Analyze all possible DAGs from a CPDAG with proper flow control
    """
    print(f"Generating all possible DAGs from CPDAG...")
    all_dags = cpdag_to_dags(cpdag, node_names)
    print(f"Generated {len(all_dags)} possible DAGs")
    
    if max_dags and len(all_dags) > max_dags:
        print(f"Limiting analysis to {max_dags} DAGs (out of {len(all_dags)})")
        all_dags = all_dags[:max_dags]

    # Create directories
    if save_model:
        os.makedirs(model_dir, exist_ok=True)

    synthetic_dir = os.path.join(os.path.dirname(model_dir), "synthetic_data")
    os.makedirs(synthetic_dir, exist_ok=True)
    
    ml_results_dir = os.path.join(os.path.dirname(model_dir), "ml_results")
    os.makedirs(ml_results_dir, exist_ok=True)
    
    # Results storage
    results = []
    ml_results = []
    
    # The TeeLogger in the main runner handles everything.
    print("\n\n=== SCM ANALYSIS: CF FAIRNESS FIRST, THEN ML ===")
    print(f"Analyzing {len(all_dags)} DAGs\n")

    # Analyze each DAG with proper control flow
    for i, dag in enumerate(tqdm(all_dags, desc="Analyzing DAGs")):
        print(f"\n{'='*60}")
        print(f"ANALYZING DAG {i+1}/{len(all_dags)}")
        print(f"DAG edges: {dag}")
        print(f"{'='*60}")
        
        result = {
            'dag_id': i,
            'edges': dag,
            'n_edges': len(dag)
        }
        
        try:
            # Step 1: Fit SCM ONCE per DAG
            print("Step 1: Fitting SCM...")
            scm = fit_scm_for_categorical_data(df, dag, var_types, random_state)
            
            if scm is None:
                print(f"❌ Failed to fit SCM for DAG {i}")
                result.update({
                    'scm_fitted': False,
                    'synthetic_generated': False,
                    'fairness_score': np.nan,
                    'error': 'SCM fitting failed',
                    'data_type': 'error'
                })
                results.append(result)
                continue
            
            result['scm_fitted'] = True
            print("✅ SCM fitted successfully")
            
            # Step 2: Generate synthetic data variants ONCE per DAG
            print("Step 2: Generating synthetic data variants...")
            synthetic_datasets = []
            
            # Determine which variants to generate
            variants_to_generate = []
            if generate_synthetic:
                variants_to_generate.append(('standard', int(len(df) * sample_percentage), False, False))
            if generate_synthetic_double:
                variants_to_generate.append(('double_size', int(len(df) * sample_percentage * 2), False, False))
            if generate_synthetic_balanced:
                variants_to_generate.append(('balanced', int(len(df) * sample_percentage), True, True))
            if generate_synthetic_protected:
                variants_to_generate.append(('protected_balanced', int(len(df) * sample_percentage), False, True))
            
            # If no synthetic variants enabled, use original data
            if not variants_to_generate:
                print("No synthetic data generation enabled - using original data")
                synthetic_datasets.append(('original', df, '0_original'))
            else:
                # Generate each variant ONCE
                for variant_name, n_samples, balance_outcome, balance_protected in variants_to_generate:
                    print(f"  Generating {variant_name} synthetic data ({n_samples} samples)...")
                    
                    synthetic_data = generate_synthetic_data(
                        scm, 
                        n_samples=n_samples,
                        var_types=var_types,
                        balance_outcome=balance_outcome,
                        balance_protected=balance_protected,
                        target=target if balance_outcome else None,
                        protected_attrs=protected_attrs if balance_protected else None
                    )
                    
                    if synthetic_data is not None:
                        file_prefix = {
                            'standard': '1_standard',
                            'double_size': '2_double_size', 
                            'balanced': '3_balanced',
                            'protected_balanced': '4_protected'
                        }[variant_name]
                        
                        synthetic_datasets.append((variant_name, synthetic_data, file_prefix))
                        print(f"  ✅ Generated {variant_name} data with {len(synthetic_data)} samples")
                    else:
                        print(f"  ❌ Failed to generate {variant_name} data")
                
                # Fallback to original if all synthetic generation failed
                if not synthetic_datasets:
                    print("❌ All synthetic data generation failed - using original data")
                    synthetic_datasets.append(('original', df, '0_original'))
                    result['synthetic_generated'] = False
                else:
                    result['synthetic_generated'] = True
            
            # Step 3 & 4: Process each dataset variant
            for data_type, dataset, file_prefix in synthetic_datasets:
                print(f"\n--- Processing {data_type} data ---")
                print(f"Dataset shape: {dataset.shape}")
                
                # # Save synthetic data (but not original)
                # if dataset is not df:
                #     file_path = os.path.join(synthetic_dir, f"{file_prefix}_dag_{i}.csv")
                #     dataset.to_csv(file_path, index=False)
                #     print(f"💾 Saved {data_type} data to {file_path}")
                
                # Step 3: Evaluate COUNTERFACTUAL FAIRNESS
                print(f"Step 3: Evaluating fairness on {data_type} data...")
                fairness_results = evaluate_fairness_on_data(
                    df=dataset,
                    protected_attrs=protected_attrs,
                    target=target,
                    disadvantage_group=disadvantage_group,
                    estimator=estimator,
                    X_cols=X_cols,
                    causal_model=scm,
                    dag=dag,
                    var_types=var_types,
                    random_state=random_state,
                    sample_percentage=sample_percentage,
                    min_samples=min_samples,
                    calculate_pse=calculate_pse,
                    pse_sample_size=pse_sample_size,
                    max_paths=max_paths
                )
                
                # Create detailed fairness result records (one per group-attribute combination)
                flattened_fairness_results = flatten_fairness_results_for_csv(fairness_results, result)
                
                for fairness_result in flattened_fairness_results:
                    fairness_result['data_type'] = data_type
                    fairness_result['synthetic_samples'] = len(dataset)
                    
                    # Store results
                    results.append(fairness_result)
                
                # Step 4: Evaluate ML performance
                print(f"Step 4: Evaluating ML performance on {data_type} data...")

                ml_eval_results = evaluate_synthetic_data_ml(
                    synthetic_data=dataset,
                    data_type=data_type,
                    dag_id=i,
                    protected_attrs=protected_attrs,
                    disadvantage_group=disadvantage_group,
                    target=target,
                    X_cols=X_cols,
                    dag=dag,
                    ml_results_dir=ml_results_dir,
                    cutoff_threshold=cutoff_threshold
                )
                ml_results.extend(ml_eval_results)
                        
                print(f"\nDAG {i+1} ({data_type}):\n")
                print(f"Edges: {dag}\n")
                print(f"ML results: {len(ml_eval_results)} models\n")
                
                print(f"✅ Completed processing {data_type} data for DAG {i+1}")
            
        except Exception as e:
            print(f"❌ Error analyzing DAG {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            result.update({
                'scm_fitted': False,
                'synthetic_generated': False,
                'fairness_score': np.nan,
                'error': str(e),
                'data_type': 'error'
            })
            results.append(result)

    
    # Convert results to DataFrames
    results_df = pd.DataFrame(results)
    
    # Handle ML results
    ml_results_df = pd.DataFrame()
    if ml_results:
        valid_dfs = []
        for result in ml_results:
            try:
                if isinstance(result, pd.DataFrame) and not result.empty:
                    valid_dfs.append(result)
                elif isinstance(result, dict):
                    valid_dfs.append(pd.DataFrame([result]))
                elif isinstance(result, list):
                    for sub_result in result:
                        if isinstance(sub_result, dict):
                            valid_dfs.append(pd.DataFrame([sub_result]))
            except Exception as e:
                print(f"Error processing ML result: {str(e)}")
        
        if valid_dfs:
            ml_results_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"Processed {len(all_dags)} DAGs")
    print(f"Total result rows: {len(results)}")
    print(f"ML evaluation rows: {len(ml_results_df)}")
    print(f"{'='*60}")
    
    return results_df, ml_results_df

# Legacy function for backward compatibility
def analyze_all_dags_fairness(*args, **kwargs):
    """
    Legacy function - redirects to improved flow.
    Maintains backward compatibility while using the new architecture.
    """
    print("Using improved flow architecture...")
    return analyze_all_dags_improved_flow(*args, **kwargs)
