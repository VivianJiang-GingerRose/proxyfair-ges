import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any
import sys
import os
from contextlib import redirect_stdout
from io import StringIO


class DataPreprocessingLogger:
    """Context manager to capture and redirect preprocessing logs to file"""
    
    def __init__(self, log_file_path: str, append_mode: bool = True):
        self.log_file_path = log_file_path
        self.append_mode = append_mode
        self.log_file = None
        
    def __enter__(self):
        # Open log file
        mode = 'a' if self.append_mode else 'w'
        self.log_file = open(self.log_file_path, mode, encoding='utf-8')
        
        # Write header to log file
        self.log_file.write(f"\n{'='*60}\n")
        self.log_file.write(f"DATA PREPROCESSING LOG\n")
        self.log_file.write(f"{'='*60}\n")
        self.log_file.flush()
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close files
        if self.log_file:
            self.log_file.close()
            
    def log_and_print(self, message: str):
        """Log message to both file and console"""
        # Print to console
        print(message)
        
        # Write to log file
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()


def convert_columns_to_required_format(df: pd.DataFrame, 
                                     log_file_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Generic function to convert all columns to required formats for causal analysis.
    Enhanced version with optional file logging capability.
    
    Args:
        df: Input DataFrame
        log_file_path: Optional path to log file for preprocessing logs
        
    Returns:
        Tuple of (converted_df, conversion_log)
    """
    df_converted = df.copy()
    conversion_log = {}
    
    # Setup logging function
    if log_file_path:
        logger = DataPreprocessingLogger(log_file_path, append_mode=True)
        with logger:
            def log_message(msg: str):
                logger.log_and_print(msg)
            
            return _perform_column_conversion(df_converted, conversion_log, log_message)
    else:
        # Use standard print if no log file specified
        def log_message(msg: str):
            print(msg)
        
        return _perform_column_conversion(df_converted, conversion_log, log_message)


def _perform_column_conversion(df_converted: pd.DataFrame, 
                              conversion_log: Dict[str, Dict], 
                              log_func) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Internal function that performs the actual column conversion with custom logging"""
    
    log_func("=" * 60)
    log_func("COLUMN CONVERSION LOG")
    log_func("=" * 60)

    log_func(f"occupation unique values: {sorted(df_converted['occupation'].unique())}")
    
    # Define specific mapping rules for each column
    column_mappings = {
        'occupation': {
            'type': 'binary',
            'rule': 'employed_unemployed',
            'description': 'Map to binary: employed (>0) = 1, unemployed (<=0) = 0'
        },
        'sex': {
            'type': 'binary_encoded',
            'mapping': {'male': 0, 'female': 1},
            'target_column': 'sex_enc',
            'description': 'Create sex_enc: male=0, female=1, drop original'
        },
        'country_birth': {
            'type': 'categorical_grouped',
            'mapping': 'group_1_vs_others',
            'description': 'Group: 1->1, [2,3]->2'
        },
        'citizenship': {
            'type': 'categorical_grouped', 
            'mapping': 'group_1_vs_others', 
            'description': 'Group: 1->1, [2,3]->2'
        }
    }
    
    # Generic categorical columns that need standard 0-based encoding
    categorical_columns = [
        'age', 'household_position', 'household_size', 'prev_residence_place',
        'edu_level', 'economic_status', 'cur_eco_activity', 'marital_status'
    ]
    
    # Process each column according to its rules
    for col in df_converted.columns:
        log_func(f"\nProcessing column: {col}")
        log_func(f"Original values: {sorted(df_converted[col].unique())}")
        log_func(f"Original distribution: {df_converted[col].value_counts().to_dict()}")
        
        if col in column_mappings:
            mapping_rule = column_mappings[col]
            
            if mapping_rule['type'] == 'binary':
                # Handle occupation binary mapping
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                original_dist = df_converted[col].value_counts()
                
                if mapping_rule['rule'] == 'employed_unemployed':
                    df_converted[col] = (df_converted[col] > 0).astype(int)
                
                new_dist = df_converted[col].value_counts()
                conversion_log[col] = {
                    'type': 'binary_employment',
                    'original_distribution': original_dist.to_dict(),
                    'new_distribution': new_dist.to_dict(),
                    'mapping': '>0 -> 1 (employed), <=0 -> 0 (unemployed)'
                }
                
            elif mapping_rule['type'] == 'binary_encoded':
                # Handle sex encoding
                target_col = mapping_rule['target_column']
                mapping = mapping_rule['mapping']
                
                df_converted[target_col] = df_converted[col].map(mapping)
                
                # Handle unmapped values
                unmapped_mask = df_converted[target_col].isna()
                if unmapped_mask.any():
                    unmapped_values = df_converted.loc[unmapped_mask, col].unique()
                    log_func(f"Warning: Unmapped values in {col}: {unmapped_values}")

                    # Instead of setting to -1, assign to majority class or default
                    handle_method = mapping_rule.get('handle_unmapped', 'majority_class')

                    if handle_method == 'majority_class':
                        # Find the most common mapped value
                        mapped_values = df_converted[target_col].dropna()
                        if len(mapped_values) > 0:
                            majority_value = mapped_values.mode()[0]
                            df_converted.loc[unmapped_mask, target_col] = majority_value
                            log_func(f"Assigned unmapped values to majority class: {majority_value}")
                        else:
                            # Fallback to first valid mapping value
                            default_value = list(mapping.values())[0]
                            df_converted.loc[unmapped_mask, target_col] = default_value
                            log_func(f"Assigned unmapped values to default: {default_value}")
                    else:
                        # Alternative: assign to a specific default value (0 or 1)
                        default_value = 0  # Could also be 1 based on domain knowledge
                        df_converted.loc[unmapped_mask, target_col] = default_value
                        log_func(f"Assigned unmapped values to default: {default_value}")
                
                df_converted[target_col] = df_converted[target_col].astype(int)
                
                conversion_log[target_col] = {
                    'type': 'binary_encoded',
                    'original_column': col,
                    'mapping': mapping,
                    'distribution': df_converted[target_col].value_counts().to_dict(),
                    'unmapped_handling': 'NO -1 VALUES INTRODUCED'
                }

                log_func(f"Dropping original column: sex")
                df_converted = df_converted.drop(columns=["sex"])
                
            elif mapping_rule['type'] == 'categorical_grouped':
                # Handle grouped categorical mappings
                if mapping_rule['mapping'] == 'group_1_vs_others':
                    # Map: 1->1, [2,3]->2
                    def group_mapping(x):
                        if x == 1:
                            return 1
                        elif x in [2, 3]:
                            return 2
                    
                    df_converted[col] = df_converted[col].apply(group_mapping)
                
                conversion_log[col] = {
                    'type': 'categorical_grouped',
                    'mapping': mapping_rule['mapping'],
                    'distribution': df_converted[col].value_counts().to_dict()
                }
        
        elif col in categorical_columns:
            # Standard categorical encoding (0-based)
            unique_values = sorted(df_converted[col].unique())
            # Remove any NaN or invalid values and handle them properly
            valid_values = [v for v in unique_values if pd.notna(v)]
            if len(valid_values) != len(unique_values):
                log_func(f"Warning: Found invalid values in {col}, will be handled")

            value_mapping = {val: idx for idx, val in enumerate(unique_values)}
            
            # Apply mapping
            df_converted[col] = df_converted[col].map(value_mapping)
            # Handle any unmapped values (NaN) by assigning to category 0
            unmapped_mask = df_converted[col].isna()
            if unmapped_mask.any():
                df_converted.loc[unmapped_mask, col] = 0
                log_func(f"Assigned {unmapped_mask.sum()} unmapped values in {col} to category 0")


            df_converted[col] = df_converted[col].astype(int)
            
            conversion_log[col] = {
                'type': 'categorical_standard',
                'mapping': value_mapping,
                'distribution': df_converted[col].value_counts().to_dict(),
                'min_value': df_converted[col].min(), 
                'max_value': df_converted[col].max()
            }
        
        else:
            # Keep other columns as-is but log them
            conversion_log[col] = {
                'type': 'unchanged',
                'note': 'Column kept in original format'
            }

        # ENHANCED: Validate no -1 values were introduced
        if col in df_converted.columns:
            final_values = sorted(df_converted[col].unique())
            log_func(f"Final values: {final_values}")
            
            # CRITICAL CHECK: Ensure no -1 values exist
            if -1 in final_values:
                log_func(f"ERROR: -1 value detected in {col}! This will cause SCM issues.")
                # Fix it immediately
                df_converted[col] = df_converted[col].replace(-1, 0)
                log_func(f"FIXED: Replaced -1 with 0 in {col}")
            
            log_func(f"Final distribution: {df_converted[col].value_counts().to_dict()}")
            log_func(f"Value range: {df_converted[col].min()} to {df_converted[col].max()}")
        
        elif col in conversion_log and 'original_column' in conversion_log[col]:
            target_col = [k for k, v in conversion_log.items() if v.get('original_column') == col][0]
            log_func(f"Created new column '{target_col}': {sorted(df_converted[target_col].unique())}")
        
        description = (mapping_rule.get('description', 'Standard categorical encoding') 
                      if col in column_mappings 
                      else 'Standard 0-based categorical encoding' if col in categorical_columns 
                      else 'No change')
        log_func(f"Conversion: {description}")
    
    # FINAL VALIDATION: Check entire dataframe for -1 values
    log_func("\n" + "=" * 60)
    log_func("FINAL VALIDATION - CHECKING FOR -1 VALUES")
    log_func("=" * 60)
    
    for col in df_converted.columns:
        if df_converted[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            has_negative_one = (df_converted[col] == -1).any()
            if has_negative_one:
                count_negative_one = (df_converted[col] == -1).sum()
                log_func(f"WARNING: Column {col} has {count_negative_one} values of -1")
                # Fix them
                df_converted[col] = df_converted[col].replace(-1, 0)
                log_func(f"FIXED: Replaced -1 values with 0 in {col}")
            else:
                log_func(f"✓ Column {col}: No -1 values (range: {df_converted[col].min()}-{df_converted[col].max()})")
    
    log_func("\n" + "=" * 60)
    log_func("CONVERSION SUMMARY")
    log_func("=" * 60)
    
    for col, log in conversion_log.items():
        log_func(f"\n{col}:")
        log_func(f"  Type: {log['type']}")
        if 'mapping' in log:
            log_func(f"  Mapping: {log['mapping']}")
        if 'distribution' in log:
            log_func(f"  Final distribution: {log['distribution']}")
        if 'min_value' in log and 'max_value' in log:
            log_func(f"  Value range: {log['min_value']} to {log['max_value']}")
    
    return df_converted, conversion_log


def load_and_preprocess_data(file_path: str, export_path: str = None, log_file_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline for Dutch census dataset.
    Enhanced version with optional file logging capability.
    
    Args:
        file_path: Path to the raw data file
        export_path: Path to export processed data (optional)
        log_file_path: Optional path to write preprocessing logs to file
    
    Returns:
        Dictionary containing original and processed dataframes
    """
    
    # Setup logging function
    if log_file_path:
        logger = DataPreprocessingLogger(log_file_path, append_mode=True)
        with logger:
            def log_message(msg: str):
                logger.log_and_print(msg)
            
            return _perform_data_loading_and_preprocessing(file_path, export_path, log_message)
    else:
        def log_message(msg: str):
            print(msg)
        
        return _perform_data_loading_and_preprocessing(file_path, export_path, log_message)


def _perform_data_loading_and_preprocessing(file_path: str, 
                                          export_path: str, 
                                          log_func) -> Dict[str, pd.DataFrame]:
    """Internal function that performs the actual data loading and preprocessing"""
    
    log_func("DUTCH CENSUS DATA LOADER")
    log_func("=" * 40)
    
    # Step 1: Load data
    df_original = pd.read_csv(file_path)
    log_func(f"Loaded original data with shape: {df_original.shape}")
    log_func(f"Original columns: {list(df_original.columns)}")
    
    # Step 2: Apply generic column conversions
    log_func(f"\nApplying generic column conversions...")
    df_processed, conversion_log = _perform_column_conversion(df_original.copy(), {}, log_func)
    
    # Step 3: Validate the results
    log_func(f"\nValidation Results:")
    log_func(f"Processed data shape: {df_processed.shape}")
    log_func(f"Final columns: {list(df_processed.columns)}")
    
    # Check for binary variables
    binary_cols = ['occupation', 'sex_enc']
    for col in binary_cols:
        if col in df_processed.columns:
            unique_vals = sorted(df_processed[col].unique())
            log_func(f"{col} values: {unique_vals} (Binary: {set(unique_vals).issubset({0, 1})})")
    
    # Check for categorical variables without -1 (except where expected)
    categorical_cols = ['age', 'household_position', 'household_size', 'prev_residence_place',
                       'edu_level', 'economic_status', 'cur_eco_activity', 'marital_status',
                       'country_birth', 'citizenship']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            unique_vals = sorted(df_processed[col].unique())
            has_negative = any(v < 0 for v in unique_vals)
            log_func(f"{col} range: {min(unique_vals)}-{max(unique_vals)} (Has negative: {has_negative})")
    

    log_func(f"\nPreprocessing pipeline complete:")
    log_func(f"  - Original data: {df_original.shape}")
    log_func(f"  - Processed data: {df_processed.shape}")

    # Export dataset if path is provided
    if export_path:
        df_processed.to_csv(export_path, index=False)
        log_func(f"Exported preprocessed dataset to {export_path}")
        
    return {
        'original': df_original,
        'preprocessed': df_processed
    }
