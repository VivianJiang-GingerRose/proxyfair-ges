# src/faircausal/data/data_loader_bank.py

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
        self.log_file.write(f"BANK MARKETING DATA PREPROCESSING LOG\n")
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
    Convert all columns in bank marketing dataset to numerical categorical format for causal analysis.
    Ensures ALL text columns are properly converted to integers.
    
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
    log_func("BANK MARKETING DATASET COLUMN CONVERSION LOG")
    log_func("=" * 60)
    
    # Define comprehensive mapping rules for ALL columns
    column_mappings = {
        # Numerical columns that need categorical binning
        'age': {
            'type': 'age_binary_binning',
            'description': 'Age binning: 0=middle(35-55), 1=young(<35) or senior(>55)'
        },
        'balance': {
            'type': 'balance_categorical_binning',
            'description': 'Balance binning: 0=negative, 1=zero, 2=low positive(1-1000), 3=high positive(>1000)'
        },
        'day': {
            'type': 'day_categorical_binning',
            'description': 'Day binning: 0=early month(1-10), 1=mid month(11-20), 2=late month(21-31)'
        },
        'duration': {
            'type': 'duration_categorical_binning',
            'description': 'Duration binning: 0=very short(≤60s), 1=short(61-180s), 2=medium(181-600s), 3=long(>600s)'
        },
        'campaign': {
            'type': 'campaign_categorical_binning',
            'description': 'Campaign binning: 0=single contact(1), 1=few contacts(2-3), 2=many contacts(4+)'
        },
        'pdays': {
            'type': 'pdays_categorical_binning',
            'description': 'Pdays binning: 0=never contacted(-1), 1=recent(0-90), 2=moderate(91-180), 3=long ago(181+)'
        },
        'previous': {
            'type': 'previous_categorical_binning',
            'description': 'Previous binning: 0=none(0), 1=few(1-2), 2=several(3-5), 3=many(6+)'
        },
        
        # Text columns that need categorical encoding
        'job': {
            'type': 'categorical_label_encoding',
            'description': 'Job categories encoded as 0-based integers'
        },
        'marital': {
            'type': 'marital_binary_binning',
            'description': 'Marital status binning: 0=married, 1=single/divorced/other'
        },
        'education': {
            'type': 'categorical_label_encoding',
            'description': 'Education level encoded as 0-based integers'
        },
        'contact': {
            'type': 'categorical_label_encoding',
            'description': 'Contact type encoded as 0-based integers'
        },
        'month': {
            'type': 'categorical_label_encoding',
            'description': 'Month encoded as 0-based integers'
        },
        'poutcome': {
            'type': 'categorical_label_encoding',
            'description': 'Previous outcome encoded as 0-based integers'
        },
        
        # Binary yes/no columns
        'default': {
            'type': 'binary_yesno',
            'description': 'Convert yes/no to 1/0'
        },
        'housing': {
            'type': 'binary_yesno',
            'description': 'Convert yes/no to 1/0'
        },
        'loan': {
            'type': 'binary_yesno',
            'description': 'Convert yes/no to 1/0'
        },
        'y': {
            'type': 'binary_yesno',
            'description': 'Convert yes/no to 1/0 (target variable)'
        }
    }
    
    # Process each column according to its rules
    for col in df_converted.columns:
        if col not in df_converted.columns:
            continue
            
        log_func(f"\nProcessing column: {col}")
        log_func(f"Original data type: {df_converted[col].dtype}")
        
        # Show original unique values (limited for readability)
        original_unique = df_converted[col].unique()
        if len(original_unique) <= 20:
            log_func(f"Original unique values: {sorted(original_unique, key=str)}")
        else:
            log_func(f"Original unique values (first 20): {sorted(original_unique, key=str)[:20]}...")
        
        log_func(f"Original distribution: {df_converted[col].value_counts().to_dict()}")
        
        if col in column_mappings:
            mapping_rule = column_mappings[col]
            
            if mapping_rule['type'] == 'binary_yesno':
                # Handle yes/no binary mapping
                # Create mapping for common variations
                df_converted[col] = df_converted[col].astype(str).str.lower().str.strip()
                
                # Define mapping
                binary_mapping = {
                    'yes': 1, 'y': 1, '1': 1, 'true': 1,
                    'no': 0, 'n': 0, '0': 0, 'false': 0
                }
                
                # Apply mapping
                df_converted[col] = df_converted[col].map(binary_mapping)
                
                # Handle any unmapped values
                unmapped_mask = df_converted[col].isna()
                if unmapped_mask.any():
                    unmapped_values = df_converted.loc[unmapped_mask, col].unique()
                    log_func(f"Warning: Unmapped values in {col}: {unmapped_values}")
                    # Assign to majority class
                    majority_value = 0  # Default to 'no'
                    df_converted.loc[unmapped_mask, col] = majority_value
                    log_func(f"Assigned unmapped values to: {majority_value}")
                
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'binary_yesno',
                    'mapping': binary_mapping,
                    'distribution': df_converted[col].value_counts().to_dict()
                }

            elif mapping_rule['type'] == 'marital_binary_binning':
                # Marital binary binning: 0=married, 1=single/divorced/other
                
                # Convert to string and clean
                df_converted[col] = df_converted[col].astype(str).str.lower().str.strip()
                
                # Handle missing values
                df_converted[col] = df_converted[col].replace(['nan', 'NaN', 'None', ''], 'unknown')
                
                # Show what values we actually have before mapping
                unique_marital_values = sorted(df_converted[col].unique())
                log_func(f"Original marital values found: {unique_marital_values}")
                
                # Define what counts as "married" vs "single/other"
                # Note: Adjust these lists based on your actual data values
                married_indicators = [
                    'married', 'mar', 'm', 'marriage', 'wed', 'wedded'
                ]
                
                # Everything else is considered "single/other"
                # This includes: single, divorced, widowed, separated, unknown, etc.
                
                # Create binary mapping
                marital_binary_map = {}
                
                for val in unique_marital_values:
                    if any(married_word in val for married_word in married_indicators):
                        marital_binary_map[val] = 0  # Married
                        log_func(f"  '{val}' → 0 (married)")
                    else:
                        marital_binary_map[val] = 1  # Single/Divorced/Other
                        log_func(f"  '{val}' → 1 (single/other)")
                
                log_func(f"Final marital binary mapping: {marital_binary_map}")
                
                # Apply the mapping
                df_converted[col] = df_converted[col].map(marital_binary_map)
                
                # Verify no unmapped values
                if df_converted[col].isna().any():
                    unmapped_count = df_converted[col].isna().sum()
                    log_func(f"ERROR: {unmapped_count} unmapped marital values found!")
                    # Assign unmapped to single/other category as fallback
                    df_converted[col] = df_converted[col].fillna(1)
                    log_func(f"Assigned unmapped values to single/other category (1)")
                
                # Ensure integer type
                df_converted[col] = df_converted[col].astype(int)
                
                # Verify we only have 0s and 1s
                final_unique = sorted(df_converted[col].unique())
                if set(final_unique) != {0, 1}:
                    log_func(f"ERROR: Marital binary conversion failed! Got values: {final_unique}")
                    log_func(f"Expected: [0, 1]")
                else:
                    log_func(f"✓ Marital successfully converted to binary: {final_unique}")
                
                conversion_log[col] = {
                    'type': 'marital_binary_binning',
                    'mapping': marital_binary_map,
                    'description': '0=married, 1=single/divorced/other',
                    'distribution': df_converted[col].value_counts().to_dict(),
                    'final_categories': sorted(df_converted[col].unique())
                }
                
                log_func(f"Final marital distribution: {df_converted[col].value_counts().to_dict()}")
                
            elif mapping_rule['type'] == 'categorical_label_encoding':
                # Handle categorical text columns with label encoding
                # Convert to string and clean
                df_converted[col] = df_converted[col].astype(str).str.strip()
                
                # Handle missing values
                df_converted[col] = df_converted[col].replace(['nan', 'NaN', 'None', ''], 'unknown')
                
                # Get unique values and create mapping
                unique_values = sorted(df_converted[col].unique())
                value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                
                # Apply mapping
                df_converted[col] = df_converted[col].map(value_mapping)
                
                # Ensure all values are mapped
                if df_converted[col].isna().any():
                    log_func(f"Warning: Some values in {col} could not be mapped")
                    df_converted[col] = df_converted[col].fillna(0)
                
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'categorical_label_encoding',
                    'mapping': value_mapping,
                    'distribution': df_converted[col].value_counts().to_dict(),
                    'min_value': df_converted[col].min(),
                    'max_value': df_converted[col].max()
                }
                
            elif mapping_rule['type'] == 'age_binary_binning':
                # Age binary binning: 0=middle(35-55), 1=young(<35) or senior(>55)
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to middle age group")
                    df_converted[col] = df_converted[col].fillna(40)
                
                conditions = [
                    (df_converted[col] >= 35) & (df_converted[col] <= 55),  # Middle age
                    (df_converted[col] < 35) | (df_converted[col] > 55)     # Young or Senior
                ]
                choices = [0, 1]
                
                df_converted[col] = np.select(conditions, choices, default=0)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'age_categorical_binning',
                    'mapping': 'middle(35-55)->0, young(<35) or senior(>55)->1',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'balance_categorical_binning':
                # Balance categorical binning
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to zero balance category")
                    df_converted[col] = df_converted[col].fillna(0)
                
                conditions = [
                    df_converted[col] < 0,                          # Negative balance
                    df_converted[col] == 0,                         # Zero balance
                    (df_converted[col] > 0) & (df_converted[col] <= 1000),   # Low positive
                    df_converted[col] > 1000                        # High positive
                ]
                choices = [0, 1, 2, 3]
                
                df_converted[col] = np.select(conditions, choices, default=1)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'balance_categorical_binning',
                    'mapping': 'negative->0, zero->1, low positive(1-1000)->2, high positive(>1000)->3',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'day_categorical_binning':
                # Day categorical binning
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to mid-month category")
                    df_converted[col] = df_converted[col].fillna(15)
                
                conditions = [
                    df_converted[col] <= 10,                        # Early month
                    (df_converted[col] > 10) & (df_converted[col] <= 20),  # Mid month
                    df_converted[col] > 20                          # Late month
                ]
                choices = [0, 1, 2]
                
                df_converted[col] = np.select(conditions, choices, default=1)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'day_categorical_binning',
                    'mapping': 'early month(1-10)->0, mid month(11-20)->1, late month(21-31)->2',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'duration_categorical_binning':
                # Duration categorical binning
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to short duration category")
                    df_converted[col] = df_converted[col].fillna(120)
                
                conditions = [
                    df_converted[col] <= 60,                        # Very short calls
                    (df_converted[col] > 60) & (df_converted[col] <= 180),   # Short calls
                    (df_converted[col] > 180) & (df_converted[col] <= 600),  # Medium calls
                    df_converted[col] > 600                         # Long calls
                ]
                choices = [0, 1, 2, 3]
                
                df_converted[col] = np.select(conditions, choices, default=1)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'duration_categorical_binning',
                    'mapping': 'very short(≤60s)->0, short(61-180s)->1, medium(181-600s)->2, long(>600s)->3',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'campaign_categorical_binning':
                # Campaign categorical binning
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to single contact category")
                    df_converted[col] = df_converted[col].fillna(1)
                
                conditions = [
                    df_converted[col] == 1,                         # Single contact
                    (df_converted[col] >= 2) & (df_converted[col] <= 3),  # Few contacts
                    df_converted[col] >= 4                          # Many contacts
                ]
                choices = [0, 1, 2]
                
                df_converted[col] = np.select(conditions, choices, default=0)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'campaign_categorical_binning',
                    'mapping': 'single contact(1)->0, few contacts(2-3)->1, many contacts(4+)->2',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'pdays_categorical_binning':
                # Pdays categorical binning with special handling for -1
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to never contacted category")
                    df_converted[col] = df_converted[col].fillna(-1)
                
                conditions = [
                    df_converted[col] == -1,                        # Never contacted
                    (df_converted[col] >= 0) & (df_converted[col] <= 90),    # Recent
                    (df_converted[col] > 90) & (df_converted[col] <= 180),   # Moderate
                    df_converted[col] > 180                         # Long ago
                ]
                choices = [0, 1, 2, 3]
                
                df_converted[col] = np.select(conditions, choices, default=0)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'pdays_categorical_binning',
                    'mapping': 'never contacted(-1)->0, recent(0-90)->1, moderate(91-180)->2, long ago(181+)->3',
                    'distribution': df_converted[col].value_counts().to_dict()
                }
                
            elif mapping_rule['type'] == 'previous_categorical_binning':
                # Previous categorical binning
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                
                # Handle NaN values
                if df_converted[col].isna().any():
                    nan_count = df_converted[col].isna().sum()
                    log_func(f"Warning: {nan_count} NaN values in {col}, assigning to no previous contacts category")
                    df_converted[col] = df_converted[col].fillna(0)
                
                conditions = [
                    df_converted[col] == 0,                         # No previous contacts
                    (df_converted[col] >= 1) & (df_converted[col] <= 2),   # Few contacts
                    (df_converted[col] >= 3) & (df_converted[col] <= 5),   # Several contacts
                    df_converted[col] >= 6                          # Many contacts
                ]
                choices = [0, 1, 2, 3]
                
                df_converted[col] = np.select(conditions, choices, default=0)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'previous_categorical_binning',
                    'mapping': 'none(0)->0, few(1-2)->1, several(3-5)->2, many(6+)->3',
                    'distribution': df_converted[col].value_counts().to_dict()
                }

        
        else:
            # For any unspecified columns, apply automatic categorical encoding
            log_func(f"Column {col} not in predefined mappings, applying automatic categorical encoding")
            
            if df_converted[col].dtype == 'object' or df_converted[col].dtype.name == 'string':
                # Text column - apply label encoding
                df_converted[col] = df_converted[col].astype(str).str.strip()
                df_converted[col] = df_converted[col].replace(['nan', 'NaN', 'None', ''], 'unknown')
                
                unique_values = sorted(df_converted[col].unique())
                value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                df_converted[col] = df_converted[col].map(value_mapping)
                df_converted[col] = df_converted[col].astype(int)
                
                conversion_log[col] = {
                    'type': 'automatic_categorical_encoding',
                    'mapping': value_mapping,
                    'distribution': df_converted[col].value_counts().to_dict()
                }
            else:
                log_func(f"Numerical column {col} found - converting to categorical")
                
                # STEP 1: Convert to numeric and handle NaN
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                df_converted[col] = df_converted[col].fillna(df_converted[col].median())
                
                # STEP 2: Calculate unique_count BEFORE applying any binning
                unique_count = df_converted[col].nunique()
                log_func(f"Original unique values in {col}: {unique_count}")
                
                # STEP 3: Apply categorical binning based on unique_count
                if unique_count <= 10:
                    # If already few unique values, use direct label encoding
                    log_func(f"Applying direct label encoding (≤10 unique values)")
                    unique_values = sorted(df_converted[col].unique())
                    value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                    df_converted[col] = df_converted[col].map(value_mapping)
                    
                    conversion_log[col] = {
                        'type': 'automatic_direct_categorical',
                        'original_unique_count': unique_count,
                        'mapping': value_mapping,
                        'distribution': df_converted[col].value_counts().to_dict()
                    }
                else:
                    # Apply quartile-based binning for many unique values
                    log_func(f"Applying quartile binning (>10 unique values)")
                    try:
                        df_converted[col] = pd.qcut(
                            df_converted[col], 
                            q=4, 
                            labels=[0, 1, 2, 3], 
                            duplicates='drop'
                        ).astype(int)
                        
                        conversion_log[col] = {
                            'type': 'automatic_quartile_binning',
                            'original_unique_count': unique_count,
                            'mapping': 'quartile-based binning: 0=Q1, 1=Q2, 2=Q3, 3=Q4',
                            'distribution': df_converted[col].value_counts().to_dict()
                        }
                    except ValueError as e:
                        # Fallback if quartile binning fails (e.g., all values are the same)
                        log_func(f"Quartile binning failed for {col}: {e}")
                        log_func(f"Applying simple binning instead")
                        
                        # Simple high/low binning as fallback
                        median_val = df_converted[col].median()
                        df_converted[col] = (df_converted[col] > median_val).astype(int)
                        
                        conversion_log[col] = {
                            'type': 'automatic_median_binning',
                            'original_unique_count': unique_count,
                            'mapping': f'median split: 0=≤{median_val}, 1=>{median_val}',
                            'distribution': df_converted[col].value_counts().to_dict()
                        }
                
                log_func(f"Applied automatic categorical binning to numerical column {col}")
                log_func(f"Final categories: {sorted(df_converted[col].unique())}")
                
        # Final validation and logging
        final_values = sorted(df_converted[col].unique())
        log_func(f"Final values: {final_values}")
        log_func(f"Final data type: {df_converted[col].dtype}")
        log_func(f"Final distribution: {df_converted[col].value_counts().to_dict()}")
        log_func(f"Value range: {df_converted[col].min()} to {df_converted[col].max()}")
        
        description = mapping_rule.get('description', 'Automatic conversion') if col in column_mappings else 'Automatic conversion'
        log_func(f"Conversion: {description}")
    
    # FINAL VALIDATION: Ensure all columns are numerical
    log_func("\n" + "=" * 60)
    log_func("FINAL VALIDATION - ENSURING ALL COLUMNS ARE NUMERICAL")
    log_func("=" * 60)
    
    all_numerical = True
    for col in df_converted.columns:
        if df_converted[col].dtype not in ['int64', 'int32', 'int16', 'int8']:
            log_func(f"WARNING: Column {col} has non-integer type: {df_converted[col].dtype}")
            all_numerical = False
            # Force convert to integer
            try:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                df_converted[col] = df_converted[col].fillna(0).astype(int)
                log_func(f"FIXED: Converted {col} to integer type")
            except Exception as e:
                log_func(f"ERROR: Could not convert {col} to integer: {e}")
        else:
            log_func(f"✓ Column {col}: {df_converted[col].dtype} (range: {df_converted[col].min()}-{df_converted[col].max()})")
    
    if all_numerical:
        log_func("✓ ALL COLUMNS ARE NUMERICAL - READY FOR GES ALGORITHM")
    else:
        log_func("⚠ SOME COLUMNS REQUIRED FIXING")
    
    log_func("\n" + "=" * 60)
    log_func("CONVERSION SUMMARY")
    log_func("=" * 60)
    
    for col, log in conversion_log.items():
        log_func(f"\n{col}:")
        log_func(f"  Type: {log['type']}")
        if 'mapping' in log:
            mapping_str = str(log['mapping'])
            if len(mapping_str) > 100:
                log_func(f"  Mapping: {mapping_str[:100]}...")
            else:
                log_func(f"  Mapping: {mapping_str}")
        if 'distribution' in log:
            log_func(f"  Final distribution: {log['distribution']}")
        if 'min_value' in log and 'max_value' in log:
            log_func(f"  Value range: {log['min_value']} to {log['max_value']}")
    
    return df_converted, conversion_log


def load_and_preprocess_data(file_path: str, export_path: str = None, log_file_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline for bank marketing dataset.
    
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
    
    log_func("BANK MARKETING DATA LOADER")
    log_func("=" * 40)
    
    # Step 1: Load data
    try:
        # Try different separators common in bank marketing dataset
        df_original = pd.read_csv(file_path, sep=',')
        log_func(f"Loaded data with semicolon separator")
    except:
        try:
            df_original = pd.read_csv(file_path, sep=',')
            log_func(f"Loaded data with comma separator")
        except Exception as e:
            log_func(f"Error loading data: {e}")
            raise
    
    log_func(f"Loaded original data with shape: {df_original.shape}")
    log_func(f"Original columns: {list(df_original.columns)}")
    
    # Step 2: Basic data info
    log_func(f"\nBasic data information:")
    for col in df_original.columns:
        unique_count = df_original[col].nunique()
        data_type = df_original[col].dtype
        sample_values = df_original[col].unique()[:5]
        log_func(f"  {col}: {data_type}, {unique_count} unique values, samples: {sample_values}")
    
    # Step 3: Apply column conversions
    log_func(f"\nApplying comprehensive column conversions...")
    df_processed, conversion_log = _perform_column_conversion(
        df_original.copy(), 
        {}, 
        log_func
    )
    
    # Step 4: Validate the results
    log_func(f"\nValidation Results:")
    log_func(f"Processed data shape: {df_processed.shape}")
    log_func(f"Final columns: {list(df_processed.columns)}")
    
    # Check all columns are numerical
    non_numerical_cols = []
    for col in df_processed.columns:
        if df_processed[col].dtype not in ['int64', 'int32', 'int16', 'int8']:
            non_numerical_cols.append(col)
    
    if non_numerical_cols:
        log_func(f"WARNING: Non-numerical columns found: {non_numerical_cols}")
    else:
        log_func("✓ All columns are numerical integers")
    
    # Summary statistics
    log_func(f"\nFinal dataset summary:")
    for col in df_processed.columns:
        unique_vals = sorted(df_processed[col].unique())
        log_func(f"{col}: range {min(unique_vals)}-{max(unique_vals)}, {len(unique_vals)} categories")
    
    log_func(f"\nPreprocessing pipeline complete:")
    log_func(f"  - Original data: {df_original.shape}")
    log_func(f"  - Processed data: {df_processed.shape}")
    log_func(f"  - All variables converted to numerical categorical format")
    log_func(f"  - Ready for GES algorithm")
    
    # Export dataset if path is provided
    if export_path:
        df_processed.to_csv(export_path, index=False)
        log_func(f"Exported preprocessed dataset to {export_path}")
        
    return {
        'original': df_original,
        'preprocessed': df_processed
    }
