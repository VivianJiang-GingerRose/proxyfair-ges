# src/faircausal/data/data_loader_compas.py

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
        self.log_file.write(f"COMPAS DATA PREPROCESSING LOG\n")
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


def apply_compas_preprocessing(df: pd.DataFrame, log_func) -> pd.DataFrame:
    """
    Apply COMPAS-specific preprocessing steps with proper column handling
    """
    log_func("Applying COMPAS-specific preprocessing steps...")
    
    # Start with the original dataframe
    df_compas = df.copy()
    
    # Log initial state
    log_func(f"Initial shape: {df_compas.shape}")
    log_func(f"Initial columns: {list(df_compas.columns)}")
    
    # Step 1: Drop unwanted columns first
    cols_to_drop = ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob','c_case_number', 
                    'c_offense_date', 'c_arrest_date', 'c_charge_desc',
                    'r_case_number','r_offense_date','r_charge_desc','violent_recid','vr_case_number', 
                    'vr_offense_date', 'vr_charge_desc',
                    'type_of_assessment','score_text', 'screening_date','v_type_of_assessment',
                    'v_score_text','v_screening_date','in_custody', 'out_custody', 
                    'c_jail_in', 'c_jail_out','r_jail_in', 'r_jail_out']
    
    # Only drop columns that exist
    existing_cols_to_drop = [col for col in cols_to_drop if col in df_compas.columns]
    if existing_cols_to_drop:
        df_compas = df_compas.drop(columns=existing_cols_to_drop)
        log_func(f"Dropped columns: {existing_cols_to_drop}")
    
    log_func(f"Shape after dropping unwanted columns: {df_compas.shape}")
    log_func(f"Remaining columns: {list(df_compas.columns)}")
    
    # Step 2: Handle duplicate column names (common issue in COMPAS data)
    # The sample shows duplicate 'decile_score' and 'priors_count' columns
    if 'decile_score' in df_compas.columns:
        # Keep only the first decile_score column
        decile_cols = [col for col in df_compas.columns if 'decile_score' in col]
        if len(decile_cols) > 1:
            log_func(f"Found multiple decile_score columns: {decile_cols}")
            # Keep the first one, drop the rest
            cols_to_drop_dup = decile_cols[1:]
            df_compas = df_compas.drop(columns=cols_to_drop_dup)
            log_func(f"Dropped duplicate decile_score columns: {cols_to_drop_dup}")
    
    if 'priors_count' in df_compas.columns:
        # Keep only the first priors_count column
        priors_cols = [col for col in df_compas.columns if 'priors_count' in col]
        if len(priors_cols) > 1:
            log_func(f"Found multiple priors_count columns: {priors_cols}")
            # Keep the first one, drop the rest
            cols_to_drop_dup = priors_cols[1:]
            df_compas = df_compas.drop(columns=cols_to_drop_dup)
            log_func(f"Dropped duplicate priors_count columns: {cols_to_drop_dup}")
    
    # Step 3: Create categorical variables from existing columns
    df_compas_processed = df_compas.copy()
    
    # Race categorization: Convert to binary white/non-white
    if 'race' in df_compas_processed.columns:
        log_func("Processing race column...")
        log_func(f"Original race values: {df_compas_processed['race'].value_counts().to_dict()}")
        df_compas_processed['race_cat'] = np.where(
            df_compas_processed['race'].str.contains('Caucasian', case=False, na=False), 
            1,  # 1 = white
            0   # 0 = non_white
        )
        log_func(f"Race categorization: {df_compas_processed['race_cat'].value_counts().to_dict()}")
    
    # Age categorization: Create binary age variable (1 = younger than 45, 0 = 45 or older)
    if 'age' in df_compas_processed.columns:
        log_func("Creating binary age variable...")
        log_func(f"Original age range: {df_compas_processed['age'].min()} to {df_compas_processed['age'].max()}")
        df_compas_processed['age_binary'] = np.where(df_compas_processed['age'] < 45, 1, 0)
        log_func(f"Age binary: {df_compas_processed['age_binary'].value_counts().to_dict()}")
        log_func("Note: 1 = younger than 45, 0 = 45 or older")
    
    # Drop original age_cat if it exists
    if 'age_cat' in df_compas_processed.columns:
        df_compas_processed = df_compas_processed.drop(columns=['age_cat'])
        log_func("Dropped original age_cat column")
    
    # Juvenile felony count categorization - return integers directly
    if 'juv_fel_count' in df_compas_processed.columns:
        log_func("Processing juv_fel_count column...")
        log_func(f"Original juv_fel_count range: {df_compas_processed['juv_fel_count'].min()} to {df_compas_processed['juv_fel_count'].max()}")
        df_compas_processed['juv_fel_count_cat'] = np.where(
            df_compas_processed['juv_fel_count'] > 0, 
            1,  # 1 = has juvenile felonies
            0   # 0 = no juvenile felonies
        )
        log_func(f"Juvenile felony categorization: {df_compas_processed['juv_fel_count_cat'].value_counts().to_dict()}")
    
    # Juvenile misdemeanor count categorization - return integers directly
    if 'juv_misd_count' in df_compas_processed.columns:
        log_func("Processing juv_misd_count column...")
        log_func(f"Original juv_misd_count range: {df_compas_processed['juv_misd_count'].min()} to {df_compas_processed['juv_misd_count'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            df_compas_processed['juv_misd_count'] == 0,
            df_compas_processed['juv_misd_count'] == 1,
            df_compas_processed['juv_misd_count'] >= 2
        ]
        choices = [0, 1, 2]  # 0='count_0', 1='count_1', 2='count_2+'
        
        df_compas_processed['juv_misd_count_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Juvenile misdemeanor categorization: {df_compas_processed['juv_misd_count_cat'].value_counts().to_dict()}")
        log_func("Note: 0=count_0, 1=count_1, 2=count_2+")
    
    # Juvenile other count categorization - return integers directly
    if 'juv_other_count' in df_compas_processed.columns:
        log_func("Processing juv_other_count column...")
        log_func(f"Original juv_other_count range: {df_compas_processed['juv_other_count'].min()} to {df_compas_processed['juv_other_count'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            df_compas_processed['juv_other_count'] == 0,
            df_compas_processed['juv_other_count'] == 1,
            df_compas_processed['juv_other_count'] >= 2
        ]
        choices = [0, 1, 2]  # 0='count_0', 1='count_1', 2='count_2+'
        
        df_compas_processed['juv_other_count_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Juvenile other categorization: {df_compas_processed['juv_other_count_cat'].value_counts().to_dict()}")
        log_func("Note: 0=count_0, 1=count_1, 2=count_2+")
    
    # Prior count categorization - return integers directly
    if 'priors_count' in df_compas_processed.columns:
        log_func("Processing priors_count column...")
        log_func(f"Original priors_count range: {df_compas_processed['priors_count'].min()} to {df_compas_processed['priors_count'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            df_compas_processed['priors_count'] == 0,
            df_compas_processed['priors_count'] == 1,
            (df_compas_processed['priors_count'] >= 2) & (df_compas_processed['priors_count'] <= 3),
            (df_compas_processed['priors_count'] >= 4) & (df_compas_processed['priors_count'] <= 6),
            (df_compas_processed['priors_count'] >= 7) & (df_compas_processed['priors_count'] <= 9),
            (df_compas_processed['priors_count'] >= 10) & (df_compas_processed['priors_count'] <= 15),
            df_compas_processed['priors_count'] > 15
        ]
        choices = [0, 1, 2, 3, 4, 5, 6]  # 0='count_0', 1='count_1', etc.
        
        df_compas_processed['priors_count_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Priors count categorization: {df_compas_processed['priors_count_cat'].value_counts().to_dict()}")
        log_func("Note: 0=count_0, 1=count_1, 2=count_2-3, 3=count_4-6, 4=count_7-9, 5=count_10-15, 6=count_15+")
    
    # Days between screening and arrest categorization - return integers directly
    if 'days_b_screening_arrest' in df_compas_processed.columns:
        log_func("Processing days_b_screening_arrest column...")
        log_func(f"Original days_b_screening_arrest range: {df_compas_processed['days_b_screening_arrest'].min()} to {df_compas_processed['days_b_screening_arrest'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            df_compas_processed['days_b_screening_arrest'] < -3,
            df_compas_processed['days_b_screening_arrest'] == -2,
            df_compas_processed['days_b_screening_arrest'] == -1,
            df_compas_processed['days_b_screening_arrest'] == 0,
            df_compas_processed['days_b_screening_arrest'] >= 1
        ]
        choices = [0, 1, 2, 3, 4]  # 0='lt_-3', 1='-2', 2='-1', 3='0', 4='1+'
        
        df_compas_processed['days_b_screening_arrest_cat'] = np.select(conditions, choices, default=3)
        log_func(f"Days between screening and arrest categorization: {df_compas_processed['days_b_screening_arrest_cat'].value_counts().to_dict()}")
        log_func("Note: 0=lt_-3, 1=-2, 2=-1, 3=0, 4=1+")
    
    # Days from COMPAS categorization - return integers directly
    if 'c_days_from_compas' in df_compas_processed.columns:
        log_func("Processing c_days_from_compas column...")
        log_func(f"Original c_days_from_compas range: {df_compas_processed['c_days_from_compas'].min()} to {df_compas_processed['c_days_from_compas'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            df_compas_processed['c_days_from_compas'] == 0,
            df_compas_processed['c_days_from_compas'] == 1,
            (df_compas_processed['c_days_from_compas'] >= 2) & (df_compas_processed['c_days_from_compas'] <= 7),
            (df_compas_processed['c_days_from_compas'] >= 8) & (df_compas_processed['c_days_from_compas'] <= 14),
            (df_compas_processed['c_days_from_compas'] >= 15) & (df_compas_processed['c_days_from_compas'] <= 21),
            (df_compas_processed['c_days_from_compas'] >= 22) & (df_compas_processed['c_days_from_compas'] <= 28),
            df_compas_processed['c_days_from_compas'] > 28
        ]
        choices = [0, 1, 2, 3, 4, 5, 6]  # 0='0', 1='1', 2='2-7', etc.
        
        df_compas_processed['c_days_from_compas_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Days from COMPAS categorization: {df_compas_processed['c_days_from_compas_cat'].value_counts().to_dict()}")
        log_func("Note: 0=0, 1=1, 2=2-7, 3=8-14, 4=15-21, 5=22-28, 6=29+")
    
    # Decile score categorization - return integers directly
    if 'decile_score' in df_compas_processed.columns:
        log_func("Processing decile_score column...")
        log_func(f"Original decile_score range: {df_compas_processed['decile_score'].min()} to {df_compas_processed['decile_score'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            (df_compas_processed['decile_score'] >= 1) & (df_compas_processed['decile_score'] <= 2),
            df_compas_processed['decile_score'] == 3,
            (df_compas_processed['decile_score'] >= 4) & (df_compas_processed['decile_score'] <= 5),
            (df_compas_processed['decile_score'] >= 6) & (df_compas_processed['decile_score'] <= 7),
            (df_compas_processed['decile_score'] >= 8) & (df_compas_processed['decile_score'] <= 10)
        ]
        choices = [0, 1, 2, 3, 4]  # 0='1-2', 1='3', 2='4-5', 3='6-7', 4='8-10'
        
        df_compas_processed['decile_score_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Decile score categorization: {df_compas_processed['decile_score_cat'].value_counts().to_dict()}")
        log_func("Note: 0=1-2, 1=3, 2=4-5, 3=6-7, 4=8-10")
    
    # Handle violence-related decile score if present - return integers directly
    if 'v_decile_score' in df_compas_processed.columns:
        log_func("Processing v_decile_score column...")
        log_func(f"Original v_decile_score range: {df_compas_processed['v_decile_score'].min()} to {df_compas_processed['v_decile_score'].max()}")
        
        # Create categorical bins and convert to integers immediately
        conditions = [
            (df_compas_processed['v_decile_score'] >= 1) & (df_compas_processed['v_decile_score'] <= 2),
            df_compas_processed['v_decile_score'] == 3,
            (df_compas_processed['v_decile_score'] >= 4) & (df_compas_processed['v_decile_score'] <= 5),
            (df_compas_processed['v_decile_score'] >= 6) & (df_compas_processed['v_decile_score'] <= 7),
            (df_compas_processed['v_decile_score'] >= 8) & (df_compas_processed['v_decile_score'] <= 10)
        ]
        choices = [0, 1, 2, 3, 4]  # 0='1-2', 1='3', 2='4-5', 3='6-7', 4='8-10'
        
        df_compas_processed['v_decile_score_cat'] = np.select(conditions, choices, default=0)
        log_func(f"Violence decile score categorization: {df_compas_processed['v_decile_score_cat'].value_counts().to_dict()}")
        log_func("Note: 0=1-2, 1=3, 2=4-5, 3=6-7, 4=8-10")
    
    # Step 4: Drop original numerical columns that we've categorized
    original_cols_to_drop = ['race', 'age', 'juv_fel_count', 'decile_score','juv_misd_count', 'juv_other_count', 'priors_count',
                            'days_b_screening_arrest', 'c_days_from_compas', 'v_decile_score']
    
    existing_original_cols_to_drop = [col for col in original_cols_to_drop if col in df_compas_processed.columns]
    if existing_original_cols_to_drop:
        df_compas_processed = df_compas_processed.drop(columns=existing_original_cols_to_drop)
        log_func(f"Dropped original numerical columns: {existing_original_cols_to_drop}")
    
    # Step 5: Drop outcome-related variables that might cause data leakage
    # Keep two_year_recid as target, but remove other outcome variables
    outcome_cols_to_drop = ['is_recid', 'is_violent_recid', 'r_charge_degree', 'vr_charge_degree', 
                           'r_days_from_arrest', 'event', 'start', 'end']
    
    existing_outcome_cols_to_drop = [col for col in outcome_cols_to_drop if col in df_compas_processed.columns]
    if existing_outcome_cols_to_drop:
        df_compas_processed = df_compas_processed.drop(columns=existing_outcome_cols_to_drop)
        log_func(f"Dropped outcome-related columns: {existing_outcome_cols_to_drop}")
    
    log_func(f"Final preprocessed shape: {df_compas_processed.shape}")
    log_func(f"Final columns: {list(df_compas_processed.columns)}")
    
    return df_compas_processed


def convert_columns_to_required_format(df: pd.DataFrame, 
                                     log_file_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Convert all columns in COMPAS dataset to numerical categorical format for causal analysis.
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
    log_func("COMPAS DATASET COLUMN CONVERSION LOG")
    log_func("=" * 60)
    
    # Process each column
    for col in df_converted.columns:
        log_func(f"\nProcessing column: {col}")
        log_func(f"Original data type: {df_converted[col].dtype}")
        
        # Show original unique values (limited for readability)
        original_unique = df_converted[col].dropna().unique()
        if len(original_unique) <= 20:
            log_func(f"Original unique values: {sorted(original_unique, key=str)}")
        else:
            log_func(f"Original unique values (first 20): {sorted(original_unique, key=str)[:20]}...")
        
        log_func(f"Original distribution: {df_converted[col].value_counts().to_dict()}")
        
        # Handle the target variable specially
        if col == 'two_year_recid':
            log_func("Processing target variable: two_year_recid")
            # Convert to string and handle various formats
            df_converted[col] = df_converted[col].astype(str).str.lower().str.strip()
            
            # Define mapping for common variations
            binary_mapping = {
                '1': 1, '1.0': 1, 'true': 1, 'yes': 1, 'y': 1,
                '0': 0, '0.0': 0, 'false': 0, 'no': 0, 'n': 0
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
                'type': 'binary_target',
                'mapping': binary_mapping,
                'distribution': df_converted[col].value_counts().to_dict()
            }
            
        elif df_converted[col].dtype == 'object' or df_converted[col].dtype.name == 'string':
            # Handle categorical text columns with label encoding
            log_func(f"Processing categorical column: {col}")
            
            # Convert to string and clean
            df_converted[col] = df_converted[col].astype(str).str.strip()
            
            # Handle missing values
            df_converted[col] = df_converted[col].replace(['nan', 'NaN', 'None', '', 'missing'], 'unknown')
            
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
            
        else:
            # Handle numerical columns (including our pre-processed categorical columns)
            log_func(f"Processing numerical column: {col}")
            
            # Convert to numeric and handle NaN
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            
            # Handle NaN values
            if df_converted[col].isna().any():
                nan_count = df_converted[col].isna().sum()
                # Use mode (most frequent value) for categorical data instead of median
                mode_val = df_converted[col].mode()
                if len(mode_val) > 0:
                    fill_val = mode_val.iloc[0]
                else:
                    fill_val = 0
                log_func(f"Warning: {nan_count} NaN values in {col}, filling with mode: {fill_val}")
                df_converted[col] = df_converted[col].fillna(fill_val)
            
            # Check if already categorical (small number of unique values)
            unique_count = df_converted[col].nunique()
            log_func(f"Unique values in {col}: {unique_count}")
            
            if unique_count <= 10:
                # If already few unique values, ensure it's properly encoded as integers
                log_func(f"Column already categorical (≤10 unique values)")
                
                # Ensure all values are integers starting from 0
                unique_values = sorted(df_converted[col].unique())
                value_mapping = {val: idx for idx, val in enumerate(unique_values)}
                df_converted[col] = df_converted[col].map(value_mapping)
                
                conversion_log[col] = {
                    'type': 'pre_processed_categorical',
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
                        'type': 'quartile_binning',
                        'original_unique_count': unique_count,
                        'mapping': 'quartile-based binning: 0=Q1, 1=Q2, 2=Q3, 3=Q4',
                        'distribution': df_converted[col].value_counts().to_dict()
                    }
                except ValueError as e:
                    # Fallback if quartile binning fails
                    log_func(f"Quartile binning failed for {col}: {e}")
                    log_func(f"Applying median split instead")
                    
                    median_val = df_converted[col].median()
                    df_converted[col] = (df_converted[col] > median_val).astype(int)
                    
                    conversion_log[col] = {
                        'type': 'median_binning',
                        'original_unique_count': unique_count,
                        'mapping': f'median split: 0=≤{median_val}, 1=>{median_val}',
                        'distribution': df_converted[col].value_counts().to_dict()
                    }
            
            # Ensure final column is integer
            df_converted[col] = df_converted[col].astype(int)
        
        # Final validation and logging
        final_values = sorted(df_converted[col].unique())
        log_func(f"Final values: {final_values}")
        log_func(f"Final data type: {df_converted[col].dtype}")
        log_func(f"Final distribution: {df_converted[col].value_counts().to_dict()}")
        log_func(f"Value range: {df_converted[col].min()} to {df_converted[col].max()}")
    
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
    
    return df_converted, conversion_log


def load_and_preprocess_data(file_path: str, export_path: str = None, log_file_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline for COMPAS dataset.
    
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
    
    log_func("COMPAS DATA LOADER")
    log_func("=" * 40)
    
    # Step 1: Load data
    try:
        df_original = pd.read_csv(file_path)
        log_func(f"Successfully loaded data from {file_path}")
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
        sample_values = df_original[col].dropna().unique()[:5]
        log_func(f"  {col}: {data_type}, {unique_count} unique values, samples: {sample_values}")
    
    # Step 3: Apply COMPAS-specific preprocessing
    log_func(f"\nApplying COMPAS-specific preprocessing...")
    df_preprocessed = apply_compas_preprocessing(df_original, log_func)
    
    # Step 4: Apply column conversions
    log_func(f"\nApplying comprehensive column conversions...")
    df_processed, conversion_log = _perform_column_conversion(
        df_preprocessed.copy(), 
        {}, 
        log_func
    )
    
    # Step 5: Validate the results
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
    log_func(f"  - After COMPAS preprocessing: {df_preprocessed.shape}")
    log_func(f"  - Final processed data: {df_processed.shape}")
    log_func(f"  - All variables converted to numerical categorical format")
    log_func(f"  - Ready for GES algorithm")
    
    # Export dataset if path is provided
    if export_path:
        df_processed.to_csv(export_path, index=False)
        log_func(f"Exported preprocessed dataset to {export_path}")
        
    return {
        'original': df_original,
        'compas_preprocessed': df_preprocessed,
        'preprocessed': df_processed,
        'conversion_log': conversion_log
    }


def get_compas_column_info() -> Dict[str, Dict[str, str]]:
    """
    Return information about COMPAS dataset columns after preprocessing.
    """
    return {
        'sex': {
            'type': 'categorical',
            'description': 'Gender encoded as 0-based integers',
            'values': 'varies (0-based)'
        },
        'race_cat': {
            'type': 'categorical',
            'description': 'Race category: 0=non_white, 1=white',
            'values': '0-1'
        },
        'age_binary': {
            'type': 'binary',
            'description': 'Age binary: 0=45 or older, 1=younger than 45',
            'values': '0-1'
        },
        'c_charge_degree': {
            'type': 'categorical',
            'description': 'Current charge degree encoded as 0-based integers',
            'values': 'varies (0-based)'
        },
        'juv_fel_count_cat': {
            'type': 'binary',
            'description': 'Juvenile felony count: 0=no felonies, 1=has felonies',
            'values': '0-1'
        },
        'juv_misd_count_cat': {
            'type': 'categorical',
            'description': 'Juvenile misdemeanor count: 0=count_0, 1=count_1, 2=count_2+',
            'values': '0-2'
        },
        'juv_other_count_cat': {
            'type': 'categorical',
            'description': 'Juvenile other count: 0=count_0, 1=count_1, 2=count_2+',
            'values': '0-2'
        },
        'priors_count_cat': {
            'type': 'categorical',
            'description': 'Prior count: 0=count_0, 1=count_1, 2=count_2-3, 3=count_4-6, 4=count_7-9, 5=count_10-15, 6=count_15+',
            'values': '0-6'
        },
        'days_b_screening_arrest_cat': {
            'type': 'categorical',
            'description': 'Days between screening and arrest: 0=lt_-3, 1=-2, 2=-1, 3=0, 4=1+',
            'values': '0-4'
        },
        'c_days_from_compas_cat': {
            'type': 'categorical',
            'description': 'Days from COMPAS: 0=0, 1=1, 2=2-7, 3=8-14, 4=15-21, 5=22-28, 6=29+',
            'values': '0-6'
        },
        'decile_score_cat': {
            'type': 'categorical',
            'description': 'COMPAS decile score: 0=1-2, 1=3, 2=4-5, 3=6-7, 4=8-10',
            'values': '0-4'
        },
        'two_year_recid': {
            'type': 'binary',
            'description': 'Target variable - two-year recidivism: 0=no, 1=yes',
            'values': '0-1'
        }
    }
