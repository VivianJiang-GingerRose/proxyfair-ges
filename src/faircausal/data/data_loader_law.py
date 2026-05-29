# src/faircausal/data/data_loader_law.py

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
        self.log_file.write(f"LAW SCHOOL DATA PREPROCESSING LOG\n")
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


def apply_law_school_preprocessing(df: pd.DataFrame, log_func) -> pd.DataFrame:
    """
    Apply Law School-specific preprocessing steps with proper column handling
    """
    log_func("Applying Law School-specific preprocessing steps...")
    
    # Start with the original dataframe
    df_law = df.copy()
    
    # Log initial state
    log_func(f"Initial shape: {df_law.shape}")
    log_func(f"Initial columns: {list(df_law.columns)}")
    
    # Step 1: Convert decile variables to categorical (already in 1-10 range)
    if 'decile1b' in df_law.columns:
        log_func("Processing decile1b column...")
        log_func(f"Original decile1b range: {df_law['decile1b'].min()} to {df_law['decile1b'].max()}")
        # Convert to integers (already in 1-10 range)
        df_law['decile1b_cat'] = df_law['decile1b'].astype(int)
        log_func(f"Decile1b categorization: {df_law['decile1b_cat'].value_counts().sort_index().to_dict()}")
    
    if 'decile3' in df_law.columns:
        log_func("Processing decile3 column...")
        log_func(f"Original decile3 range: {df_law['decile3'].min()} to {df_law['decile3'].max()}")
        # Convert to integers (already in 1-10 range)
        df_law['decile3_cat'] = df_law['decile3'].astype(int)
        log_func(f"Decile3 categorization: {df_law['decile3_cat'].value_counts().sort_index().to_dict()}")
    
    # Step 2: Bin LSAT scores (range: 11-48)
    if 'lsat' in df_law.columns:
        log_func("Processing LSAT column...")
        log_func(f"Original LSAT range: {df_law['lsat'].min()} to {df_law['lsat'].max()}")
        
        # Create 7 bins for LSAT scores based on distribution
        lsat_bins = [10, 25, 30, 35, 40, 45, 50]  # 6 categories
        lsat_labels = [0, 1, 2, 3, 4, 5]  # 0-based encoding
        
        df_law['lsat_cat'] = pd.cut(df_law['lsat'], bins=lsat_bins, labels=lsat_labels, include_lowest=True)
        df_law['lsat_cat'] = df_law['lsat_cat'].astype(int)
        log_func(f"LSAT categorization: {df_law['lsat_cat'].value_counts().sort_index().to_dict()}")
        log_func("LSAT bins: 0=[11-25], 1=[26-30], 2=[31-35], 3=[36-40], 4=[41-45], 5=[46-48]")
    
    # Step 3: Bin UGPA scores (range: 1.5-4.0)
    if 'ugpa' in df_law.columns:
        log_func("Processing UGPA column...")
        log_func(f"Original UGPA range: {df_law['ugpa'].min()} to {df_law['ugpa'].max()}")
        
        # Create 5 bins for UGPA scores
        ugpa_bins = [1.4, 2.5, 3.0, 3.3, 3.7, 4.1]  # 5 categories
        ugpa_labels = [0, 1, 2, 3, 4]  # 0-based encoding
        
        df_law['ugpa_cat'] = pd.cut(df_law['ugpa'], bins=ugpa_bins, labels=ugpa_labels, include_lowest=True)
        df_law['ugpa_cat'] = df_law['ugpa_cat'].astype(int)
        log_func(f"UGPA categorization: {df_law['ugpa_cat'].value_counts().sort_index().to_dict()}")
        log_func("UGPA bins: 0=[1.5-2.5], 1=[2.6-3.0], 2=[3.1-3.3], 3=[3.4-3.7], 4=[3.8-4.0]")
    
    # Step 4: Bin ZFYGPA scores (normalized first-year GPA, range: -3.35 to 3.48)
    if 'zfygpa' in df_law.columns:
        log_func("Processing ZFYGPA column...")
        log_func(f"Original ZFYGPA range: {df_law['zfygpa'].min()} to {df_law['zfygpa'].max()}")
        
        # Create 6 bins for ZFYGPA scores based on distribution
        zfygpa_bins = [-4.0, -1.0, -0.3, 0.3, 1.0, 2.0, 4.0]  # 6 categories
        zfygpa_labels = [0, 1, 2, 3, 4, 5]  # 0-based encoding
        
        df_law['zfygpa_cat'] = pd.cut(df_law['zfygpa'], bins=zfygpa_bins, labels=zfygpa_labels, include_lowest=True)
        df_law['zfygpa_cat'] = df_law['zfygpa_cat'].astype(int)
        log_func(f"ZFYGPA categorization: {df_law['zfygpa_cat'].value_counts().sort_index().to_dict()}")
        log_func("ZFYGPA bins: 0=[-3.35 to -1.0], 1=[-1.0 to -0.3], 2=[-0.3 to 0.3], 3=[0.3 to 1.0], 4=[1.0 to 2.0], 5=[2.0 to 3.48]")
    
    # Step 5: Bin ZGPA scores (normalized cumulative GPA, range: -6.44 to 4.01)
    if 'zgpa' in df_law.columns:
        log_func("Processing ZGPA column...")
        log_func(f"Original ZGPA range: {df_law['zgpa'].min()} to {df_law['zgpa'].max()}")
        
        # Create 6 bins for ZGPA scores based on distribution
        zgpa_bins = [-7.0, -1.2, -0.4, 0.4, 1.2, 2.5, 5.0]  # 6 categories
        zgpa_labels = [0, 1, 2, 3, 4, 5]  # 0-based encoding
        
        df_law['zgpa_cat'] = pd.cut(df_law['zgpa'], bins=zgpa_bins, labels=zgpa_labels, include_lowest=True)
        df_law['zgpa_cat'] = df_law['zgpa_cat'].astype(int)
        log_func(f"ZGPA categorization: {df_law['zgpa_cat'].value_counts().sort_index().to_dict()}")
        log_func("ZGPA bins: 0=[-6.44 to -1.2], 1=[-1.2 to -0.4], 2=[-0.4 to 0.4], 3=[0.4 to 1.2], 4=[1.2 to 2.5], 5=[2.5 to 4.01]")
    
    # Step 6: Process race variable (already encoded as White/Non-White)
    if 'race' in df_law.columns:
        log_func("Processing race column...")
        log_func(f"Original race values: {df_law['race'].value_counts().to_dict()}")
        df_law['race_cat'] = np.where(
            df_law['race'] == 'White', 
            1,  # 1 = white
            0   # 0 = non_white
        )
        log_func(f"Race categorization: {df_law['race_cat'].value_counts().to_dict()}")
        log_func("Note: 1 = White, 0 = Non-White")
    
    # Step 7: Process gender (male column is already binary)
    if 'male' in df_law.columns:
        log_func("Processing male column...")
        log_func(f"Original male values: {df_law['male'].value_counts().to_dict()}")
        df_law['male_cat'] = df_law['male'].astype(int)
        log_func(f"Male categorization: {df_law['male_cat'].value_counts().to_dict()}")
        log_func("Note: 1 = Male, 0 = Female")
    
    # Step 8: Process fulltime (already binary 1/2, convert to 0/1)
    if 'fulltime' in df_law.columns:
        log_func("Processing fulltime column...")
        log_func(f"Original fulltime values: {df_law['fulltime'].value_counts().to_dict()}")
        # Convert 1->1 (fulltime), 2->0 (part-time)
        df_law['fulltime_cat'] = np.where(df_law['fulltime'] == 1, 1, 0)
        log_func(f"Fulltime categorization: {df_law['fulltime_cat'].value_counts().to_dict()}")
        log_func("Note: 1 = Full-time, 0 = Part-time")
    
    # Step 9: Process family income (already categorical 1-5)
    if 'fam_inc' in df_law.columns:
        log_func("Processing fam_inc column...")
        log_func(f"Original fam_inc values: {df_law['fam_inc'].value_counts().sort_index().to_dict()}")
        df_law['fam_inc_cat'] = df_law['fam_inc'].astype(int)
        log_func(f"Family income categorization: {df_law['fam_inc_cat'].value_counts().sort_index().to_dict()}")
        log_func("Note: Family income bracket (1-5)")
    
    # Step 10: Process tier (already categorical 1-6)
    if 'tier' in df_law.columns:
        log_func("Processing tier column...")
        log_func(f"Original tier values: {df_law['tier'].value_counts().sort_index().to_dict()}")
        df_law['tier_cat'] = df_law['tier'].astype(int)
        log_func(f"Tier categorization: {df_law['tier_cat'].value_counts().sort_index().to_dict()}")
        log_func("Note: School tier (1-6)")
    
    # Step 11: Process target variable (pass_bar)
    if 'pass_bar' in df_law.columns:
        log_func("Processing pass_bar column...")
        log_func(f"Original pass_bar values: {df_law['pass_bar'].value_counts().to_dict()}")
        df_law['pass_bar'] = df_law['pass_bar'].astype(int)
        log_func(f"Pass bar target: {df_law['pass_bar'].value_counts().to_dict()}")
        log_func("Note: 1 = Passed bar exam, 0 = Failed bar exam")
    
    # Drop original numerical columns, keep only categorical versions
    columns_to_drop = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'race', 'male', 'fulltime', 'fam_inc', 'tier']
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_law.columns]
    if existing_cols_to_drop:
        df_law = df_law.drop(columns=existing_cols_to_drop)
        log_func(f"Dropped original columns: {existing_cols_to_drop}")
    
    log_func(f"Shape after preprocessing: {df_law.shape}")
    log_func(f"Final columns: {list(df_law.columns)}")
    
    return df_law


def convert_law_school_columns_to_numerical(df: pd.DataFrame, log_file_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Convert all columns in Law School dataset to numerical categorical format for causal analysis.
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
    log_func("LAW SCHOOL DATASET COLUMN CONVERSION LOG")
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
        if col == 'pass_bar':
            log_func("Processing target variable: pass_bar")
            # Ensure it's properly encoded as 0/1
            df_converted[col] = df_converted[col].astype(int)
            
            conversion_log[col] = {
                'type': 'binary_target',
                'mapping': {0: 0, 1: 1},
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
            
            conversion_log[col] = {
                'type': 'categorical',
                'mapping': value_mapping,
                'reverse_mapping': {v: k for k, v in value_mapping.items()},
                'distribution': df_converted[col].value_counts().to_dict()
            }
            
        else:
            # Handle numerical columns (ensure they are integers)
            log_func(f"Processing numerical column: {col}")
            
            # Handle missing values by filling with mode
            if df_converted[col].isna().any():
                mode_value = df_converted[col].mode().iloc[0] if not df_converted[col].mode().empty else 0
                df_converted[col] = df_converted[col].fillna(mode_value)
                log_func(f"Filled {df_converted[col].isna().sum()} missing values with mode: {mode_value}")
            
            # Convert to integer
            df_converted[col] = df_converted[col].astype(int)
            
            conversion_log[col] = {
                'type': 'numerical_categorical',
                'range': f"{df_converted[col].min()}-{df_converted[col].max()}",
                'distribution': df_converted[col].value_counts().to_dict()
            }
        
        log_func(f"Converted data type: {df_converted[col].dtype}")
        log_func(f"Final unique values: {sorted(df_converted[col].unique())}")
        log_func(f"Final distribution: {df_converted[col].value_counts().to_dict()}")
    
    log_func("\n" + "=" * 60)
    log_func("CONVERSION SUMMARY")
    log_func("=" * 60)
    
    log_func(f"Total columns processed: {len(df_converted.columns)}")
    log_func(f"Final data shape: {df_converted.shape}")
    
    # Verify all columns are numerical
    non_numerical_cols = []
    for col in df_converted.columns:
        if df_converted[col].dtype not in ['int64', 'int32', 'int16', 'int8']:
            non_numerical_cols.append(col)
    
    if non_numerical_cols:
        log_func(f"WARNING: Non-numerical columns found: {non_numerical_cols}")
    else:
        log_func("SUCCESS: All columns are numerical!")
    
    # Final verification
    log_func(f"\nFinal column types:")
    for col in df_converted.columns:
        log_func(f"  {col}: {df_converted[col].dtype}")
    
    return df_converted, conversion_log


def load_and_preprocess_data(file_path: str, export_path: str = None, log_file_path: str = None) -> Dict[str, pd.DataFrame]:
    """
    Complete data loading and preprocessing pipeline for Law School dataset.
    
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
    
    log_func("LAW SCHOOL DATA LOADER")
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
    
    # Step 3: Apply Law School-specific preprocessing
    log_func(f"\nApplying Law School-specific preprocessing...")
    df_preprocessed = apply_law_school_preprocessing(df_original, log_func)
    
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
        log_func(f"WARNING: Found non-numerical columns: {non_numerical_cols}")
        # Try to convert them
        for col in non_numerical_cols:
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0).astype(int)
                log_func(f"Successfully converted {col} to numerical")
            except Exception as e:
                log_func(f"Failed to convert {col}: {e}")
    else:
        log_func("All columns are numerical - ready for analysis!")
    
    # Check for missing values
    missing_counts = df_processed.isnull().sum()
    if missing_counts.sum() > 0:
        log_func(f"\nMissing values found:")
        for col, count in missing_counts[missing_counts > 0].items():
            log_func(f"  {col}: {count} missing values")
    else:
        log_func("No missing values found!")
    
    # Check for target variable balance
    if 'pass_bar' in df_processed.columns:
        target_dist = df_processed['pass_bar'].value_counts()
        log_func(f"\nTarget variable (pass_bar) distribution:")
        for value, count in target_dist.items():
            percentage = (count / len(df_processed)) * 100
            log_func(f"  {value}: {count} ({percentage:.1f}%)")
    
    # Step 6: Export if requested
    if export_path:
        try:
            df_processed.to_csv(export_path, index=False)
            log_func(f"\nExported processed data to: {export_path}")
        except Exception as e:
            log_func(f"Error exporting data: {e}")
    
    # Return both original and processed data
    result = {
        'original': df_original,
        'preprocessed': df_preprocessed,
        'processed': df_processed,
        'conversion_log': conversion_log
    }
    
    log_func(f"\nData loading and preprocessing completed successfully!")
    return result


def get_data_schema() -> Dict[str, Dict[str, str]]:
    """
    Returns schema information for the processed Law School dataset
    """
    return {
        'decile1b_cat': {
            'type': 'categorical',
            'description': 'First year decile ranking in school (1-10)',
            'values': '1-10'
        },
        'decile3_cat': {
            'type': 'categorical', 
            'description': 'Third year decile ranking in school (1-10)',
            'values': '1-10'
        },
        'lsat_cat': {
            'type': 'categorical',
            'description': 'LSAT score bins: 0=[11-25], 1=[26-30], 2=[31-35], 3=[36-40], 4=[41-45], 5=[46-48]',
            'values': '0-5'
        },
        'ugpa_cat': {
            'type': 'categorical',
            'description': 'Undergraduate GPA bins: 0=[1.5-2.5], 1=[2.6-3.0], 2=[3.1-3.3], 3=[3.4-3.7], 4=[3.8-4.0]',
            'values': '0-4'
        },
        'zfygpa_cat': {
            'type': 'categorical',
            'description': 'Normalized first-year law GPA bins: 0=[-3.35 to -1.0], 1=[-1.0 to -0.3], 2=[-0.3 to 0.3], 3=[0.3 to 1.0], 4=[1.0 to 2.0], 5=[2.0 to 3.48]',
            'values': '0-5'
        },
        'zgpa_cat': {
            'type': 'categorical',
            'description': 'Normalized cumulative law GPA bins: 0=[-6.44 to -1.2], 1=[-1.2 to -0.4], 2=[-0.4 to 0.4], 3=[0.4 to 1.2], 4=[1.2 to 2.5], 5=[2.5 to 4.01]',
            'values': '0-5'
        },
        'race_cat': {
            'type': 'binary',
            'description': 'Race: 0=Non-White, 1=White',
            'values': '0-1'
        },
        'male_cat': {
            'type': 'binary',
            'description': 'Gender: 0=Female, 1=Male',
            'values': '0-1'
        },
        'fulltime_cat': {
            'type': 'binary',
            'description': 'Work schedule: 0=Part-time, 1=Full-time',
            'values': '0-1'
        },
        'fam_inc_cat': {
            'type': 'categorical',
            'description': 'Family income bracket (1-5)',
            'values': '1-5'
        },
        'tier_cat': {
            'type': 'categorical',
            'description': 'School tier (1-6)',
            'values': '1-6'
        },
        'pass_bar': {
            'type': 'binary',
            'description': 'Target variable - bar exam pass: 0=Failed, 1=Passed',
            'values': '0-1'
        }
    }
