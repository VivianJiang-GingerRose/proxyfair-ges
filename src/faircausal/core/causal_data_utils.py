import pandas as pd
import numpy as np
from typing import List, Tuple

def prepare_data_for_ges(df: pd.DataFrame, selected_columns: List[str] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Extract columns from dataset and prepare them for GES.
    Assumes data is already properly formatted by preprocessing.
    """
    # Get selected columns or use all
    if selected_columns is None:
        selected_columns = df.columns.tolist()
    else:
        selected_columns = [col for col in selected_columns if col in df.columns]
        missing = [col for col in (selected_columns if selected_columns else []) if col not in df.columns]
        if missing:
            print(f"Warning: The following columns are missing: {missing}")
    
    # Create subset with selected columns
    df_subset = df[selected_columns].copy()
    
    print(f"Preparing data for GES:")
    print(f"  Selected columns: {selected_columns}")
    print(f"  Data shape: {df_subset.shape}")
    
    # Ensure all columns are numeric
    for col in df_subset.columns:
        if df_subset[col].dtype not in ['int64', 'int32', 'float64', 'float32']:
            print(f"Warning: Column {col} has unexpected dtype {df_subset[col].dtype}")
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')
    
    # Convert to numpy array
    X = df_subset.values
    column_names = df_subset.columns.tolist()
    
    print(f"Prepared data for GES with shape: {X.shape}")
    
    return X, column_names


def prepare_data_for_scm(df: pd.DataFrame, selected_columns: List[str] = None, 
                        categorical_columns: List[str] = None) -> pd.DataFrame:
    """
    Prepare data specifically for SCM fitting by converting categorical columns to strings.
    """
    # Get selected columns or use all
    if selected_columns is None:
        selected_columns = df.columns.tolist()
    else:
        selected_columns = [col for col in selected_columns if col in df.columns]
    
    df_scm = df[selected_columns].copy()
    
    print(f"Preparing data for SCM:")
    print(f"  Selected columns: {selected_columns}")
    print(f"  Data shape: {df_scm.shape}")
    
    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = auto_detect_categorical_columns(df_scm)
    
    # Convert categorical columns to strings
    for col in categorical_columns:
        if col in df_scm.columns:
            print(f"  Converting {col} to categorical strings")
            df_scm[col] = df_scm[col].astype(str)
    
    print(f"SCM data preparation complete")
    return df_scm


def auto_detect_categorical_columns(df: pd.DataFrame, max_unique_ratio: float = 0.1, 
                                   max_unique_count: int = 20) -> List[str]:
    """
    Auto-detect categorical columns based on cardinality and data type.
    """
    categorical_columns = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        # Consider categorical if:
        # 1. Low cardinality (few unique values)
        # 2. Low unique ratio relative to total rows
        # 3. Integer type with reasonable range
        if (unique_count <= max_unique_count and 
            unique_ratio <= max_unique_ratio and 
            df[col].dtype in ['int64', 'int32', 'object']):
            categorical_columns.append(col)
    
    return categorical_columns


def get_data_for_algorithm(df: pd.DataFrame, algorithm_type: str = 'ges', 
                          selected_columns: List[str] = None, 
                          categorical_columns: List[str] = None):
    """
    Universal function to get appropriately formatted data for different algorithms.
    """
    if algorithm_type.lower() == 'ges':
        return prepare_data_for_ges(df, selected_columns)
    elif algorithm_type.lower() == 'scm':
        return prepare_data_for_scm(df, selected_columns, categorical_columns)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}. Supported: 'ges', 'scm'")
    