# src/faircausal/core/ml_classifer_runner.py
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import PrecisionRecallDisplay, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, classification_report, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier



def one_hot_encoding_dataframe_columns(df):
    """Function that one-hot encodes given columns"""
    ohc = OneHotEncoder()
    for col in df.columns:
        if (df[col].dtype == 'object') or (df[col].dtype.name == 'category'):
            df_ohc = pd.DataFrame(ohc.fit_transform(df[[col]]).toarray(), columns=ohc.categories_)

            df_ohc = df_ohc.add_prefix(col+'_')

            # # One-hot encode the column
            # df_ohc = pd.DataFrame(ohc.fit_transform(df[[col]]))

            # # Flatten multi-level column index
            # df_ohc.columns = [f"{col}_{category}" for category in ohc.categories_[0]]

            # Drop the original column and concatenate the one-hot encoded data
            df = pd.concat([df.drop([col], axis=1), df_ohc], axis=1)
    return df

   
def preprocess_data(df, cols_ohe, target_name):
    """Function that pre-processes the data, ready for modelling"""
    # Make a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Use the OneHotEncoder as in your original function
    ohc = OneHotEncoder()
    for col in cols_ohe:
        if col in df.columns:
            # Create a DataFrame with the one-hot encoded columns
            encoded_cols = pd.DataFrame(
                ohc.fit_transform(df[[col]]).toarray(),
                columns=[(f"{col}_{val}",) for val in ohc.categories_[0]]  # Format: ('col_val',)
            )
            
            # Print the encoded column names for debugging
            print(f"One-hot encoded {col} into: {encoded_cols.columns.tolist()}")
            
            # Concatenate with the processed DataFrame
            df_processed = pd.concat([df_processed.drop(col, axis=1), encoded_cols], axis=1)
    
    # Ensure all columns are strings
    df_processed.columns = df_processed.columns.map(str)
    
    # Print column names for debugging
    print(f"First 5 columns after encoding: {df_processed.columns.tolist()[:5]}")
    
    # Split data for modeling
    X = df_processed.drop([target_name], axis=1)
    y = df_processed[target_name]
    
    return df_processed, X, y

##-----------------------------------
## 1. Build logistic regression model
##-----------------------------------
def build_logistic_regression_model(X_train,  X_test, y_train, y_test, cutoff_threshold=0.5):

    # Build logistic regression model
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    
    # Get prediction probabilities for the positive class (label = 1)
    y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]
    
    # Apply custom cutoff threshold for label = 1
    y_pred = (y_pred_proba >= cutoff_threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # accuracy_score = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)


    return logreg_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced


##-----------------------------------
## 2. Build Random Forest model
##-----------------------------------
def build_random_forest_model(X_train,  X_test, y_train, y_test, cutoff_threshold=0.5):

    # Create a random forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    rf_model.fit(X_train, y_train)
    
    # Get prediction probabilities for the positive class (label = 1)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Apply custom cutoff threshold for label = 1
    y_pred = (y_pred_proba >= cutoff_threshold).astype(int)

    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
    
    return rf_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced


##-----------------------------------
## 3. Build XGBoost model
##-----------------------------------
def build_xgboost_model(X_train,  X_test, y_train, y_test, cutoff_threshold=0.5):
    # Create an XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    
    # Fit the model to the training data
    xgb_model.fit(X_train, y_train)
    
    # Get prediction probabilities for the positive class (label = 1)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Apply custom cutoff threshold for label = 1
    y_pred = (y_pred_proba >= cutoff_threshold).astype(int)
    
    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)
    
    return xgb_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced

##-----------------------------------
## 4. Build MLP model
##-----------------------------------
def build_mlp_model(X_train,  X_test, y_train, y_test, cutoff_threshold=0.5):
    # Create an MLP classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)

    # Fit the model to the training data
    mlp_model.fit(X_train, y_train)

    # Get prediction probabilities for the positive class (label = 1)
    y_pred_proba = mlp_model.predict_proba(X_test)[:, 1]
    
    # Apply custom cutoff threshold for label = 1
    y_pred = (y_pred_proba >= cutoff_threshold).astype(int)

    # Calculate the evaluation metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_balanced = balanced_accuracy_score(y_test, y_pred)

    return mlp_model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced




##-----------------------------------
## Test fairness of model
##-----------------------------------

def calc_fairness_metrics(y, y_pred, sf_values):
    # Debug the sensitive features
    print(f"\nDEBUG: calc_fairness_metrics called")
    print(f"  y shape: {np.array(y).shape}, unique values: {np.unique(y)}")
    print(f"  y_pred shape: {np.array(y_pred).shape}, unique values: {np.unique(y_pred)}")
    print(f"  sf_values type: {type(sf_values)}, length: {len(sf_values) if hasattr(sf_values, '__len__') else 'No length'}")
    
    if hasattr(sf_values, '__len__') and len(sf_values) > 0:
        unique_sf = np.unique(sf_values)
        print(f"  sf_values unique: {unique_sf}")
        print(f"  sf_values sample: {sf_values[:10] if len(sf_values) >= 10 else sf_values}")
        
        if len(unique_sf) <= 1:
            print(f"  ⚠️  WARNING: Only {len(unique_sf)} unique value(s) in sensitive features!")
            print(f"  This will cause fairness metrics to be 0!")

    ##-------------
    ## 1. Demographic parity (DP), (TP + FP)
    ##-------------
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
    ## Check if y_pred only contain one class
    if len(np.unique(y_pred)) > 1:
        DPD = demographic_parity_difference(y, y_pred, sensitive_features=sf_values)
        DPR = demographic_parity_ratio(y, y_pred, sensitive_features=sf_values)
    else:
        DPD = 0
        DPR = 0

    ##-------------
    ## 2. Equalized odds (EO) FPR, TP / (TP + FN)
    ##-------------
    from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
    if len(np.unique(y_pred)) > 1:
        EO = equalized_odds_difference(y, y_pred,sensitive_features=sf_values)
    else:
        EO = 0

    print(f"  Calculated fairness metrics: DPD={DPD}, DPR={DPR}, EO={EO}")
    
    return DPD, DPR, EO


def calc_ftu(model, X, y, benchmark_col, protected_col):

    ## Over-write all the protected column to benchmark population value
    X_benchmark = X.copy()
    X_benchmark[benchmark_col] = 1
    X_benchmark[protected_col] = 0

    ## Score the dataset
    benchmark_pred = model.predict(X_benchmark)
    benchmark_pred_label = (benchmark_pred >= 0.5).astype(int)

    ## Over-write all the protected column to protected population value
    X_protected = X.copy()
    X_protected[protected_col] = 1
    X_protected[benchmark_col] = 0

    ## Score the dataset
    protected_pred = model.predict(X_protected)
    protected_pred_label = (protected_pred >= 0.5).astype(int)

    ## Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm1=confusion_matrix(y, benchmark_pred_label)
    TN1, FP1, FN1, TP1 = cm1.ravel()
    ftu1 = (TP1+FP1)/(TP1+FP1+FN1+TN1)
    # print(ftu1)

    cm2=confusion_matrix(y, protected_pred_label)
    TN2, FP2, FN2, TP2 = cm2.ravel()
    ftu2 = (TP2+FP2)/(TP2+FP2+FN2+TN2)
    # print(ftu2)
    
    ftu_ratio =  ftu1/ftu2
    ftu_diff = abs(ftu1-ftu2)
 
    ## return the scored dataset
    return ftu_diff


## Function that takes in a list and convert them into mean ± std format
def mean_plus_minus_std(values):
    # Calculating the mean and standard deviation of the values in the list
    mean_value = np.mean(values)
    std_value = np.std(values)
    
    # Formatting the mean and standard deviation to 4 decimal places
    formatted_mean = f"{mean_value:.4f}"
    formatted_std = f"{std_value:.4f}"
    
    # Returning the combined formatted string
    return f"{formatted_mean} ± {formatted_std}"


def revert_one_hot_encoding(df, benchmark_col, protected_col):
    """Function to extract the sensitive feature values from one-hot encoded columns"""
    
    # FIXED: Check if the columns exist directly in the dataframe (not one-hot encoded)
    if benchmark_col in df.columns:
        values = df[benchmark_col].tolist()
        return values
    elif protected_col in df.columns:
        values = df[protected_col].tolist()
        return values
    
    # Original logic for one-hot encoded data
    sf_cols = [benchmark_col, protected_col]
    original_list = []
    
    # Remove parentheses and quotes from column names for comparison
    clean_sf_cols = [col.strip("(),'\"") for col in sf_cols]
    
    # Check if any column names in df match the pattern
    found_cols = []
    for col in df.columns:
        col_str = str(col)
        for clean_col in clean_sf_cols:
            if clean_col in col_str:
                found_cols.append(col)
                break
    
    # If no columns found, return default values
    if not found_cols:
        print(f"Warning: Could not find columns matching {sf_cols}")
        return ['unknown'] * len(df)
    
    # Extract values based on which column has 1
    for index, row in df.iterrows():
        value_found = False
        for col in found_cols:
            if row[col] == 1:
                # Extract the value from the column name (format: 'attr_value')
                col_str = str(col).strip("(),'\"")
                parts = col_str.split('_')
                if len(parts) > 1:
                    value = parts[-1]  # Last part is the value
                    original_list.append(value)
                    value_found = True
                    break
        
        # If no value found for this row, use a default
        if not value_found:
            original_list.append('unknown')
    
    return original_list

## Cross validation
def perform_cv(X_tr, y_tr, X_te, y_te, model_type, protected_attrs, df_original, benchmark_col=None, protected_col=None, sf_col_name=None, random_state=None, n_splits=5, sf_indata='Y', cutoff_threshold=0.5):

    # Stratified 5-fold cross-validation
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)

    auroc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_balanced_scores = []
    dpd_scores = {attr: [] for attr in protected_attrs}
    eo_scores = {attr: [] for attr in protected_attrs}
    ftu_scores = []
    
    for train_idx, test_idx in cv.split(X_tr, y_tr):

        ## If there is not a seperate testing data, we perform cross validation on the training data.
        if X_tr.equals(X_te):
            X_train, X_test = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
            y_train, y_test = y_tr.iloc[train_idx], y_tr.iloc[test_idx]

        ## If there is a seperate testing data, we train the model with the cross validation training data 
        ## and test it on the cross validited seperate testing data.
        else:
            ## Create model training data based on the CV
            X_train, X_test_tmp = X_tr.iloc[train_idx], X_tr.iloc[test_idx]
            y_train, y_test_tmp = y_tr.iloc[train_idx], y_tr.iloc[test_idx]
            ## Seperately, split the training data
            for train_idx, test_idx in cv.split(X_te, y_te):
                X_train_tmp, X_test = X_te.iloc[train_idx], X_te.iloc[test_idx]
                y_train_tmp, y_test = y_te.iloc[train_idx], y_te.iloc[test_idx]


        ##-----------------------------------
        # 1. Calculate model performance metrics
        ##-----------------------------------
        if model_type == "logistic_regression":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_logistic_regression_model(X_train,  X_test, y_train, y_test, cutoff_threshold)
        elif model_type == "random_forest":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_random_forest_model(X_train,  X_test, y_train, y_test, cutoff_threshold)
        elif model_type == "xgboost":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_xgboost_model(X_train,  X_test, y_train, y_test, cutoff_threshold)
        elif model_type == "mlp":
            model, y_pred, precision, recall, f1, roc_auc, accuracy_balanced = build_mlp_model(X_train,  X_test, y_train, y_test, cutoff_threshold)
        else:
            raise ValueError("Invalid model type. Choose 'logistic_regression', 'random_forest', 'xgboost' or 'mlp'.")
        
        auroc_scores.append(roc_auc)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_balanced_scores.append(accuracy_balanced)

        ##-----------------------------------
        # 2. Calculate fairness metrics per protected attribute
        ##-----------------------------------
        src = X_test if sf_indata == 'Y' else df_original.iloc[test_idx]
        for attr in protected_attrs:
            sf_data = revert_one_hot_encoding(src, attr, attr)
            DPD, _DPR, EO = calc_fairness_metrics(y_test, y_pred, sf_data)
            dpd_scores[attr].append(DPD)
            eo_scores[attr].append(EO)

        if sf_indata == 'Y' and benchmark_col is not None:
            FTU = calc_ftu(model, X_test, y_test, benchmark_col, protected_col or benchmark_col)
        else:
            FTU = 0
        ftu_scores.append(FTU)


    result = {
        "Precision": mean_plus_minus_std(precision_scores),
        "Recall": mean_plus_minus_std(recall_scores),
        "F1": mean_plus_minus_std(f1_scores),
        "AUROC": mean_plus_minus_std(auroc_scores),
        "Balanced Accuracy": mean_plus_minus_std(accuracy_balanced_scores),
    }
    for attr in protected_attrs:
        result[f"DPD_{attr}"] = mean_plus_minus_std(dpd_scores[attr])
        result[f"EO_{attr}"] = mean_plus_minus_std(eo_scores[attr])
    return result

##-----------------------------------
## Combine them all together
##-----------------------------------
def run_samples(sample_name, df_model, target_name, X_tr, y_tr, X_te, y_te, protected_attrs, df_original, benchmark_col=None, protected_col=None, sf_col_name=None, random_state=None, sf_indata='Y', cutoff_threshold=0.5):

    def _cv(model_type):
        return perform_cv(
            X_tr, y_tr, X_te, y_te, model_type, protected_attrs, df_original,
            benchmark_col=benchmark_col, protected_col=protected_col,
            sf_col_name=sf_col_name, random_state=random_state,
            sf_indata=sf_indata, cutoff_threshold=cutoff_threshold,
        )

    def _make_row(model_type, res):
        row = {
            'sample_name': sample_name,
            'model_type': model_type,
            'precision': res.get('Precision'),
            'recall': res.get('Recall'),
            'f1': res.get('F1'),
            'AUROC': res.get('AUROC'),
            'accuracy': res.get('Balanced Accuracy'),
        }
        for key, val in res.items():
            if key.startswith('DPD_') or key.startswith('EO_'):
                row[key] = val
        return row

    rows = [
        _make_row('logistic_regression', _cv('logistic_regression')),
        _make_row('random_forest', _cv('random_forest')),
        _make_row('xgboost', _cv('xgboost')),
        _make_row('mlp', _cv('mlp')),
    ]
    return pd.DataFrame(rows)

import pandas as pd
import numpy as np
import xgboost as xgb

def debug_protected_attributes(df, benchmark_col, protected_col, sf_values):
    """
    Debug function to investigate why fairness metrics are returning 0.
    """
    print(f"\n{'='*60}")
    print("DEBUGGING PROTECTED ATTRIBUTES")
    print(f"{'='*60}")
    
    print(f"Benchmark column: '{benchmark_col}'")
    print(f"Protected column: '{protected_col}'") 
    print(f"Sensitive features type: {type(sf_values)}")
    print(f"Sensitive features length: {len(sf_values) if hasattr(sf_values, '__len__') else 'No length'}")
    
    # Check if columns exist in dataframe
    print(f"\nDataFrame columns: {list(df.columns)}")
    print(f"Benchmark col '{benchmark_col}' in df: {benchmark_col in df.columns}")
    print(f"Protected col '{protected_col}' in df: {protected_col in df.columns}")
    
    # Show sample of sf_values
    if hasattr(sf_values, '__len__') and len(sf_values) > 0:
        unique_sf_values = np.unique(sf_values)
        print(f"Unique values in sf_values: {unique_sf_values}")
        print(f"First 10 sf_values: {sf_values[:10] if len(sf_values) >= 10 else sf_values}")
    
    # Check if all values are the same (which would cause 0 fairness metrics)
    if hasattr(sf_values, '__len__') and len(sf_values) > 0:
        unique_count = len(np.unique(sf_values))
        print(f"Number of unique values in sf_values: {unique_count}")
        if unique_count <= 1:
            print("⚠️  WARNING: Only one unique value in sensitive features - this causes fairness metrics to be 0!")
    
    print(f"{'='*60}\n")
    
    return unique_count if hasattr(sf_values, '__len__') and len(sf_values) > 0 else 0

