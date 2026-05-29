# fairness_metrics.py

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def calculate_individual_fairness(df, protected_attrs, target):
    """
    Calculates individual fairness metrics on a given dataset using AIF360.

    This function measures the fairness inherent in the dataset's ground-truth labels,
    providing a baseline for bias before any model is applied. It is designed for
    data with categorical features represented numerically (e.g., 0 and 1).

    Args:
        df (pd.DataFrame): The dataset to evaluate. Must contain only numeric data.
        protected_attrs (list): A list of column names for protected attributes.
        target (str): The name of the target variable (outcome) column.

    Returns:
        dict: A dictionary containing the Theil Index and Consistency score.
    """
    try:
        # Ensure dataframe has no unexpected non-numeric types
        df_numeric = df.apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_numeric) != len(df):
            print("Warning: Rows with non-numeric data were dropped for fairness calculation.")

        # AIF360 requires a specific dataset format.
        # We explicitly define the favorable (positive) and unfavorable (negative) labels.
        aif_df = BinaryLabelDataset(df=df_numeric,
                                    label_names=[target],
                                    protected_attribute_names=protected_attrs,
                                    favorable_label=1,
                                    unfavorable_label=0)

        # Define privileged/unprivileged groups based on the protected attributes.
        # This assumes '1' is the privileged value for the protected attribute.
        privileged_groups = [{attr: 1 for attr in protected_attrs}]
        unprivileged_groups = [{attr: 0 for attr in protected_attrs}]

        #  Pass the dataset as both the true and classified dataset.
        metric = ClassificationMetric(aif_df,
                                      aif_df,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)

        # Theil Index: Measures inequality in outcomes. 0 is perfect equality.
        theil_index = metric.theil_index()

        # Consistency: Measures how many individuals have the same outcome as their
        # 'k' nearest neighbors. 1.0 is perfect consistency.
        consistency = metric.consistency(n_neighbors=5)

        print("\nIndividual Fairness Results (on input data):")
        print(f"  - Theil Index: {theil_index:.4f} (Measures outcome inequality; lower is better)")
        print(f"  - Consistency: {consistency:.4f} (Measures local outcome similarity; higher is better)")

        return {
            'theil_index': theil_index,
            'consistency': consistency
        }
    except Exception as e:
        print(f"An unexpected error occurred during individual fairness calculation: {str(e)}")
        # Provide traceback for easier debugging
        import traceback
        traceback.print_exc()
        return {
            'theil_index': np.nan,
            'consistency': np.nan,
            'error_individual_fairness': str(e)
        }
    