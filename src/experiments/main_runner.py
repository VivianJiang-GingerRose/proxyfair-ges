# # Run all experiments on Dutch dataset only
# poetry run python src/experiments/main_runner.py --dataset dutch

# # Run baseline experiment on Dutch dataset only
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment baseline

# # Run domain knowledge experiment (no fairness constraint) on Dutch dataset only
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment domain_knowledge

# # Run specific experiment on Dutch dataset only
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment fairness

# # Run soft-fairness experiment with soft constraints in GES on Dutch dataset only
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment soft_fairness

# # Run with custom settings on Dutch dataset
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment fairness --max-dags 50

# # Run all experiments on each real dataset
# poetry run python src/experiments/main_runner.py --dataset law --experiment all
# poetry run python src/experiments/main_runner.py --dataset compas --experiment all
# poetry run python src/experiments/main_runner.py --dataset dutch --experiment all
# poetry run python src/experiments/main_runner.py --dataset bank --experiment all

"""
Fair Causal Discovery Runner with Simplified Configuration
Supports multiple datasets with LLM-generated constraints
"""
import pandas as pd
import numpy as np
import sys
import os
import datetime
import pickle
import argparse
import json
import importlib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
SUPPORTED_DATASETS = ("law", "compas", "dutch", "bank")

from src.faircausal.core.causal_data_utils import get_data_for_algorithm
from src.faircausal.core.prior_knowledge_processor import PriorKnowledgeProcessor
from src.faircausal.core.ges_runner_causal_learn import run_ges
from src.faircausal.core.counterfactual_fairness_runner import (
    analyze_all_dags_fairness,
    analyze_all_dags_improved_flow,
    select_fairest_dag,
)

# Add this class definition near the top of main_runner.py
import sys
from contextlib import contextmanager


def _format_lambda_for_run_name(lambda_value):
    """Encode lambda values in a compact run-folder-friendly form."""
    return f"{float(lambda_value):g}".replace('.', 'p').replace('-', 'm')

class TeeLogger:
    """
    A file-like object that writes to a log file and another stream (e.g., stdout).
    """
    def __init__(self, log_path, original_stream, mode='a', encoding='utf-8'):
        self.log_file = open(log_path, mode, encoding=encoding, errors='replace')
        self.original_stream = original_stream

    def write(self, message):
        try:
            self.original_stream.write(message)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fall back to ASCII-safe representation when terminal can't handle Unicode
            safe_msg = message.encode(self.original_stream.encoding or 'ascii', errors='replace').decode(self.original_stream.encoding or 'ascii')
            self.original_stream.write(safe_msg)
        self.log_file.write(message)

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

class DatasetConfig:
    """Dataset-specific configuration loader"""
    
    @classmethod
    def load_dataset_config(cls, dataset_name, config_path=None):
        """Load configuration for a specific dataset from JSON file"""
        if config_path is None:
            config_path = PROJECT_ROOT / "src" / "experiments" / "configs" / f"{dataset_name}_config.json"
        else:
            config_path = Path(config_path)

        # Convert to absolute path and resolve any path issues
        config_path = config_path.resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Loaded configuration for {dataset_name} from {config_path}")
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    @classmethod
    def get_available_datasets(cls):
        """Get list of available dataset configurations"""
        configs_dir = PROJECT_ROOT / "src" / "experiments" / "configs"
        if not configs_dir.exists():
            return []
        
        available = []
        for config_file in configs_dir.glob("*_config.json"):
            dataset_name = config_file.stem.replace("_config", "")
            if dataset_name in SUPPORTED_DATASETS:
                available.append(dataset_name)
        
        return sorted(available)


class ExperimentConfig:
    """Simplified configuration class for experiments"""
    
    def __init__(self, dataset_name='law', experiment_type='baseline', custom_config=None):
        # Load dataset-specific configuration
        self.dataset_config = DatasetConfig.load_dataset_config(dataset_name)
        self.dataset_name = dataset_name
        self.experiment_type = experiment_type
        
        # Set constraint usage based on experiment type
        self._set_experiment_constraints(experiment_type)
        
        # Override with custom configuration if provided
        if custom_config:
            self._update_from_dict(custom_config)
    
    def _set_experiment_constraints(self, experiment_type):
        """Set constraint usage based on experiment type"""
        constraint_configs = {
            'baseline': {
                'use_blacklist': False,
                'use_whitelist': False,
                'use_fairness_constraints': False,
            },
            'domain_knowledge': {
                'use_blacklist': True,
                'use_whitelist': True, 
                'use_fairness_constraints': False,
            },
            'fairness': {
                'use_blacklist': True,
                'use_whitelist': True,
                'use_fairness_constraints': True,
            },
            'soft_fairness': {
                'use_blacklist': True,
                'use_whitelist': True,
                'use_fairness_constraints': True,
            },
            'soft_fairness_only': {
                'use_blacklist': False,
                'use_whitelist': False,
                'use_fairness_constraints': False,
            },
        }
        
        if experiment_type in constraint_configs:
            for key, value in constraint_configs[experiment_type].items():
                setattr(self, key, value)
        else:
            # Default to fairness experiment
            for key, value in constraint_configs['fairness'].items():
                setattr(self, key, value)
    
    def _update_from_dict(self, config_dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if key == 'fairness_lambda_override':
                try:
                    lambda_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid fairness lambda override: {value}"
                    ) from exc

                soft_settings = self.dataset_config.setdefault('soft_fairness_settings', {})
                soft_settings['lambda'] = lambda_value
                # Keep a copy on the config object for run naming and auditing.
                setattr(self, key, lambda_value)
                continue

            setattr(self, key, value)


class FairCausalDiscoveryRunner:
    """Main runner class for fair causal discovery experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_config = config.dataset_config
        self.log_file = None
        self.run_dir = None
        self.data_loader = None

    def _resolve_experiment_name(self, experiment_name):
        """Append lambda suffix for soft-fairness runs when CLI override is used."""
        if self.config.experiment_type not in ('soft_fairness', 'soft_fairness_only'):
            return experiment_name

        lambda_override = getattr(self.config, 'fairness_lambda_override', None)
        if lambda_override is None:
            return experiment_name

        lambda_suffix = _format_lambda_for_run_name(lambda_override)
        expected_suffix = f"_lambda_{lambda_suffix}"
        if experiment_name.endswith(expected_suffix):
            return experiment_name

        return f"{experiment_name}{expected_suffix}"
        
    def _load_data_loader(self):
        """Dynamically load the appropriate data loader module"""
        try:
            module_name = self.dataset_config['data_settings']['data_loader_module']
            module = importlib.import_module(module_name)
            self.data_loader = module
            print(f"Loaded data loader: {module_name}")
        except ImportError as e:
            raise ImportError(f"Could not import data loader module: {e}")

    @staticmethod
    def _resolve_input_path(path_value):
        """Resolve repository-relative input paths without depending on the caller's cwd."""
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def _load_experiment_data(self, preprocessing_log_path=None):
        """Load canonical processed data, or explicitly preprocess a supplied raw file."""
        reprocess_data = bool(getattr(self.config, "reprocess_data", False))

        if reprocess_data:
            raw_data_path = getattr(self.config, "raw_data_path", None)
            if not raw_data_path:
                raise ValueError(
                    "Raw preprocessing requires --raw-data-path PATH together with "
                    "--reprocess-data."
                )

            resolved_raw_path = self._resolve_input_path(raw_data_path)
            if not resolved_raw_path.is_file():
                raise FileNotFoundError(f"Raw dataset not found: {resolved_raw_path}")

            self._load_data_loader()
            data_dict = self.data_loader.load_and_preprocess_data(
                str(resolved_raw_path),
                export_path=None,
                log_file_path=preprocessing_log_path,
            )
            if "preprocessed" not in data_dict:
                raise KeyError("Data loader did not return a 'preprocessed' dataframe.")
            return data_dict["preprocessed"]

        data_settings = self.dataset_config.get("data_settings", {})
        analysis_data_path = data_settings.get("analysis_data_path")
        if not analysis_data_path:
            raise KeyError(
                f"Dataset '{self.config.dataset_name}' does not define analysis_data_path."
            )

        resolved_analysis_path = self._resolve_input_path(analysis_data_path)
        if not resolved_analysis_path.is_file():
            raise FileNotFoundError(
                f"Canonical processed dataset not found: {resolved_analysis_path}"
            )

        dataframe = pd.read_csv(resolved_analysis_path)
        if preprocessing_log_path:
            log_path = Path(preprocessing_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "Loaded canonical processed dataset without reprocessing.\n"
                f"Path: {resolved_analysis_path}\n"
                f"Shape: {dataframe.shape}\n",
                encoding="utf-8",
            )
        return dataframe
        
    def setup_logging(self, experiment_name="baseline"):
        """Set up logging directory and file with a unique run folder"""
        base_log_dir = Path("results/experiments") / self.config.dataset_name
        base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a unique run folder with timestamp and experiment name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment_name}_{timestamp}"
        self.run_dir = base_log_dir / run_name
        self.run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.run_dir / "scm_models").mkdir(exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        
        # Create log file
        self.log_file = self.run_dir / "experiment_log.txt"
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== FAIR CAUSAL DISCOVERY EXPERIMENT LOG ===\n")
            f.write(f"Dataset: {self.config.dataset_name}\n")
            f.write(f"Experiment: {experiment_name}\n")
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Configuration: {vars(self.config)}\n\n")
        
        return self.log_file, self.run_dir
    
    def load_and_split_data(self):
        """
        Load and preprocess data, then split into structure discovery and SCM/evaluation sets.
        
        This function:
        1. Loads and preprocesses the dataset (all existing functionality)
        2. Splits into Structure Discovery vs SCM/Evaluation sets 
        3. Prepares both datasets for their respective purposes
        4. Returns split data to avoid winner's curse
        """
        print("Loading and splitting data to avoid winner's curse...")
        
        # ===== STEP 1: load data in =====
        print(f"Loading {self.config.dataset_name} dataset...")

        preprocessing_log_path = str(self.run_dir / "data_preprocessing_log.txt")
        df_full = self._load_experiment_data(preprocessing_log_path)
        
        # ===== STEP 2: Split data to avoid winner's curse =====
        
        # Get split configuration
        counterfactual_settings = self.dataset_config.get('counterfactual_settings', {})
        structure_discovery_ratio = counterfactual_settings.get('structure_discovery_ratio', 0.5)
        enable_sample_splitting = counterfactual_settings.get('enable_sample_splitting', True)
        random_state = getattr(self.config, 'random_state', 42)
        
        # Get target for stratification
        fairness_settings = self.dataset_config['fairness_settings']
        target = fairness_settings['target']
        
        # Log STEP 2 to preprocessing log
        self._append_to_preprocessing_log(preprocessing_log_path, "\n" + "="*60)
        self._append_to_preprocessing_log(preprocessing_log_path, "STEP 2: Split data to avoid winner's curse")
        self._append_to_preprocessing_log(preprocessing_log_path, "="*60)
        
        if enable_sample_splitting:
            self._append_to_preprocessing_log(preprocessing_log_path, "SAMPLE SPLITTING ENABLED - Splitting data to avoid winner's curse:")
            self._append_to_preprocessing_log(preprocessing_log_path, f"  {structure_discovery_ratio:.1%} for structure discovery (GES)")
            self._append_to_preprocessing_log(preprocessing_log_path, f"  {1-structure_discovery_ratio:.1%} for SCM fitting & fairness evaluation")
            
            print(f"SAMPLE SPLITTING ENABLED - Splitting data to avoid winner's curse:")
            print(f"  {structure_discovery_ratio:.1%} for structure discovery (GES)")
            print(f"  {1-structure_discovery_ratio:.1%} for SCM fitting & fairness evaluation")
            
            # CRITICAL SPLIT: Separate data for structure discovery vs SCM/evaluation
            df_structure_discovery, df_scm_evaluation = train_test_split(
                df_full,
                test_size=(1 - structure_discovery_ratio),  # Remaining for SCM & evaluation
                random_state=random_state,
                stratify=df_full[target] if target in df_full.columns else None
            )
            
            split_info = f"Structure discovery set: {len(df_structure_discovery)} samples"
            print(split_info)
            self._append_to_preprocessing_log(preprocessing_log_path, split_info)
            
            split_info = f"SCM & evaluation set: {len(df_scm_evaluation)} samples"
            print(split_info)
            self._append_to_preprocessing_log(preprocessing_log_path, split_info)
            
            # Log target distributions to verify stratification worked
            if target in df_full.columns:
                struct_dist = f"Structure discovery {target} distribution: {df_structure_discovery[target].value_counts().to_dict()}"
                scm_dist = f"SCM & evaluation {target} distribution: {df_scm_evaluation[target].value_counts().to_dict()}"
                
                print(struct_dist)
                print(scm_dist)
                self._append_to_preprocessing_log(preprocessing_log_path, struct_dist)
                self._append_to_preprocessing_log(preprocessing_log_path, scm_dist)
        else:
            split_info = "SAMPLE SPLITTING DISABLED - Using traditional approach (may suffer from winner's curse)"
            print(split_info)
            self._append_to_preprocessing_log(preprocessing_log_path, split_info)
            
            # Use the same data for both (original behavior)
            df_structure_discovery = df_full.copy()
            df_scm_evaluation = df_full.copy()
        
        # ===== STEP 3: Dataset-specific analysis (existing functionality) =====
        # Log STEP 3 to preprocessing log
        self._append_to_preprocessing_log(preprocessing_log_path, "\n" + "="*60)
        self._append_to_preprocessing_log(preprocessing_log_path, "STEP 3: Dataset-specific analysis")
        self._append_to_preprocessing_log(preprocessing_log_path, "="*60)
        
        # Perform this on the structure discovery set since that's what GES will see
        self._perform_dataset_analysis_with_logging(df_structure_discovery, preprocessing_log_path)
        
        # ===== STEP 4: Prepare data for GES (existing functionality) =====
        # Log STEP 4 to preprocessing log
        self._append_to_preprocessing_log(preprocessing_log_path, "\n" + "="*60)
        self._append_to_preprocessing_log(preprocessing_log_path, "STEP 4: Prepare data for GES (existing functionality)")
        self._append_to_preprocessing_log(preprocessing_log_path, "="*60)
        
        # Prepare structure discovery data for GES algorithm    
        X, node_names = get_data_for_algorithm(df_structure_discovery, 'ges')
        
        node_info = f"Node names: {node_names}"
        print(node_info)
        self._append_to_preprocessing_log(preprocessing_log_path, node_info)
        
        # Log final data shapes
        final_shapes = f"Final data prepared for GES: X.shape = {X.shape}"
        self._append_to_preprocessing_log(preprocessing_log_path, final_shapes)
        
        # Log completion
        self._append_to_preprocessing_log(preprocessing_log_path, "\n" + "="*60)
        self._append_to_preprocessing_log(preprocessing_log_path, "DATA LOADING AND PREPROCESSING PIPELINE COMPLETE")
        self._append_to_preprocessing_log(preprocessing_log_path, "="*60)
        
        return df_structure_discovery, df_scm_evaluation, X, node_names


    def _append_to_preprocessing_log(self, log_file_path, message):
        """Helper function to append messages to the preprocessing log file"""
        try:
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to preprocessing log: {e}")


    def _perform_dataset_analysis_with_logging(self, df_for_ges, log_file_path):
        """Perform dataset-specific exploratory analysis with logging"""
        protected_attrs = self.dataset_config['fairness_settings']['protected_attrs']
        target = self.dataset_config['fairness_settings']['target']
        
        analysis_info = f"Performing dataset-specific analysis on structure discovery data ({len(df_for_ges)} samples)"
        print(analysis_info)
        self._append_to_preprocessing_log(log_file_path, analysis_info)
        
        # Check protected attribute vs target distribution
        if protected_attrs and target and target in df_for_ges.columns:
            for protected_attr in protected_attrs:
                if protected_attr in df_for_ges.columns:
                    cross_tab = pd.crosstab(
                        df_for_ges[protected_attr], 
                        df_for_ges[target], 
                        normalize='index'
                    )
                    
                    cross_tab_info = f"{protected_attr.title()} vs. {target.title()} distribution:\n{cross_tab}"
                    print(cross_tab_info)
                    self._append_to_preprocessing_log(log_file_path, cross_tab_info)
        
        # Log basic dataset info
        dataset_info = f"Dataset shape for structure discovery: {df_for_ges.shape}"
        dataset_cols = f"Columns available: {list(df_for_ges.columns)}"
        
        print(dataset_info)
        print(dataset_cols)
        self._append_to_preprocessing_log(log_file_path, dataset_info)
        self._append_to_preprocessing_log(log_file_path, dataset_cols)
        
        # Log data types
        dtype_info = "Data types:"
        self._append_to_preprocessing_log(log_file_path, dtype_info)
        for col in df_for_ges.columns:
            col_dtype = f"  {col}: {df_for_ges[col].dtype}"
            self._append_to_preprocessing_log(log_file_path, col_dtype)

    def setup_constraints(self, node_names):
        """Enhanced constraint setup with detailed validation"""
        print("Setting up constraints with enhanced validation...")
        
        # Get constraints from config
        constraint_settings = self.dataset_config['constraint_settings']
        
        # Define whitelisted edges
        whitelisted_edges = []
        if self.config.use_whitelist:
            whitelisted_edges = constraint_settings.get('whitelist_edges', [])
        
        # Define blacklisted edges
        blacklisted_edges = []
        if self.config.use_blacklist:
            # Add demographic constraints (Type A)
            demographic_constraints = constraint_settings['blacklist_edges'].get('demographic_constraints', [])
            blacklisted_edges.extend(demographic_constraints)
        
        # Add fairness constraints (Type B)
        if self.config.use_fairness_constraints:
            fairness_constraints = constraint_settings['blacklist_edges'].get('fairness_constraints', [])
            blacklisted_edges.extend(fairness_constraints)
        
        # Log original constraints
        print(f"Original whitelist constraints: {len(whitelisted_edges)}")
        print(f"Original blacklist constraints: {len(blacklisted_edges)}")
        
        # Filter constraints to only include nodes that exist in the data
        original_whitelist = whitelisted_edges.copy()
        original_blacklist = blacklisted_edges.copy()
        
        whitelisted_edges = [(a, b) for a, b in whitelisted_edges if a in node_names and b in node_names]
        blacklisted_edges = [(a, b) for a, b in blacklisted_edges if a in node_names and b in node_names]
        
        # Log filtering results
        dropped_whitelist = len(original_whitelist) - len(whitelisted_edges)
        dropped_blacklist = len(original_blacklist) - len(blacklisted_edges)
        
        if dropped_whitelist > 0:
            print(f"Warning: Dropped {dropped_whitelist} whitelist constraints (nodes not in data)")
        if dropped_blacklist > 0:
            print(f"Warning: Dropped {dropped_blacklist} blacklist constraints (nodes not in data)")
        
        print(f"Applied {len(whitelisted_edges)} whitelist constraints")
        print(f"Applied {len(blacklisted_edges)} blacklist constraints")
        
        # Log actual constraints being applied
        if whitelisted_edges:
            print("Whitelist edges to be applied:")
            for src, dst in whitelisted_edges:
                print(f"  {src} -> {dst}")
        
        if blacklisted_edges:
            print("Blacklist edges to be applied:")
            for src, dst in blacklisted_edges:
                print(f"  {src} -> {dst} (FORBIDDEN)")
        
        # Process constraints
        processor = PriorKnowledgeProcessor(node_names)
        constraint_matrices = processor.process_all_constraints(
            temporal_order=None,
            blacklist=blacklisted_edges,
            whitelist=whitelisted_edges
        )
        
        # ENHANCED: Validate constraint matrices
        if constraint_matrices.get('combined_blacklist') is not None:
            blacklist_matrix = constraint_matrices['combined_blacklist']
            print(f"\nBlacklist matrix shape: {blacklist_matrix.shape}")
            print(f"Number of forbidden edges in matrix: {np.sum(blacklist_matrix)}")
            
            # Debug the blacklist matrix - write to log file
            with open(self.log_file, 'a') as f:
                from src.faircausal.core.ges_runner_causal_learn import debug_blacklist_matrix
                debug_blacklist_matrix(blacklist_matrix, node_names, file=f)
        
        return constraint_matrices
    
    def run_ges_algorithm(self, df_for_ges, node_names, constraint_matrices):
        """Enhanced GES algorithm runner with comprehensive validation"""
        print("Running enhanced GES algorithm with validation...")
        
        # Determine which constraints to use
        blacklist_matrix = None
        whitelist_matrix = None
        
        if self.config.use_blacklist or self.config.use_fairness_constraints:
            blacklist_matrix = constraint_matrices['combined_blacklist']
            print(f"Using blacklist matrix with {np.sum(blacklist_matrix)} forbidden edges")
        
        if self.config.use_whitelist:
            whitelist_matrix = constraint_matrices['whitelist']
            print(f"Using whitelist matrix with {np.sum(whitelist_matrix)} required edges")
        
        # Resolve fairness regularization parameters for soft_fairness experiment
        fairness_lambda = None
        fairness_tau_c_for_ges = 5.0
        protected_attr_indices_for_ges = None
        outcome_index_for_ges = None
        path_effect_method_for_ges = 'distance'

        if self.config.experiment_type in ('soft_fairness', 'soft_fairness_only'):
            soft_settings = self.dataset_config.get('soft_fairness_settings', {})
            fairness_settings = self.dataset_config['fairness_settings']

            fairness_lambda = soft_settings.get('lambda', 1.0)
            fairness_tau_c_for_ges = soft_settings.get('tau_c', 5.0)
            path_effect_method_for_ges = soft_settings.get('path_effect_method', 'distance')

            protected_attrs = fairness_settings['protected_attrs']
            target = fairness_settings['target']

            protected_attr_indices_for_ges = [
                node_names.index(a) for a in protected_attrs if a in node_names
            ]
            outcome_index_for_ges = node_names.index(target) if target in node_names else None

            print(f"Soft-fairness Stage 1: lambda={fairness_lambda}, tau_c={fairness_tau_c_for_ges}, "
                  f"protected={protected_attrs}, target={target}, "
                  f"path_effect={path_effect_method_for_ges}")

        result = run_ges(
            df_for_ges.values,
            node_names=node_names,
            blacklist_matrix=blacklist_matrix,
            whitelist_matrix=whitelist_matrix,
            debug=True,
            log_file=self.log_file,
            run_dir=self.run_dir,
            experiment_type=self.config.experiment_type,
            fairness_lambda=fairness_lambda,
            fairness_tau_c=fairness_tau_c_for_ges,
            protected_attr_indices=protected_attr_indices_for_ges,
            outcome_index=outcome_index_for_ges,
            path_effect_method=path_effect_method_for_ges,
        )
        
        # CRITICAL: Check if blacklist was violated
        if not result.get('blacklist_compliant', True):
            violations = result.get('blacklist_violations', [])
            print(f"\n[ERROR] CRITICAL ERROR: GES produced {len(violations)} blacklist violations!")
            print("This indicates a serious problem with the constraint handling.")
            
            # Log violations to main log
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write("BLACKLIST VIOLATION ERROR\n")
                f.write(f"{'='*60}\n")
                f.write(f"Number of violations: {len(violations)}\n")
                for src, dst, edge_type in violations:
                    f.write(f"  VIOLATION: {src} -> {dst} ({edge_type})\n")
            
            # You might want to raise an exception or implement fallback logic here
            raise ValueError(f"GES algorithm violated {len(violations)} blacklist constraints!")
        else:
            print("[OK] GES result successfully complies with all blacklist constraints")

        return result
    
    def validate_final_dag_for_fairness(self, cpdag, node_names, blacklist_matrix):
        """
        Validate that the DAG being passed to fairness analysis is correct
        """
        print("\n" + "="*60)
        print("VALIDATING DAG BEFORE FAIRNESS ANALYSIS")
        print("="*60)
        
        # Import validation function
        from src.faircausal.core.ges_runner_causal_learn import validate_blacklist_compliance, print_graph_edges
        
        # Print detailed edge information
        with open(self.log_file, 'a') as f:
            print_graph_edges(cpdag, node_names, file=f)
        
        # Also print to console
        print_graph_edges(cpdag, node_names)
        
        # Validate blacklist compliance
        if blacklist_matrix is not None:
            is_valid, violations = validate_blacklist_compliance(
                cpdag, 
                blacklist_matrix, 
                node_names
            )
            
            if not is_valid:
                print(f"\n[ERROR] ERROR: DAG for fairness analysis has {len(violations)} blacklist violations!")
                
                # Log to file
                with open(self.log_file, 'a') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write("DAG VALIDATION ERROR BEFORE FAIRNESS ANALYSIS\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Number of violations: {len(violations)}\n")
                    for src, dst, edge_type in violations:
                        f.write(f"  VIOLATION: {src} -> {dst} ({edge_type})\n")
                
                raise ValueError(f"DAG for fairness analysis violates {len(violations)} constraints!")
            else:
                print("[OK] DAG is valid for fairness analysis")
        
        print("="*60)

    def run_fairness_analysis(self, df_scm_evaluation, cpdag, node_names):
        """Run counterfactual fairness analysis"""
        print("Setting up fairness analysis parameters...")
        
        # Get fairness settings from config
        fairness_settings = self.dataset_config['fairness_settings']
        variable_settings = self.dataset_config['variable_settings']
        experiment_settings = self.dataset_config['experiment_settings']

        # Get counterfactual settings with defaults
        counterfactual_settings = self.dataset_config.get('counterfactual_settings', {})
        sample_percentage = getattr(self.config, 'sample_percentage', 
                                   counterfactual_settings.get('sample_percentage', 1.0))
        min_samples = getattr(self.config, 'min_samples', 
                             counterfactual_settings.get('min_samples', 10))
        random_state = getattr(self.config, 'random_state', 
                              counterfactual_settings.get('random_state', 42))
        cutoff_threshold = getattr(self.config, 'cutoff_threshold', 
                                  counterfactual_settings.get('cutoff_threshold', 0.5))
        calculate_pse = getattr(self.config, 'calculate_pse', 
                       counterfactual_settings.get('calculate_pse', False))
        pse_sample_size = getattr(self.config, 'pse_sample_size', 
                     counterfactual_settings.get('pse_sample_size', None))
        max_paths = getattr(self.config, 'max_paths', 
                   counterfactual_settings.get('max_paths', None))
        max_dags = getattr(self.config, 'max_dags', 
                  experiment_settings.get('max_dags', 100))
        
        protected_attrs = fairness_settings['protected_attrs']
        target = fairness_settings['target']
        disadvantage_group = fairness_settings['disadvantage_group']
        var_types = variable_settings['var_types']

        generate_synthetic = experiment_settings.get('generate_synthetic', True)
        generate_synthetic_double = experiment_settings.get('synthetic_variants', {}).get('double', False)
        generate_synthetic_balanced = experiment_settings.get('synthetic_variants', {}).get('balanced', False)
        generate_synthetic_protected = experiment_settings.get('synthetic_variants', {}).get('protected', False)

        # Runtime guardrail: when PSE is enabled, evaluate only the standard synthetic variant
        # to control multiplicative runtime from per-variant PSE computations.
        if calculate_pse and (generate_synthetic_double or generate_synthetic_balanced or generate_synthetic_protected):
            print(
                "PSE enabled: forcing synthetic variants to standard only "
                "(disabling double/balanced/protected variants for runtime control)."
            )
            generate_synthetic_double = False
            generate_synthetic_balanced = False
            generate_synthetic_protected = False
        
        # Setup analysis parameters
        X_cols = [col for col in node_names if col != target]

        print(
            f"Fairness analysis settings: max_dags={max_dags}, sample_percentage={sample_percentage}, "
            f"min_samples={min_samples}, calculate_pse={calculate_pse}, "
            f"pse_sample_size={pse_sample_size}, max_paths={max_paths}"
        )
        
        print("Analyzing all possible DAGs for counterfactual fairness...")
        cf_results_df, ml_results_df = analyze_all_dags_improved_flow(
            df=df_scm_evaluation,
            cpdag=cpdag,
            node_names=node_names,
            protected_attrs=protected_attrs,
            target=target,
            X_cols=X_cols,
            disadvantage_group=disadvantage_group,
            estimator=RandomForestClassifier,
            var_types=var_types,
            max_dags=max_dags,
            log_file=self.log_file,
            save_model=experiment_settings.get('save_models', False),
            model_dir=self.run_dir / "scm_models",
            generate_synthetic=generate_synthetic,
            generate_synthetic_double=generate_synthetic_double,
            generate_synthetic_balanced=generate_synthetic_balanced,
            generate_synthetic_protected=generate_synthetic_protected,
            sample_percentage=sample_percentage,
            min_samples=min_samples,
            random_state=random_state,
            cutoff_threshold=cutoff_threshold,
            calculate_pse=calculate_pse,
            pse_sample_size=pse_sample_size,
            max_paths=max_paths
        )
        
        return cf_results_df, ml_results_df

    def save_results(self, cf_results_df, ml_results_df, experiment_name):
        """Save experiment results"""
        results_dir = self.run_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save dataframes
        cf_results_df.to_csv(results_dir / f"{experiment_name}_cf_results.csv", index=False)
        ml_results_df.to_csv(results_dir / f"{experiment_name}_ml_results.csv", index=False)
        
        # Save configuration
        config_dict = {
            'dataset_name': self.config.dataset_name,
            'experiment_type': self.config.experiment_type,
            'dataset_config': self.dataset_config,
            'experiment_config': {k: v for k, v in vars(self.config).items() 
                                 if k not in ['dataset_config', 'dataset_name']}
        }
        with open(results_dir / f"{experiment_name}_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        print(f"Results saved to: {results_dir}")

    def _run_stage2_selection(self, df_for_stage2, cpdag, node_names):
        """Stage 2 of the Two-Stage Pipeline: select the fairest DAG from the MEC."""
        from src.faircausal.causal_learn.causallearn.graph.GeneralGraph import GeneralGraph
        from src.faircausal.causal_learn.causallearn.graph.GraphNode import GraphNode
        from src.faircausal.causal_learn.causallearn.graph.Endpoint import Endpoint

        soft_settings = self.dataset_config.get('soft_fairness_settings', {})
        fairness_settings = self.dataset_config['fairness_settings']

        fairness_lambda = soft_settings.get('lambda', 1.0)
        beta = soft_settings.get('beta', 0.0)
        max_stage2_dags = soft_settings.get('max_stage2_dags', 50)

        protected_attrs = fairness_settings['protected_attrs']
        target = fairness_settings['target']

        protected_attr_indices = [
            node_names.index(a) for a in protected_attrs if a in node_names
        ]
        outcome_index = node_names.index(target) if target in node_names else None

        if outcome_index is None:
            print(f"Warning: Target '{target}' not found in node_names. Skipping Stage 2.")
            return cpdag

        stage2_result = select_fairest_dag(
            cpdag=cpdag,
            data=df_for_stage2.values,
            node_names=node_names,
            protected_attr_indices=protected_attr_indices,
            outcome_index=outcome_index,
            fairness_lambda=fairness_lambda,
            beta=beta,
            max_stage2_dags=max_stage2_dags,
        )

        n_evaluated = stage2_result.get('n_dags_evaluated', 0)
        if n_evaluated <= 1:
            print(f"Stage 2: Only {n_evaluated} DAG(s) in MEC — returning cpdag unchanged.")
            return cpdag

        dag_edges = stage2_result.get('dag_edges', [])
        if not dag_edges:
            print("Stage 2: No edges in best DAG result — returning cpdag unchanged.")
            return cpdag

        # Rebuild a GeneralGraph from the best-selected DAG edge list
        nodes = [GraphNode(name) for name in node_names]
        best_dag_graph = GeneralGraph(nodes)
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        TAIL_val = Endpoint.TAIL.value
        ARROW_val = Endpoint.ARROW.value

        for src, tgt in dag_edges:
            si, ti = name_to_idx[src], name_to_idx[tgt]
            best_dag_graph.graph[si, ti] = TAIL_val
            best_dag_graph.graph[ti, si] = ARROW_val

        print(
            f"Stage 2 complete: FS={stage2_result['score']:.4f}, "
            f"BIC={stage2_result['bic_score']:.4f}, "
            f"penalty={stage2_result['fairness_penalty']:.4f}"
        )
        return best_dag_graph

    def run_experiment(self, experiment_name=None):
        """Run a complete experiment"""
        if experiment_name is None:
            experiment_name = self.config.experiment_type
        experiment_name = self._resolve_experiment_name(experiment_name)
        
        # Setup logging directories and get the log file path
        # Note: We are not writing the header here yet, the TeeLogger will do it.
        self.setup_logging(experiment_name)
        
        # Store the original stdout
        original_stdout = sys.stdout
        tee_logger = None
        
        try:
            # Initialize our TeeLogger to hijack stdout
            tee_logger = TeeLogger(self.log_file, original_stdout)
            sys.stdout = tee_logger

            # Now, any print() statement will go to both the console and the file.
            # We can re-write the header here.
            print(f"\n{'='*60}")
            print(f"Running Experiment: {experiment_name}")
            print(f"Dataset: {self.config.dataset_name}")
            print(f"Full logs will be saved to: {self.log_file}")
            print(f"{'='*60}\n")

            # STEP 1: Load and split data (UPDATED - replaces load_and_prepare_data)
            df_structure_discovery, df_scm_evaluation, X, node_names = self.load_and_split_data()
            
            # STEP 2: Setup constraints (existing functionality)
            # Use the structure discovery data for constraint setup
            constraint_matrices = self.setup_constraints(node_names)
            
             # STEP 3: Run GES algorithm on structure discovery data ONLY 
            result = self.run_ges_algorithm(df_structure_discovery, node_names, constraint_matrices)
            cpdag = result['G']
            
            # STEP 3a: Stage 2 — select fairest DAG from MEC (soft fairness variants)
            if self.config.experiment_type in ('soft_fairness', 'soft_fairness_only'):
                print("\nRunning Stage 2: Selecting fairest DAG from MEC...")
                cpdag = self._run_stage2_selection(df_structure_discovery, cpdag, node_names)
            
            # STEP 4: Validate DAG before fairness analysis
            blacklist_matrix = constraint_matrices.get('combined_blacklist')
            self.validate_final_dag_for_fairness(cpdag, node_names, blacklist_matrix)

            # STEP 5: Run fairness analysis on SCM/evaluation data ONLY
            cf_results_df, ml_results_df = self.run_fairness_analysis(df_scm_evaluation, cpdag, node_names)
            
            # Save results
            self.save_results(cf_results_df, ml_results_df, experiment_name)
            
            print(f"Experiment '{experiment_name}' on '{self.config.dataset_name}' completed successfully!")
            return {
                'cf_results': cf_results_df,
                'ml_results': ml_results_df,
                'structure_discovery_samples': len(df_structure_discovery),
                'scm_evaluation_samples': len(df_scm_evaluation),
                'winner_curse_avoided': True
            }
            
        except Exception as e:
            print(f"Error in experiment '{experiment_name}': {str(e)}")
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"\nERROR: {str(e)}\n")
            raise


def run_multiple_experiments(dataset_name='law', custom_config=None):
    """Run multiple predefined experiments for a specific dataset"""
    
    experiments = ['baseline', 'domain_knowledge', 'fairness', 'soft_fairness_only', 'soft_fairness']
    results = {}
    custom_config = custom_config or {}
    
    for experiment_type in experiments:
        print(f"\nStarting {experiment_type} experiment for {dataset_name} dataset...")
        
        try:
            config = ExperimentConfig(dataset_name, experiment_type, custom_config)
            runner = FairCausalDiscoveryRunner(config)
            results[experiment_type] = runner.run_experiment()
        except Exception as e:
            print(f"Failed to run {experiment_type}: {str(e)}")
            results[experiment_type] = None
    
    return results


def build_argument_parser():
    """Build the public command-line interface for real-world experiments."""
    parser = argparse.ArgumentParser(description='Fair Causal Discovery Experiments')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=SUPPORTED_DATASETS,
        default='law',
        help='Which supported dataset to use',
    )
    parser.add_argument('--experiment', type=str,
                       choices=['baseline', 'domain_knowledge', 'fairness', 'soft_fairness', 'soft_fairness_only', 'all'],
                       default='all', help='Which experiment to run')
    parser.add_argument('--config', type=str, help='Path to JSON configuration file')
    parser.add_argument('--max-dags', type=int, default=None, help='Maximum number of DAGs to analyze (overrides dataset config)')
    parser.add_argument('--lambda-fairness', type=float, default=None, help='Override soft fairness lambda for soft_fairness/soft_fairness_only runs')
    parser.add_argument('--cutoff-threshold', type=float, default=None, help='Cutoff threshold for XGBoost predictions (overrides dataset config)')
    parser.add_argument('--calculate-pse', action='store_true', help='Enable Path-Specific Effect (PSE) calculation')
    parser.add_argument('--pse-sample-size', type=int, default=None, help='Optional subsample size for PSE evaluation')
    parser.add_argument('--max-paths', type=int, default=None, help='Maximum number of causal paths to evaluate per protected attribute')
    parser.add_argument(
        '--reprocess-data',
        action='store_true',
        help='Preprocess a caller-supplied raw dataset in memory instead of using the committed processed data',
    )
    parser.add_argument(
        '--raw-data-path',
        type=str,
        default=None,
        help='Path to a raw dataset; valid only together with --reprocess-data',
    )
    return parser


def validate_cli_args(parser, args):
    """Validate cross-argument constraints that argparse cannot express directly."""
    if args.max_dags is not None and args.max_dags <= 0:
        parser.error('--max-dags must be a positive integer when provided')
    if args.lambda_fairness is not None and args.lambda_fairness < 0:
        parser.error('--lambda-fairness must be non-negative when provided')
    if args.pse_sample_size is not None and args.pse_sample_size <= 0:
        parser.error('--pse-sample-size must be a positive integer when provided')
    if args.max_paths is not None and args.max_paths <= 0:
        parser.error('--max-paths must be a positive integer when provided')
    if args.reprocess_data and not args.raw_data_path:
        parser.error('--reprocess-data requires --raw-data-path PATH')
    if args.raw_data_path and not args.reprocess_data:
        parser.error('--raw-data-path is valid only together with --reprocess-data')


def main():
    """Main function with command line interface"""
    parser = build_argument_parser()
    args = parser.parse_args()
    validate_cli_args(parser, args)
    
    # Load custom configuration if provided
    custom_config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
    
    # Add command line arguments to custom config
    if args.max_dags is not None:
        custom_config['max_dags'] = args.max_dags
    if args.lambda_fairness is not None:
        custom_config['fairness_lambda_override'] = args.lambda_fairness
    if args.cutoff_threshold is not None:
        custom_config['cutoff_threshold'] = args.cutoff_threshold
    if args.calculate_pse:
        custom_config['calculate_pse'] = True
    if args.pse_sample_size is not None:
        custom_config['pse_sample_size'] = args.pse_sample_size
    if args.max_paths is not None:
        custom_config['max_paths'] = args.max_paths
    if args.reprocess_data:
        custom_config['reprocess_data'] = True
        custom_config['raw_data_path'] = args.raw_data_path
    
    if args.experiment == 'all':
        results = run_multiple_experiments(args.dataset, custom_config)
    else:
        config = ExperimentConfig(args.dataset, args.experiment, custom_config)
        runner = FairCausalDiscoveryRunner(config)
        results = runner.run_experiment()
    
    print("\n" + "="*60)
    print(f"All experiments for {args.dataset} dataset completed!")
    print("="*60)


if __name__ == "__main__":
    main()
