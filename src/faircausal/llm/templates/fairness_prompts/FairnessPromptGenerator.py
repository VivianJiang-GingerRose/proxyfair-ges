# poetry run python src\faircausal\llm\templates\fairness_prompts\FairnessPromptGenerator.py

import json
import os
import re
from typing import Dict, Any, List

# --- PATH DEFINITIONS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.dirname(SCRIPT_DIR)
LLM_DIR = os.path.dirname(TEMPLATES_DIR)
CONTEXT_DIR = os.path.join(TEMPLATES_DIR, "dataset_context")
OUTPUT_DIR = os.path.join(LLM_DIR, "final_prompts")
# --- END PATH DEFINITIONS ---

# Define the normative frameworks
FRAMEWORKS = {
    'statistical': {
        'template': 'FairnessPromptTemplate_Statistical.txt',
        'suffix': 'Statistical',
        'description': 'Statistical validity only, no fairness considerations'
    },
    'anti_classification': {
        'template': 'FairnessPromptTemplate_AntiClassification.txt',
        'suffix': 'AntiClassification',
        'description': 'Formal equality - block protected attributes directly, allow proxies'
    },
    'anti_subordination': {
        'template': 'FairnessPromptTemplate_AntiSubordination.txt',
        'suffix': 'AntiSubordination',
        'description': 'Substantive equality - block protected attributes AND proxies'
    },
    'meritocratic': {
        'template': 'FairnessPromptTemplate_Meritocratic.txt',
        'suffix': 'Meritocratic',
        'description': 'Equal opportunity - allow merit pathways, block ascriptive advantages'
    }
}


class FairnessPromptGenerator:
    """Generates fairness-focused causal discovery prompts across multiple normative frameworks."""

    def __init__(self, framework: str = 'anti_classification'):
        """Initializes the prompt generator for a specific normative framework."""
        if framework not in FRAMEWORKS:
            raise ValueError(
                f"Unknown framework '{framework}'. Must be one of: {list(FRAMEWORKS.keys())}"
            )

        self.framework = framework
        self.framework_config = FRAMEWORKS[framework]
        template_filename = self.framework_config['template']
        self.template_path = os.path.join(SCRIPT_DIR, template_filename)
        self.template = self._load_template()

    def _load_template(self) -> str:
        """Loads the prompt template from its absolute path."""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Template file not found: {self.template_path}\n"
                f"Expected template: {self.framework_config['template']}"
            )

    def load_dataset_config(self, config_path: str) -> Dict[str, Any]:
        """Loads dataset configuration from an absolute path."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data[0] if isinstance(data, list) else data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset config file not found at: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file '{config_path}': {e}")

    def _format_variables_section(self, config: Dict[str, Any]) -> str:
        """Formats the 'Variables' section using detailed info from the config."""
        columns = config.get('columns', [])
        details = config.get('column_details', {})
        outcome_var = config.get('outcome_variable', '')
        lines = []
        for col in columns:
            col_info = details.get(col, {})
            desc = col_info.get('description', f"Description for {col} not available.")
            dtype = col_info.get('datatype', 'Unknown')
            tag = "**TARGET OUTCOME** - " if col == outcome_var else ""
            lines.append(f"- {col}: {tag}{desc} (datatype: {dtype})")
        return "\n".join(lines)

    def generate_prompt(self, config_path: str) -> str:
        """Generates the final prompt by populating the template with dataset configuration."""
        config = self.load_dataset_config(config_path)

        placeholders = {
            'dataset_name': config.get('dataset_name', 'the dataset'),
            'expert_persona': config.get('expert_persona', 'a domain expert'),
            'dataset_context': config.get('dataset_context', 'No context provided.'),
            'country_or_region': config.get('country_or_region', 'the relevant region'),
            'data_collection_period': config.get('data_collection_period', 'the historical period'),
            'societal_considerations': config.get('societal_considerations', 'No specific societal considerations were provided.'),
            'prediction_context': config.get('prediction_context', 'future predictions'),
            'outcome_variable': config.get('outcome_variable', 'target_outcome'),
            'outcome_variable_description': config.get('outcome_variable_description', 'the outcome variable'),
            'root_nodes_list': ", ".join([f"`{node}`" for node in config.get('root_nodes_list', [])]),
            'root_nodes_description': config.get('root_nodes_description', 'These are considered root nodes based on domain knowledge.'),
            'example_root_node': config.get('example_root_node', 'an example root node'),
            'example_influential_variable': config.get('example_influential_variable', 'an influential variable'),
            'example_mediating_pathways': ", ".join([f"`{p}`" for p in config.get('example_mediating_pathways', [])]),
            'variable_list_formatted': self._format_variables_section(config),
        }

        return self.template.format(**placeholders)

    def generate_and_save_prompt(self, config_filename: str, output_filename: str = None):
        """Loads config, generates a prompt, and saves it to a file."""
        config_path = os.path.join(CONTEXT_DIR, config_filename)

        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(config_filename))[0]
            dataset_suffix = base_name.split('_')[-1] if '_' in base_name else base_name
            framework_suffix = self.framework_config['suffix']
            output_filename = f"FinalPrompt_{dataset_suffix}_{framework_suffix}.txt"

        output_path = os.path.join(OUTPUT_DIR, output_filename)
        prompt = self.generate_prompt(config_path)

        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Generated {self.framework} prompt saved to: {output_path}")


def generate_all_frameworks(config_filename: str):
    """Generates prompts for all 4 normative frameworks for a given dataset config."""
    print(f"\nGenerating prompts for all frameworks using config: {config_filename}")

    for framework_name in FRAMEWORKS.keys():
        try:
            generator = FairnessPromptGenerator(framework=framework_name)
            generator.generate_and_save_prompt(config_filename)
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"  ERROR in {framework_name}: {e}")


def main():
    """Main function to generate prompts for all frameworks across multiple datasets."""
    dataset_config_filenames: List[str] = [
        "DatasetContext_bank.json",
        # "DatasetContext_compas.json",
        # "DatasetContext_law.json",
        # "DatasetContext_dutch.json",
    ]

    # Verify templates exist
    print("Checking for template files...")
    for framework_name, framework_config in FRAMEWORKS.items():
        template_path = os.path.join(SCRIPT_DIR, framework_config['template'])
        if not os.path.exists(template_path):
            print(f"  WARNING: Template missing for {framework_name}: {framework_config['template']}")
        else:
            print(f"  \u2713 Found {framework_name} template")

    # Generate prompts
    for config_file in dataset_config_filenames:
        config_full_path = os.path.join(CONTEXT_DIR, config_file)
        if os.path.exists(config_full_path):
            generate_all_frameworks(config_file)
        else:
            print(f"  SKIPPING: Config file not found: {config_full_path}")


if __name__ == "__main__":
    main()
