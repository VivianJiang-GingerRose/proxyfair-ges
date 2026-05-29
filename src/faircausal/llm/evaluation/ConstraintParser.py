"""
Parse LLM-generated constraint outputs into structured format.
Handles multiple response formats and validates constraint sets.
"""

import re
import ast
import json
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConstraintSet:
    """Structured representation of causal constraints."""
    whitelisted_edges: Set[Tuple[str, str]]
    blacklisted_type_a: Set[Tuple[str, str]]  # Theoretical impossibility
    blacklisted_type_b: Set[Tuple[str, str]]  # Fairness violations
    framework: str
    dataset: str
    temperature: float
    run_id: int
    model: str = "gpt-4"
    provider: str = "openai"
    
    @property
    def all_blacklisted(self) -> Set[Tuple[str, str]]:
        """Combined blacklist (Type A + Type B)."""
        return self.blacklisted_type_a | self.blacklisted_type_b
    
    @property
    def fairness_constraints_only(self) -> Set[Tuple[str, str]]:
        """Only the fairness-based constraints (Type B)."""
        return self.blacklisted_type_b
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'whitelisted_edges': list(self.whitelisted_edges),
            'blacklisted_type_a': list(self.blacklisted_type_a),
            'blacklisted_type_b': list(self.blacklisted_type_b),
            'framework': self.framework,
            'dataset': self.dataset,
            'temperature': self.temperature,
            'run_id': self.run_id,
            'model': self.model,
            'provider': self.provider
        }


class ConstraintParser:
    """
    Parse LLM text responses into structured ConstraintSet objects.
    Handles multiple response formats and validates outputs.
    """
    
    def __init__(self):
        # Regex patterns for different output formats
        self.patterns = {
            'python_list': re.compile(
                r'(whitelisted_edges|blacklisted_edges_type_[AB])\s*=\s*\[(.*?)\]',
                re.DOTALL
            ),
            'markdown_code': re.compile(
                r'```python\n(.*?)\n```',
                re.DOTALL
            ),
            'edge_format': re.compile(
                r'\(\s*["\'](\w+)["\']\s*,\s*["\'](\w+)["\']\s*\)'
            )
        }
    
    def parse_llm_response(
        self,
        response_text: str,
        framework: str,
        dataset: str,
        temperature: float = 0.0,
        run_id: int = 0,
        model: str = "gpt-4",
        provider: str = "openai"
    ) -> ConstraintSet:
        """
        Parse LLM response text into a ConstraintSet.
        
        Args:
            response_text: Raw text output from LLM
            framework: Which normative framework ('statistical', 'anti_classification', etc.)
            dataset: Dataset name ('bank', 'compas', etc.)
            temperature: Sampling temperature used
            run_id: Run identifier for tracking
            model: LLM model name
            provider: LLM provider name
            
        Returns:
            ConstraintSet object with parsed constraints
            
        Raises:
            ValueError: If parsing fails or format is invalid
        """
        # Try to extract Python code block first
        code_blocks = self.patterns['markdown_code'].findall(response_text)
        if code_blocks:
            response_text = '\n'.join(code_blocks)
        
        # Extract constraint lists
        whitelisted = self._extract_edge_list(response_text, 'whitelisted_edges')
        blacklisted_a = self._extract_edge_list(response_text, 'blacklisted_edges_type_A')
        blacklisted_b = self._extract_edge_list(response_text, 'blacklisted_edges_type_B')
        
        # Validate
        self._validate_constraints(whitelisted, blacklisted_a, blacklisted_b, framework)
        
        return ConstraintSet(
            whitelisted_edges=set(whitelisted),
            blacklisted_type_a=set(blacklisted_a),
            blacklisted_type_b=set(blacklisted_b),
            framework=framework,
            dataset=dataset,
            temperature=temperature,
            run_id=run_id,
            model=model,
            provider=provider
        )
    
    def parse_json_response(
        self,
        json_path: Path
    ) -> ConstraintSet:
        """
        Parse LLM response from JSON file (as stored by your prompt_dispatcher).
        
        Args:
            json_path: Path to JSON file containing LLM response
            
        Returns:
            ConstraintSet object with parsed constraints
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata from JSON
        provider = data.get('provider', 'openai')
        model = data.get('model', 'gpt-4')
        dataset = data.get('dataset', 'unknown')
        framework = data.get('framework', 'unknown')
        temperature = data.get('temperature', 0.0)
        
        # Extract run_id from timestamp
        timestamp = data.get('timestamp_utc', '00000000_000000')
        run_id = int(timestamp.split('_')[1]) if '_' in timestamp else 0
        
        # Parse the response text
        response_text = data.get('response', '')
        
        return self.parse_llm_response(
            response_text=response_text,
            framework=framework,
            dataset=dataset,
            temperature=temperature,
            run_id=run_id,
            model=model,
            provider=provider
        )
    
    def _extract_edge_list(
        self,
        text: str,
        list_name: str
    ) -> List[Tuple[str, str]]:
        """
        Extract edge list from text.
        
        Args:
            text: Text containing the edge list
            list_name: Variable name (e.g., 'whitelisted_edges')
            
        Returns:
            List of (source, target) tuples
        """
        # Find the list assignment
        pattern = re.compile(
            rf'{list_name}\s*=\s*\[(.*?)\]',
            re.DOTALL | re.IGNORECASE
        )
        match = pattern.search(text)
        
        if not match:
            # If not found, return empty list (valid for some cases like Statistical Type B)
            return []
        
        list_content = match.group(1)
        
        # Extract all edge tuples
        edges = []
        for match in self.patterns['edge_format'].finditer(list_content):
            source, target = match.groups()
            edges.append((source, target))
        
        return edges
    
    def _validate_constraints(
        self,
        whitelisted: List[Tuple[str, str]],
        blacklisted_a: List[Tuple[str, str]],
        blacklisted_b: List[Tuple[str, str]],
        framework: str
    ):
        """
        Validate constraint sets for consistency.
        
        Raises:
            ValueError: If constraints are invalid
        """
        # Filter out self-loops with warning
        self_loops = []
        for edge_list in [whitelisted, blacklisted_a, blacklisted_b]:
            i = 0
            while i < len(edge_list):
                if edge_list[i][0] == edge_list[i][1]:
                    self_loops.append(edge_list[i])
                    edge_list.pop(i)
                else:
                    i += 1
        
        if self_loops:
            print(f"Warning: Filtered out {len(self_loops)} self-loop(s): {self_loops}")
        
        # Check for overlap between whitelist and blacklist
        white_set = set(whitelisted)
        black_set = set(blacklisted_a) | set(blacklisted_b)
        
        overlap = white_set & black_set
        if overlap:
            raise ValueError(f"Overlap between whitelist and blacklist: {overlap}")
        
        # Framework-specific validation
        if framework.lower() == 'statistical':
            if blacklisted_b:
                print(f"Warning: Statistical framework has {len(blacklisted_b)} Type B edges (expected 0)")
    
    def parse_batch(
        self,
        results_dir: Path,
        framework: str = None,
        dataset: str = None
    ) -> List[ConstraintSet]:
        """
        Parse all LLM responses in a directory for a given framework/dataset.
        
        Args:
            results_dir: Directory containing LLM response JSON files
            framework: Framework name (optional, will parse all if not specified)
            dataset: Dataset name (optional, will parse all if not specified)
            
        Returns:
            List of ConstraintSet objects
        """
        constraint_sets = []
        
        # Build pattern for matching files
        pattern_parts = []
        if dataset:
            pattern_parts.append(dataset)
        else:
            pattern_parts.append('*')
        
        if framework:
            pattern_parts.append(framework)
        else:
            pattern_parts.append('*')
        
        pattern_parts.append('*.json')
        pattern = '_'.join(pattern_parts[:2]) + '_' + pattern_parts[2]
        
        for file_path in results_dir.glob(pattern):
            try:
                constraint_set = self.parse_json_response(file_path)
                constraint_sets.append(constraint_set)
            except Exception as e:
                print(f"Warning: Failed to parse {file_path.name}: {e}")
        
        return constraint_sets


# Example usage
if __name__ == "__main__":
    parser = ConstraintParser()
    
    # Test parsing from sample response
    sample_response = """
    # Define whitelisted edges
    whitelisted_edges = [
        ("age", "job"),
        ("education", "job"),
    ]
    
    # Define blacklisted edges (theoretical impossibility)
    blacklisted_edges_type_A = [
        ("job", "age"),
        ("balance", "age"),
    ]
    
    # Define blacklisted edges (fairness)
    blacklisted_edges_type_B = [
        ("age", "y"),
        ("marital", "y"),
    ]
    """
    
    constraint_set = parser.parse_llm_response(
        response_text=sample_response,
        framework='anti_classification',
        dataset='bank',
        temperature=0.0,
        run_id=1
    )
    
    print(f"Parsed {len(constraint_set.all_blacklisted)} total constraints")
    print(f"Fairness constraints: {constraint_set.fairness_constraints_only}")
