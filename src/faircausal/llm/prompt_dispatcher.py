"""Send generated prompts to configured LLM providers and persist responses."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from .clients import BaseLLMClient, build_clients
    from .config import ProviderSettings
except ImportError:
    # Handle direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from clients import BaseLLMClient, build_clients
    from config import ProviderSettings

_PROMPT_PATTERN = re.compile(r"FinalPrompt_(?P<dataset>[A-Za-z0-9-]+)_(?P<framework>[A-Za-z]+)\.txt")
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "final_prompts"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "llm_responses"


def _parse_prompt_metadata(path: Path) -> Optional[Tuple[str, str]]:
    match = _PROMPT_PATTERN.match(path.name)
    if not match:
        return None
    dataset = match.group("dataset")
    framework = match.group("framework")
    return dataset, framework


def _discover_prompts(prompts_dir: Path) -> List[Path]:
    return sorted(prompts_dir.glob("FinalPrompt_*_*.txt"))


def _should_include(value: str, filters: Optional[Set[str]]) -> bool:
    if not filters:
        return True
    return value.lower() in filters


def _persist_response(
    output_dir: Path,
    dataset: str,
    framework: str,
    provider: str,
    prompt_path: Path,
    prompt_text: str,
    response_text: str,
    client: BaseLLMClient,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_dir = output_dir / dataset.lower()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dataset}_{framework}_{provider}_{timestamp}.json"
    
    # Use relative path for anonymity - only store filename
    prompt_file_relative = prompt_path.name
    
    record = {
        "provider": provider,
        "model": client.model,
        "dataset": dataset,
        "framework": framework,
        "prompt_file": prompt_file_relative,
        "temperature": client.temperature,
        "max_output_tokens": client.max_output_tokens,
        "timestamp_utc": timestamp,
        "prompt": prompt_text,
        "response": response_text,
    }
    output_path = dataset_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return output_path


def dispatch_prompts(
    prompts_dir: Path,
    output_dir: Path,
    datasets: Optional[List[str]],
    frameworks: Optional[List[str]],
    providers: Optional[List[str]],
    config_file: Optional[str],
    max_prompts: Optional[int],
) -> Dict[str, List[Path]]:
    settings = ProviderSettings.load(config_file)
    clients = build_clients(settings)

    if providers:
        providers_lower = {name.lower() for name in providers}
        clients = {name: client for name, client in clients.items() if name in providers_lower}

    if not clients:
        raise RuntimeError("No LLM providers are configured. Set the appropriate API keys or config file.")

    prompts = _discover_prompts(prompts_dir)
    if not prompts:
        raise FileNotFoundError(f"No prompt files found in {prompts_dir}")

    dataset_filters = {item.lower() for item in datasets} if datasets else None
    framework_filters = {item.lower() for item in frameworks} if frameworks else None

    results: Dict[str, List[Path]] = {provider: [] for provider in clients}
    processed = 0

    for prompt_path in prompts:
        metadata = _parse_prompt_metadata(prompt_path)
        if not metadata:
            continue
        dataset, framework = metadata
        if not _should_include(dataset, dataset_filters):
            continue
        if not _should_include(framework, framework_filters):
            continue

        prompt_text = prompt_path.read_text(encoding="utf-8")

        for provider_name, client in clients.items():
            try:
                response_text = client.generate(prompt_text)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] {provider_name} failed for {prompt_path.name}: {exc}")
                continue

            output_path = _persist_response(
                output_dir=output_dir,
                dataset=dataset,
                framework=framework,
                provider=provider_name,
                prompt_path=prompt_path,
                prompt_text=prompt_text,
                response_text=response_text,
                client=client,
            )
            results[provider_name].append(output_path)

        processed += 1
        if max_prompts is not None and processed >= max_prompts:
            break

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send generated prompts to configured LLM providers.")
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=_DEFAULT_PROMPTS_DIR,
        help="Directory containing FinalPrompt_*.txt files (default: templates/final_prompts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where provider responses will be stored (default: llm_responses).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of dataset suffixes to process (e.g., bank compas).",
    )
    parser.add_argument(
        "--frameworks",
        nargs="*",
        help="Optional subset of framework suffixes to process (e.g., Statistical AntiClassification).",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        choices=["openai", "anthropic", "google"],
        help="Limit dispatch to specific providers.",
    )
    parser.add_argument(
        "--config-file",
        help="Optional JSON file containing provider API keys and defaults.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        help="Limit the number of prompt files processed (useful for smoke tests).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = dispatch_prompts(
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        datasets=args.datasets,
        frameworks=args.frameworks,
        providers=args.providers,
        config_file=args.config_file,
        max_prompts=args.max_prompts,
    )

    print("\nDispatch complete. Saved responses:")
    for provider, files in results.items():
        print(f"  {provider}: {len(files)} files")
        for path in files:
            print(f"    - {path}")


if __name__ == "__main__":
    main()
