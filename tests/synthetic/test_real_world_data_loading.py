import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.experiments.main_runner import (
    DatasetConfig,
    ExperimentConfig,
    FairCausalDiscoveryRunner,
    SUPPORTED_DATASETS,
    build_argument_parser,
    validate_cli_args,
)


class TestRealWorldDataLoading(unittest.TestCase):
    def test_supported_dataset_configs_are_complete(self):
        self.assertEqual(
            DatasetConfig.get_available_datasets(),
            sorted(SUPPORTED_DATASETS),
        )

        for dataset_name in SUPPORTED_DATASETS:
            config = DatasetConfig.load_dataset_config(dataset_name)
            self.assertIn("analysis_data_path", config["data_settings"])
            self.assertIn("data_loader_module", config["data_settings"])

    def test_committed_processed_data_load_for_all_supported_datasets(self):
        for dataset_name in SUPPORTED_DATASETS:
            with self.subTest(dataset=dataset_name):
                config = ExperimentConfig(dataset_name, "baseline")
                runner = FairCausalDiscoveryRunner(config)
                dataframe = runner._load_experiment_data()

                self.assertFalse(dataframe.empty)
                expected_columns = set(config.dataset_config["fairness_settings"]["protected_attrs"])
                expected_columns.add(config.dataset_config["fairness_settings"]["target"])
                self.assertTrue(expected_columns.issubset(dataframe.columns))

    def test_missing_processed_data_has_clear_error(self):
        config = ExperimentConfig("law", "baseline")
        config.dataset_config["data_settings"]["analysis_data_path"] = (
            "data/does-not-exist.csv"
        )
        runner = FairCausalDiscoveryRunner(config)

        with self.assertRaisesRegex(FileNotFoundError, "Canonical processed dataset"):
            runner._load_experiment_data()

    def test_raw_preprocessing_requires_a_path(self):
        config = ExperimentConfig(
            "law",
            "baseline",
            custom_config={"reprocess_data": True},
        )
        runner = FairCausalDiscoveryRunner(config)

        with self.assertRaisesRegex(ValueError, "--raw-data-path"):
            runner._load_experiment_data()

    def test_raw_preprocessing_stays_in_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            raw_path.write_text("feature,target\n0,1\n", encoding="utf-8")
            expected = pd.DataFrame({"feature": [0], "target": [1]})

            config = ExperimentConfig(
                "law",
                "baseline",
                custom_config={
                    "reprocess_data": True,
                    "raw_data_path": str(raw_path),
                },
            )
            runner = FairCausalDiscoveryRunner(config)
            loader = mock.Mock()
            loader.load_and_preprocess_data.return_value = {
                "preprocessed": expected
            }
            runner._load_data_loader = mock.Mock(
                side_effect=lambda: setattr(runner, "data_loader", loader)
            )

            actual = runner._load_experiment_data()

            pd.testing.assert_frame_equal(actual, expected)
            loader.load_and_preprocess_data.assert_called_once_with(
                str(raw_path.resolve()),
                export_path=None,
                log_file_path=None,
            )

    def test_default_cli_dataset_is_supported(self):
        parser = build_argument_parser()
        args = parser.parse_args([])

        self.assertEqual(args.dataset, "law")
        self.assertIn(args.dataset, SUPPORTED_DATASETS)

    def test_cli_reprocessing_arguments_are_paired(self):
        parser = build_argument_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validate_cli_args(parser, parser.parse_args(["--reprocess-data"]))

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                validate_cli_args(
                    parser,
                    parser.parse_args(["--raw-data-path", "raw.csv"]),
                )

        valid_args = parser.parse_args(
            ["--reprocess-data", "--raw-data-path", "raw.csv"]
        )
        validate_cli_args(parser, valid_args)


if __name__ == "__main__":
    unittest.main()
