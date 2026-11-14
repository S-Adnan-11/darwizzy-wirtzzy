"""Fictional regression, paired coverage and chronology checks."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from decimal import Decimal
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from darwizzy_wirtzzy import baseline, regression
from darwizzy_wirtzzy.cli import main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/fictional_regression_seasons.csv"
HAS_MODEL = importlib.util.find_spec("sklearn") is not None


class DependencyTests(unittest.TestCase):
    def test_missing_extra_has_clear_cli_error_and_default_still_works(self):
        original_import = builtins.__import__
        def without_model(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ModuleNotFoundError("No optional model dependencies")
            return original_import(name, *args, **kwargs)
        for option in (["--model", "ridge"], ["--compare"]):
            out, err = io.StringIO(), io.StringIO()
            with patch("builtins.__import__", side_effect=without_model), redirect_stdout(out), redirect_stderr(err):
                code = main(["evaluate", "--input", str(EXAMPLE), "--season", "2025/26", *option])
            self.assertEqual(code, 2)
            self.assertEqual(out.getvalue(), "")
            self.assertIn('pip install ".[model]"', err.getvalue())
        with patch("builtins.__import__", side_effect=without_model), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["evaluate", "--input", str(EXAMPLE), "--season", "2025/26"]), 0)

    def test_insufficient_training_is_clear_before_model_import(self):
        rows = baseline.load_csv(EXAMPLE)
        training = next(row for row in rows if row.season == "2023/24")
        holdout = next(row for row in rows if row.season == "2025/26")
        with self.assertRaisesRegex(baseline.InputError, "at least two eligible historical"):
            regression.evaluate([training, holdout], "2025/26")


@unittest.skipUnless(HAS_MODEL, "Install the optional model dependencies to test Ridge")
class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.rows = baseline.load_csv(EXAMPLE)

    def test_regression_cli_and_lookup_keep_money_conventions(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["lookup", "--input", str(EXAMPLE), "--season", "2025/26", "--player-id", "fictional-r08", "--model", "ridge"])
        self.assertEqual(code, 0)
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["reference_valuation_date"], "2026-06-10")
        self.assertEqual(result["reference_eur"], "36000000.00")
        self.assertGreater(Decimal(result["estimate_eur"]), 0)
        # Gap uses the unrounded estimate, rounded only for display.
        self.assertLess(abs(Decimal(result["signed_gap_eur"]) - (Decimal(result["estimate_eur"]) - Decimal(result["reference_eur"]))), Decimal("0.02"))

    def test_predictions_are_deterministic_under_input_reordering(self):
        self.assertEqual(regression.evaluate(self.rows, "2025/26"), regression.evaluate(list(reversed(self.rows)), "2025/26"))

    def test_holdout_amounts_change_errors_without_changing_predictions(self):
        changed = [replace(row, reference_eur=Decimal("999999999")) if row.season == "2025/26" else row for row in self.rows]
        self.assertEqual(regression.lookup(self.rows, "fictional-r08", "2025/26")["estimate_eur"],
                         regression.lookup(changed, "fictional-r08", "2025/26")["estimate_eur"])
        self.assertNotEqual(regression.evaluate(self.rows, "2025/26")["mean_absolute_error_eur"], regression.evaluate(changed, "2025/26")["mean_absolute_error_eur"])

    def test_future_rows_do_not_affect_predictions(self):
        future = replace(self.rows[0], player_id="fictional-future", season="2027/28", reference_eur=Decimal("999999999"),
                         feature_cutoff_date=date(2028, 5, 20), reference_valuation_date=date(2028, 6, 1), minutes=999999)
        self.assertEqual(regression.evaluate(self.rows, "2025/26"), regression.evaluate([*self.rows, future], "2025/26"))

    def test_unavailable_historical_labels_are_excluded_before_preprocessing(self):
        altered = replace(self.rows[0], reference_valuation_date=date(2026, 5, 25), minutes=999999)
        rows = [altered, *self.rows[1:]]
        training, holdout = baseline.split(rows, "2025/26")
        self.assertNotIn(altered, training)
        self.assertEqual(regression.evaluate(rows, "2025/26"), regression.evaluate(self.rows[1:], "2025/26"))
        model = regression.fit_training(training)
        scaler = model.pipeline.named_steps["features"].named_transformers_["numeric"]
        self.assertAlmostEqual(scaler.mean_[1], sum(row.minutes for row in training) / len(training))
        self.assertNotAlmostEqual(scaler.mean_[1], sum(row.minutes for row in [*training, *holdout]) / (len(training) + len(holdout)))

    def test_display_and_identity_fields_do_not_enter_features(self):
        changed = [replace(row, player_id=f"renamed-{row.player_id}", player_name="Another Fictional Name", club="Other Invented Club") for row in self.rows]
        original = regression.lookup(self.rows, "fictional-r08", "2025/26")
        renamed = regression.lookup(changed, "renamed-fictional-r08", "2025/26")
        self.assertEqual(original["estimate_eur"], renamed["estimate_eur"])

    def test_paired_keys_match_both_models_eligible_rows(self):
        result = regression.compare(self.rows, "2025/26")
        expected_keys = [{"player_id": f"fictional-r{number:02}", "season": "2025/26"} for number in (1, 4, 6, 8)]
        self.assertEqual(result["paired_player_seasons"], expected_keys)
        self.assertEqual(result["paired_row_count"], 4)
        self.assertEqual(result["skipped_reasons"], {"zero_minutes": 1, "no_training_for_position": 0})
        for model in result["models"].values():
            self.assertEqual(model["paired_row_count"], 4)
            self.assertEqual(model["eligible_row_count"], 4)
        self.assertEqual(result["models"]["median"]["mean_absolute_error_eur"], baseline.evaluate(self.rows, "2025/26")["mean_absolute_error_eur"])
        self.assertEqual(result["models"]["ridge"]["mean_absolute_error_eur"], regression.evaluate(self.rows, "2025/26")["mean_absolute_error_eur"])

    def test_unseen_position_is_skipped_by_both_models(self):
        rows = [row for row in self.rows if row.season == "2025/26" or row.position != "Goalkeeper"]
        result = regression.compare(rows, "2025/26")
        self.assertEqual(result["paired_row_count"], 3)
        self.assertEqual(result["skipped_reasons"]["no_training_for_position"], 1)
        lookup = regression.lookup(rows, "fictional-r01", "2025/26")
        self.assertEqual(lookup["status"], "no_training_for_position")
        self.assertIsNone(lookup["estimate_eur"])
        self.assertIsNone(lookup["signed_gap_eur"])

    def test_no_paired_rows_reports_null_errors(self):
        rows = [row for row in self.rows if (row.season != "2025/26" and row.position == "Defender") or
                (row.season == "2025/26" and row.position != "Defender")]
        result = regression.compare(rows, "2025/26")
        self.assertEqual(result["paired_row_count"], 0)
        self.assertEqual(result["paired_player_seasons"], [])
        self.assertTrue(all(model["mean_absolute_error_eur"] is None for model in result["models"].values()))

    def test_zero_goals_and_assists_are_usable_and_zero_minutes_are_visible(self):
        lookup = regression.lookup(self.rows, "fictional-r01", "2025/26")
        self.assertEqual(lookup["status"], "ok")
        zero = regression.lookup(self.rows, "fictional-r09", "2025/26")
        self.assertEqual(zero["status"], "zero_minutes")
        self.assertIsNone(zero["estimate_eur"])
        self.assertEqual(regression.evaluate(self.rows, "2025/26")["skipped_zero_minutes"], 1)

    def test_nonfinite_and_underflow_predictions_are_rejected(self):
        training, holdout = baseline.split(self.rows, "2025/26")
        model = regression.fit_training(training)
        for log_value in (float("nan"), float("inf"), 1000, -1000, -20):
            with self.subTest(log_value=log_value), patch.object(model.pipeline, "predict", return_value=[log_value]):
                with self.assertRaisesRegex(baseline.InputError, "estimate|EUR"):
                    model.predict([holdout[0]])

    def test_finite_large_estimate_remains_reportable(self):
        training, holdout = baseline.split(self.rows, "2025/26")
        model = regression.fit_training(training)
        with patch.object(model.pipeline, "predict", return_value=[100.0]):
            estimate = model.predict([holdout[0]])
        self.assertGreater(next(iter(estimate.values())), Decimal("1e40"))

    def test_extreme_counts_have_clear_finite_range_errors(self):
        training, holdout = baseline.split(self.rows, "2025/26")
        with self.assertRaisesRegex(baseline.InputError, "finite numeric range"):
            regression.fit_training([replace(training[0], minutes=10**400), *training[1:]])
        model = regression.fit_training(training)
        with self.assertRaisesRegex(baseline.InputError, "finite numeric range"):
            model.predict([replace(holdout[0], minutes=10**400)])


if __name__ == "__main__":
    unittest.main()
