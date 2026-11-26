"""Versioned CLI output checks using fictional CSVs only."""

import builtins
from contextlib import redirect_stderr, redirect_stdout
import csv
from decimal import Decimal, ROUND_HALF_UP, localcontext
import importlib.util
import io
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from darwizzy_wirtzzy import baseline, regression
from darwizzy_wirtzzy.cli import main

ROOT = Path(__file__).resolve().parents[1]
HAS_MODEL = importlib.util.find_spec("sklearn") is not None
METRICS = ("mean_absolute_error_eur", "median_absolute_error_eur", "mean_signed_error_eur")


def fictional_row(player_id, season="2021/22", position="Forward", amount="10.00", **values):
    end = int(season[:4]) + 1
    row = dict(zip(baseline.FIELDS, (
        player_id, "Invented " + player_id, season, "Paper Town", position,
        "24", "100", "0", "0", f"{end}-05-20", f"{end}-06-01", amount,
    )))
    return {**row, **values}


def expected_metrics(scored, estimates):
    # Calculate from unrounded predictions, independently of result formatters.
    with localcontext() as context:
        context.prec = 330
        errors = [estimates[(row.player_id, row.season)] - row.reference_eur for row in scored]
        values = ((sum(abs(value) for value in errors) / len(errors),
                   median(abs(value) for value in errors), sum(errors) / len(errors))
                  if errors else (None, None, None))
        return {name: format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")
                if value is not None else None for name, value in zip(METRICS, values)}


class JsonCase:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "fictional.csv"
        self.records = [
            fictional_row("t1", amount="10.01"), fictional_row("t2", amount="10.02"),
            fictional_row("t3", position="Defender", amount="2.00"),
            fictional_row("t-zero", amount="999.00", minutes="0"),
            fictional_row("t-late", amount="900.00", reference_valuation_date="2024-06-10"),
            fictional_row("future", "2025/26", amount="800.00"),
            fictional_row("e1", "2023/24", amount="10.01"),
            fictional_row("e2", "2023/24", amount="10.03"),
            fictional_row("e3", "2023/24", "Defender", "1.99"),
            fictional_row("e-zero", "2023/24", amount="1000.00", minutes="0"),
            fictional_row("e-unseen", "2023/24", "Goalkeeper", "1.00"),
        ]
        self.write()

    def write(self):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=baseline.FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.records)

    def invoke(self, action="evaluate", *options):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main([action, "--input", str(self.path), "--season", "2023/24", *options])
            except SystemExit as error:
                code = error.code
        return code, out.getvalue(), err.getvalue()

    def structured(self, action="evaluate", *options):
        code, out, err = self.invoke(action, *options, "--format", "json")
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertTrue(out.endswith("\n"))
        self.assertEqual(out.count("\n"), 1)
        value = json.loads(out, parse_constant=lambda token: self.fail("Nonfinite JSON token: " + token))
        self.assertIsInstance(value, dict)
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["command"], action)
        self.assertEqual(value["currency"], "EUR")
        return value, out

    def assert_metrics(self, actual, expected):
        self.assertEqual({field: actual[field] for field in METRICS}, expected)
        for field in METRICS:
            if actual[field] is not None:
                self.assertIsInstance(actual[field], str)
                self.assertRegex(actual[field], r"^-?\d+\.\d{2}$")


class JsonContractTests(JsonCase, unittest.TestCase):
    def test_default_evaluation_retains_exact_legacy_output(self):
        expected = {
            "evaluation_season": "2023/24", "training_seasons": ["2021/22"],
            "training_row_count": 3, "holdout_row_count": 5, "evaluated_row_count": 3,
            "skipped_zero_minutes": 1, "skipped_unseen_position": 1,
            "positions_without_training": ["Goalkeeper"],
            "position_medians_eur": {"Defender": "2.00", "Forward": "10.02"},
            "mean_absolute_error_eur": "0.01",
        }
        expected_text = json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        for options in ((), ("--format", "legacy")):
            with self.subTest(options=options):
                code, out, err = self.invoke("evaluate", *options)
                self.assertEqual((code, out, err), (0, expected_text, ""))

    def test_default_lookup_retains_legacy_shape_and_unrounded_gap(self):
        expected = {
            "player_id": "e2", "player_name": "Invented e2", "season": "2023/24",
            "club": "Paper Town", "position": "Forward", "status": "ok",
            "estimate_eur": "10.02", "reference_eur": "10.03",
            "reference_valuation_date": "2024-06-01", "signed_gap_eur": "-0.02",
        }
        code, out, err = self.invoke("lookup", "--player-id", "e2")
        self.assertEqual((code, out, err), (0, json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False) + "\n", ""))

    def test_median_json_reports_actual_input_counts_and_all_metrics(self):
        result, _ = self.structured()
        self.assertEqual(result["model"], "median")
        self.assertEqual(result["evaluation_season"], "2023/24")
        self.assertEqual(result["training_seasons"], ["2021/22"])
        self.assertEqual(result["training_row_count"], 3)
        self.assertEqual(result["holdout_row_count"], 5)
        self.assertEqual(result["eligible_evaluation_row_count"], 3)
        self.assertEqual(result["scored_row_count"], 3)
        self.assertEqual(result["skipped_zero_minutes"], 1)
        self.assertEqual(result["skipped_unseen_position"], 1)
        self.assertEqual(result["positions_without_training"], ["Goalkeeper"])
        self.assertEqual(list(result["models"]), ["median"])
        self.assertEqual(result["models"]["median"]["eligible_row_count"], 3)
        self.assertEqual(result["models"]["median"]["scored_row_count"], 3)
        self.assert_metrics(result["models"]["median"], {
            "mean_absolute_error_eur": "0.01", "median_absolute_error_eur": "0.01",
            "mean_signed_error_eur": "0.00",
        })
        self.assertNotIn("candidate_count", result)
        self.assertNotIn("provider_exclusions", result)

    def test_lookup_json_preserves_identity_reference_date_and_gap(self):
        result, _ = self.structured("lookup", "--player-id", "e2")
        self.assertEqual(result["model"], "median")
        self.assertEqual(result["player_id"], "e2")
        self.assertEqual(result["season"], "2023/24")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["estimate_eur"], "10.02")
        self.assertEqual(result["reference_eur"], "10.03")
        self.assertEqual(result["reference_valuation_date"], "2024-06-01")
        self.assertEqual(result["signed_gap_eur"], "-0.02")

    def test_successful_unscorable_lookups_retain_status_and_nulls(self):
        for player_id, status in (("e-zero", "zero_minutes"), ("e-unseen", "no_training_for_position")):
            with self.subTest(status=status):
                result, _ = self.structured("lookup", "--player-id", player_id)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["estimate_eur"])
                self.assertIsNone(result["signed_gap_eur"])
                self.assertRegex(result["reference_eur"], r"^\d+\.\d{2}$")

    def test_unscorable_evaluation_keeps_coverage_and_null_metrics(self):
        for row in self.records:
            if row["season"] == "2023/24" and row["minutes"] != "0":
                row["position"] = "Goalkeeper"
        self.write()
        result, _ = self.structured()
        self.assertEqual(result["holdout_row_count"], 5)
        self.assertEqual(result["eligible_evaluation_row_count"], 0)
        self.assertEqual(result["scored_row_count"], 0)
        self.assertEqual(result["skipped_unseen_position"], 4)
        self.assert_metrics(result["models"]["median"], dict.fromkeys(METRICS))

    def test_unicode_quotes_backslashes_and_newlines_round_trip(self):
        name = '\u00c9lodie "Papier" \\ \u96ea\nLine Two'
        club = "Fictional M\u00fcnchen"
        for row in self.records:
            if row["player_id"] == "e1":
                row.update(player_name=name, club=club)
        self.write()
        result, out = self.structured("lookup", "--player-id", "e1")
        self.assertEqual(result["player_name"], name)
        self.assertEqual(result["club"], club)
        self.assertTrue(out.isascii())
        self.assertIn("\\u", out)

    def test_json_bytes_are_deterministic_under_reordered_csv(self):
        first, before = self.structured()
        self.records.reverse()
        self.write()
        second, after = self.structured()
        self.assertEqual(first, second)
        self.assertEqual(before, after)

    def test_invalid_input_and_missing_key_have_empty_stdout(self):
        scenarios = [
            ("lookup", ["--player-id", "absent"], "Player-season not found"),
            ("evaluate", ["--season", "2023/25"], "Invalid season"),
            ("evaluate", ["--input", str(self.path.with_name("missing.csv"))], "error:"),
        ]
        for action, options, message in scenarios:
            with self.subTest(action=action, options=options):
                code, out, err = self.invoke(action, *options, "--format", "json")
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn(message, err)
        self.records[0]["reference_eur"] = "NaN"
        self.write()
        code, out, err = self.invoke("evaluate", "--format", "json")
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("CSV line", err)

    def test_argument_errors_and_help_follow_existing_convention(self):
        for options, message in ((["--format", "yaml"], "invalid choice"),
                                 (["--compare", "--model", "ridge", "--format", "json"], "omit --model")):
            with self.subTest(options=options):
                code, out, err = self.invoke("evaluate", *options)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn(message, err)
        code, out, err = self.invoke("evaluate", "--help")
        self.assertEqual(code, 0)
        self.assertIn("--format", out)
        self.assertEqual(err, "")

    def test_missing_optional_model_dependencies_keep_json_stdout_empty(self):
        original_import = builtins.__import__
        def without_model(name, *args, **kwargs):
            if name == "sklearn" or name.startswith("sklearn."):
                raise ModuleNotFoundError("No optional model dependencies")
            return original_import(name, *args, **kwargs)
        for action, options in (("evaluate", ["--model", "ridge"]),
                                ("evaluate", ["--compare"]),
                                ("lookup", ["--model", "ridge", "--player-id", "e1"])):
            with self.subTest(action=action, options=options), patch("builtins.__import__", side_effect=without_model):
                code, out, err = self.invoke(action, *options, "--format", "json")
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn('pip install ".[model]"', err)
        with patch("builtins.__import__", side_effect=without_model):
            result, _ = self.structured()
        self.assertEqual(result["model"], "median")

    def test_nonfinite_serialization_fails_before_writing_stdout(self):
        rows = baseline.load_csv(self.path)
        valid = baseline.lookup(rows, "e1", "2023/24")
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), patch("darwizzy_wirtzzy.cli.lookup", return_value={**valid, "estimate_eur": value}):
                code, out, err = self.invoke("lookup", "--player-id", "e1", "--format", "json")
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertTrue(err.startswith("error:"), err)

    def test_metrics_use_predictions_before_cent_rounding(self):
        rows = baseline.load_csv(self.path)
        scored = [row for row in rows if row.season == "2023/24" and row.player_id in {"e1", "e2", "e3"}]
        estimates = {(row.player_id, row.season): row.reference_eur +
                     (Decimal("-0.0149") if row.player_id == "e3" else Decimal("0.0049")) for row in scored}
        model = SimpleNamespace(predict=lambda holdout: estimates)
        with patch.object(regression, "fit_training", return_value=model):
            result, _ = self.structured("evaluate", "--model", "ridge")
        expected = expected_metrics(scored, estimates)
        self.assertEqual(expected["mean_absolute_error_eur"], "0.01")
        self.assertEqual(expected["median_absolute_error_eur"], "0.00")
        self.assert_metrics(result["models"]["ridge"], expected)
        # Rounding predictions first would incorrectly report zero MAE here.
        rounded = {key: value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) for key, value in estimates.items()}
        self.assertEqual(expected_metrics(scored, rounded)["mean_absolute_error_eur"], "0.00")

    def test_comparison_scores_the_intersection_not_each_available_set(self):
        rows = baseline.load_csv(self.path)
        paired = [row for row in rows if row.season == "2023/24" and row.player_id in {"e1", "e3"}]
        ridge_estimates = {(row.player_id, row.season): row.reference_eur +
                           (Decimal("1") if row.player_id == "e1" else Decimal("-1")) for row in paired}
        model = SimpleNamespace(predict=lambda holdout: ridge_estimates)
        with patch.object(regression, "fit_training", return_value=model):
            result, _ = self.structured("evaluate", "--compare")
        self.assertEqual(result["model"], "comparison")
        self.assertEqual(result["eligible_evaluation_row_count"], 3)
        self.assertEqual(result["paired_row_count"], 2)
        self.assertEqual(result["scored_row_count"], 2)
        self.assertEqual(result["paired_player_seasons"], [{"player_id": row.player_id, "season": row.season} for row in paired])
        self.assertEqual(result["models"]["median"]["eligible_row_count"], 3)
        self.assertEqual(result["models"]["ridge"]["eligible_row_count"], 2)
        for name in ("median", "ridge"):
            self.assertEqual(result["models"][name]["scored_row_count"], 2)
        medians = {("e1", "2023/24"): Decimal("10.015"), ("e3", "2023/24"): Decimal("2.00")}
        self.assert_metrics(result["models"]["median"], expected_metrics(paired, medians))
        self.assert_metrics(result["models"]["ridge"], expected_metrics(paired, ridge_estimates))

    def test_module_entry_point_emits_the_same_successful_object(self):
        value, expected = self.structured("lookup", "--player-id", "e2")
        command = [sys.executable, "-B", "-m", "darwizzy_wirtzzy", "lookup", "--input", str(self.path),
                   "--season", "2023/24", "--player-id", "e2", "--format", "json"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual(json.loads(result.stdout), value)
        self.assertEqual(result.stdout, expected)


@unittest.skipUnless(HAS_MODEL, "Install the optional model dependencies to test Ridge")
class ModelJsonContractTests(JsonCase, unittest.TestCase):
    def test_ridge_evaluation_metrics_match_unrounded_predictions(self):
        rows = baseline.load_csv(self.path)
        training, holdout = baseline.split(rows, "2023/24")
        estimates = regression.fit_training(training).predict(holdout)
        scored = [row for row in holdout if (row.player_id, row.season) in estimates]
        result, _ = self.structured("evaluate", "--model", "ridge")
        self.assertEqual(result["model"], "ridge")
        self.assertEqual(result["training_row_count"], len(training))
        self.assertEqual(result["scored_row_count"], len(scored))
        self.assertEqual(result["models"]["ridge"]["eligible_row_count"], len(estimates))
        self.assert_metrics(result["models"]["ridge"], expected_metrics(scored, estimates))

    def test_comparison_and_lookup_preserve_predictions_and_model_order(self):
        rows = baseline.load_csv(self.path)
        training, holdout = baseline.split(rows, "2023/24")
        ridge_estimates = regression.fit_training(training).predict(holdout)
        paired = [row for row in holdout if row.minutes > 0 and row.position in {item.position for item in training}]
        medians = {position: median(row.reference_eur for row in training if row.position == position)
                   for position in {row.position for row in training}}
        median_estimates = {(row.player_id, row.season): medians[row.position] for row in paired}
        result, before = self.structured("evaluate", "--compare")
        self.assertEqual(list(result["models"]), ["median", "ridge"])
        self.assertEqual(result["paired_row_count"], len(paired))
        self.assertEqual(result["paired_player_seasons"], [{"player_id": row.player_id, "season": row.season} for row in paired])
        self.assertEqual(result["skipped_reasons"], {"zero_minutes": 1, "no_training_for_position": 1})
        self.assert_metrics(result["models"]["median"], expected_metrics(paired, median_estimates))
        self.assert_metrics(result["models"]["ridge"], expected_metrics(paired, ridge_estimates))
        original = regression.lookup(rows, "e2", "2023/24")
        looked_up, _ = self.structured("lookup", "--player-id", "e2", "--model", "ridge")
        for field, value in original.items():
            self.assertEqual(looked_up[field], value)
        self.records.reverse()
        self.write()
        _, after = self.structured("evaluate", "--compare")
        self.assertEqual(before, after)

    def test_no_comparison_pair_reports_null_metrics_for_both_models(self):
        for row in self.records:
            if row["season"] == "2023/24" and row["minutes"] != "0":
                row["position"] = "Goalkeeper"
        self.write()
        result, _ = self.structured("evaluate", "--compare")
        self.assertEqual(result["paired_player_seasons"], [])
        self.assertEqual(result["paired_row_count"], 0)
        self.assertEqual(result["scored_row_count"], 0)
        for model in result["models"].values():
            self.assert_metrics(model, dict.fromkeys(METRICS))


if __name__ == "__main__":
    unittest.main()
