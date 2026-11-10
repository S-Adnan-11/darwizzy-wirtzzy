"""Tests use fictional player-seasons and never load provider data."""

from contextlib import redirect_stderr, redirect_stdout
import csv
from datetime import date
from decimal import Decimal
import io
from pathlib import Path
import tempfile
import unittest

from darwizzy_wirtzzy.baseline import FIELDS, InputError, evaluate, load_csv, lookup
from darwizzy_wirtzzy.cli import main


def fictional_row(player_id="t1", season="2021/22", position="Forward", amount="10"):
    end_year = int(season[:4]) + 1
    return dict(zip(FIELDS, (
        player_id, f"Invented {player_id}", season, "Paper Town", position, "24", "100", "0", "0",
        f"{end_year}-05-20", f"{end_year}-06-01", amount,
    )))


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "independent.csv"
        # Even median (10 + 30)/2 = 20; odd median (2, 6, 10) = 6.
        self.records = [fictional_row("t1", amount="10"), fictional_row("t2", amount="30"),
                        fictional_row("t3", position="Defender", amount="2"),
                        fictional_row("t4", position="Defender", amount="6"),
                        fictional_row("t5", position="Defender", amount="10"),
                        fictional_row("e1", "2023/24", amount="26"),
                        fictional_row("e2", "2023/24", "Defender", "4")]

    def write(self, records=None, fields=FIELDS):
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records if records is None else records)
        return self.path

    def rows(self):
        return load_csv(self.write())

    def test_independent_medians_and_mae(self):
        result = evaluate(self.rows(), "2023/24")
        self.assertEqual(result["position_medians_eur"], {"Defender": "6.00", "Forward": "20.00"})
        self.assertEqual(result["training_row_count"], 5)
        self.assertEqual(result["holdout_row_count"], 2)
        self.assertEqual(result["evaluated_row_count"], 2)
        self.assertEqual(result["mean_absolute_error_eur"], "4.00")  # (6 + 2) / 2

    def test_holdout_and_future_labels_cannot_change_fit(self):
        self.records[-2]["reference_eur"] = "999999"
        self.records.append(fictional_row("future", "2025/26", amount="888888"))
        result = evaluate(self.rows(), "2023/24")
        self.assertEqual(result["training_seasons"], ["2021/22"])
        self.assertEqual(result["position_medians_eur"]["Forward"], "20.00")
        self.assertEqual(result["holdout_row_count"], 2)

    def test_historical_label_unavailable_at_cutoff_is_excluded(self):
        self.records[0]["reference_valuation_date"] = "2024-05-21"
        self.assertEqual(evaluate(self.rows(), "2023/24")["position_medians_eur"]["Forward"], "30.00")

    def test_lookup_reference_date_and_signed_gap(self):
        result = lookup(self.rows(), "e1", "2023/24")
        self.assertEqual(result["estimate_eur"], "20.00")
        self.assertEqual(result["reference_eur"], "26.00")
        self.assertEqual(result["reference_valuation_date"], "2024-06-01")
        self.assertEqual(result["signed_gap_eur"], "-6.00")

    def test_positive_gap_and_cents_are_exact(self):
        self.records[0]["reference_eur"] = "0.10"
        self.records[1]["reference_eur"] = "0.20"
        self.records[-2]["reference_eur"] = "0.12"
        self.assertEqual(lookup(self.rows(), "e1", "2023/24")["signed_gap_eur"], "0.03")

    def test_zero_goals_and_assists_are_values(self):
        row = self.rows()[0]
        self.assertEqual((row.goals, row.assists), (0, 0))
        self.assertIsInstance(row.reference_eur, Decimal)
        self.assertIsInstance(row.feature_cutoff_date, date)

    def test_zero_minutes_excluded_from_fit_and_score(self):
        self.records[0]["minutes"] = "0"
        self.records[-1]["minutes"] = "0"
        rows = self.rows()
        result = evaluate(rows, "2023/24")
        self.assertEqual(result["position_medians_eur"]["Forward"], "30.00")
        self.assertEqual(result["holdout_row_count"], 2)
        self.assertEqual(result["evaluated_row_count"], 1)
        self.assertEqual(result["skipped_zero_minutes"], 1)
        self.assertEqual(lookup(rows, "e2", "2023/24")["status"], "zero_minutes")
        self.assertIsNone(lookup(rows, "e2", "2023/24")["estimate_eur"])

    def test_unseen_position_has_no_fallback(self):
        self.records[-1]["position"] = "Goalkeeper"
        rows = self.rows()
        self.assertEqual(evaluate(rows, "2023/24")["positions_without_training"], ["Goalkeeper"])
        self.assertEqual(evaluate(rows, "2023/24")["skipped_unseen_position"], 1)
        result = lookup(rows, "e2", "2023/24")
        self.assertEqual(result["status"], "no_training_for_position")
        self.assertIsNone(result["signed_gap_eur"])

    def test_unscorable_holdout_has_null_mae(self):
        for row in self.records[-2:]:
            row["position"] = "Goalkeeper"
        self.assertIsNone(evaluate(self.rows(), "2023/24")["mean_absolute_error_eur"])

    def test_duplicate_compound_key_rejected(self):
        self.records.append(dict(self.records[0], club="Other Paper Club"))
        with self.assertRaisesRegex(InputError, "Duplicate player-season key"):
            self.rows()

    def test_same_player_in_distinct_seasons_is_valid(self):
        self.records[-1]["player_id"] = "t1"
        self.assertEqual(len(self.rows()), 7)

    def test_missing_values_are_not_zero_filled(self):
        for field in FIELDS:
            with self.subTest(field=field):
                records = [dict(self.records[0], **{field: " "})]
                with self.assertRaisesRegex(InputError, "CSV line 2"):
                    load_csv(self.write(records))

    def test_reference_on_or_before_cutoff_rejected(self):
        for value in ("2022-05-20", "2022-05-19"):
            with self.subTest(value=value):
                self.records[0]["reference_valuation_date"] = value
                with self.assertRaisesRegex(InputError, "strictly after"):
                    self.rows()

    def test_schema_and_empty_file_errors(self):
        self.path.write_text("player_id,season\nt1,2021/22\n", encoding="utf-8")
        with self.assertRaisesRegex(InputError, "headers"):
            load_csv(self.path)
        self.write([])
        with self.assertRaisesRegex(InputError, "no player-season"):
            load_csv(self.path)
        self.path.write_text(",".join(FIELDS) + ",player_id\n", encoding="utf-8")
        with self.assertRaisesRegex(InputError, "headers"):
            load_csv(self.path)

    def test_invalid_values_rejected(self):
        cases = {"minutes": ("-1", "1.5"), "goals": ("-1", "NaN"), "assists": ("unknown",),
                 "reference_eur": ("NaN", "Infinity", "-1", "0", "1.001", "1e30"),
                 "feature_cutoff_date": ("2022-02-30", "20220520", "2020-05-20"),
                 "age_at_cutoff": ("101", "-1"), "season": ("2021/23", "2021"), "position": ("Winger",)}
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(InputError):
                        load_csv(self.write([dict(self.records[0], **{field: value})]))

    def test_no_eligible_training_or_holdout_is_an_error(self):
        rows = self.rows()
        with self.assertRaisesRegex(InputError, "No positive-minute training"):
            evaluate(rows, "2021/22")
        with self.assertRaisesRegex(InputError, "No holdout"):
            evaluate(rows, "2022/23")

    def test_deterministic_with_reordered_input(self):
        expected = evaluate(self.rows(), "2023/24")
        self.records.reverse()
        self.assertEqual(evaluate(self.rows(), "2023/24"), expected)

    def test_missing_lookup_cli_exit_and_stderr(self):
        path = self.write()
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["lookup", "--input", str(path), "--season", "2023/24", "--player-id", "missing"])
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("Player-season not found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
