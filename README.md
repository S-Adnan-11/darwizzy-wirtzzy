# darwizzy-wirtzzy

A small offline workflow for estimating historical football player values. Fit a
median reference value for each broad position on earlier seasons, then evaluate
it on a later season and look up an individual player-season. An optional Ridge
model adds age and season totals for comparison; the position median stays the
default.

**Every player, club and valuation in the example is fictional.** The example
demonstrates the workflow; its errors say nothing about real football accuracy.
The default workflow uses Python's standard library. Model fitting and lookups
make no network requests.

## Run locally

Use Python 3.11 or later from the repository root. No install is needed:

```powershell
python -m darwizzy_wirtzzy evaluate --input examples/fictional_player_seasons.csv --season 2025/26
python -m darwizzy_wirtzzy lookup --input examples/fictional_player_seasons.csv --season 2025/26 --player-id fictional-001
python -m unittest discover -s tests -v
```

The example fits five rows from 2023/24. Its 2025/26 holdout has five rows, of which
three can be scored; one has zero minutes and one has no historical position
baseline. Mean absolute error is EUR 1,666,666.67 across those three scored rows.
The forward, defender and goalkeeper medians are EUR 15,000,000, EUR 6,000,000 and
EUR 5,000,000 respectively.

The lookup for `fictional-001` returns an estimate of EUR 15,000,000, a reference of
EUR 18,000,000 dated 2026-06-10, and a signed gap of EUR -3,000,000. The signed gap
is **estimate minus reference**. Money fields in JSON are decimal strings in EUR,
rounded to cents using half-up rounding.

A missing player-season prints a useful error to stderr and exits with code 2:

```powershell
python -m darwizzy_wirtzzy lookup --input examples/fictional_player_seasons.csv --season 2025/26 --player-id missing
```

## Versioned JSON output

The existing default output is unversioned, pretty-printed JSON. It remains
unchanged; `--format legacy` selects it explicitly. Use `--format json` for the
versioned contract on either command:

```powershell
python -m darwizzy_wirtzzy evaluate --input examples/fictional_player_seasons.csv --season 2025/26 --format json
python -m darwizzy_wirtzzy lookup --input examples/fictional_player_seasons.csv --season 2025/26 --player-id fictional-001 --format json
```

Each successful invocation writes one compact JSON object followed by a newline.
All objects have `schema_version: 1`, `command` (`evaluate` or `lookup`),
`currency: "EUR"`, and `model` (`median`, `ridge`, or `comparison`). Keys and
model ordering are deterministic. Unicode and special characters are JSON-escaped
and decode to the original text. NaN and Infinity are rejected. Errors keep
stdout empty, write the actionable message to stderr and exit with code 2;
`--help` retains the usual CLI behavior.

Lookup adds the same identity and status fields as the existing lookup, with ISO
reference dates and nullable estimates/gaps. The fictional lookup above returns:

```json
{
  "schema_version": 1,
  "command": "lookup",
  "currency": "EUR",
  "model": "median",
  "player_id": "fictional-001",
  "player_name": "Arlo Vale",
  "season": "2025/26",
  "club": "Cloudbridge FC",
  "position": "Forward",
  "status": "ok",
  "estimate_eur": "15000000.00",
  "reference_eur": "18000000.00",
  "reference_valuation_date": "2026-06-10",
  "signed_gap_eur": "-3000000.00"
}
```

All EUR amounts are decimal strings, rounded to cents using half-up rounding.
Signed gap and signed error mean **estimate minus reference**. Aggregate metrics
use unrounded predictions and errors before the final formatting.

Evaluation retains its season, training seasons, coverage and skipped fields,
and adds the following fields. Counts describe only the validated CSV passed to
the command; they do not count players absent from that input.

| Field | Meaning |
| --- | --- |
| `evaluation_season` | Requested held-out season |
| `training_row_count` | Historical rows actually used after the existing positive-minute and label-availability rules |
| `holdout_row_count` | All validated rows for the requested season, including zero minutes and unsupported roles |
| `eligible_evaluation_row_count` | Positive-minute holdout rows with a position supported by training; holdout minus both skipped counts |
| `scored_row_count` | Rows actually scored; comparison uses only the common paired keys |
| `skipped_zero_minutes`, `skipped_unseen_position`, `positions_without_training` | Existing coverage diagnostics; no replacement estimate is invented |
| `models.<model>.eligible_row_count` | Predictions available from that model under its existing eligibility rules |
| `models.<model>.scored_row_count` | Predictions included in that model's reported metrics |
| `models.<model>.mean_absolute_error_eur` | Mean absolute error, or null when no rows are scored |
| `models.<model>.median_absolute_error_eur` | Median absolute error, or null when no rows are scored |
| `models.<model>.mean_signed_error_eur` | Mean signed error, or null when no rows are scored |

For a single model, `evaluated_row_count` is also retained and `models` contains
only that model. Median evaluation retains `position_medians_eur`; Ridge retains
`ridge_alpha`, `features` and `target`. These options require the existing model
extra and use its unchanged predictions:

```powershell
python -m darwizzy_wirtzzy evaluate --input examples/fictional_regression_seasons.csv --season 2025/26 --model ridge --format json
python -m darwizzy_wirtzzy evaluate --input examples/fictional_regression_seasons.csv --season 2025/26 --compare --format json
python -m darwizzy_wirtzzy lookup --input examples/fictional_regression_seasons.csv --season 2025/26 --player-id fictional-r08 --model ridge --format json
```

Comparison sets `model` to `comparison`, retains `paired_row_count`,
`paired_player_seasons` (objects with `player_id` and `season`) and
`skipped_reasons`, and reports `models.median` before `models.ridge`. Both models
score exactly those paired keys. Each model also retains `paired_row_count`,
and Ridge retains `alpha`. A model's available prediction count can exceed the
paired count; its comparison metrics still use only the paired rows.

Parse the dependency-free fictional evaluation directly in PowerShell:

```powershell
$result = python -m darwizzy_wirtzzy evaluate --input examples/fictional_player_seasons.csv --season 2025/26 --format json | ConvertFrom-Json
$result.schema_version  # 1
$result.models.median.mean_absolute_error_eur  # 1666666.67
```

## CSV contract

The CSV header must contain exactly these fields. Each field is required.

| Field | Meaning |
| --- | --- |
| `player_id`, `season` | Unique compound key; season uses `YYYY/YY` |
| `player_name`, `club` | Display name and club at the feature cutoff |
| `position` | `Goalkeeper`, `Defender`, `Midfielder` or `Forward` |
| `age_at_cutoff` | Whole years at the feature cutoff, between 0 and 100 |
| `minutes`, `goals`, `assists` | Nonnegative whole season totals at the cutoff |
| `feature_cutoff_date` | `YYYY-MM-DD`, within the season's July–June interval |
| `reference_valuation_date` | `YYYY-MM-DD`, strictly after the feature cutoff |
| `reference_eur` | Positive EUR amount, at most two decimal places and at most 1 trillion |

Blank or missing fields, extra cells, duplicate keys, invalid dates and invalid
numbers reject the whole input with the CSV line number. Zero goals and assists
are valid values. Zero-minute rows remain visible in holdout counts and lookups
but do not train or score. Lookups for these return `zero_minutes` and null
estimate/gap. A position without training examples returns
`no_training_for_position`; evaluation reports the missing position and skipped
count. If no rows can be scored, MAE is null. No holdout rows or no eligible
historical training rows is an error.

## Chronology and baseline

Training uses only seasons strictly earlier than the requested season. Training
references must also be available by the earliest feature cutoff in that holdout,
so a late historical label cannot leak into the experiment. The requested season
is held out in full; later seasons cannot affect fitting. Outputs are deterministic
and have no timestamps. Age and season totals are validated for future models;
this baseline uses only position and historical reference amounts.

Reference valuations are estimates, not observed sale prices. Real-data source
selection and provenance checks come before fitting real rows or publishing an app.
Keep credentials, provider responses, private audit reports and account records
outside this repository. The ignore rules are a fallback, not a storage workflow.

## Optional Ridge model

The tested model setup is 64-bit Windows with CPython 3.15. Create a separate
environment so the dependency-free setup stays available. These commands install
the optional extra with the tested binary dependency versions:

```powershell
py -3.15 -m venv ..\darwizzy-model-venv
..\darwizzy-model-venv\Scripts\python.exe -m pip install --only-binary=:all: --constraint requirements-model-win-py315.txt ".[model]"
..\darwizzy-model-venv\Scripts\python.exe -m unittest discover -s tests -v
```

`requirements-model-win-py315.txt` pins the tested Windows/Python combination.
Other supported interpreters can install `.[model]` in their own environment
without that platform-specific constraint file. The optional dependency is
scikit-learn; requesting Ridge without it prints an installation command and
exits with code 2.

Use the larger fictional example for Ridge, paired evaluation and lookup:

```powershell
..\darwizzy-model-venv\Scripts\python.exe -m darwizzy_wirtzzy evaluate --input examples/fictional_regression_seasons.csv --season 2025/26 --model ridge
..\darwizzy-model-venv\Scripts\python.exe -m darwizzy_wirtzzy evaluate --input examples/fictional_regression_seasons.csv --season 2025/26 --compare
..\darwizzy-model-venv\Scripts\python.exe -m darwizzy_wirtzzy lookup --input examples/fictional_regression_seasons.csv --season 2025/26 --player-id fictional-r08 --model ridge
```

The example has 12 eligible training rows from 2023/24 and 2024/25, followed by
five 2025/26 holdout rows. Four can be scored; the fifth has zero minutes. Both
models use the same four player-season keys. In the tested setup, paired MAE is
EUR 5,250,000.00 for the median and EUR 3,120,508.18 for Ridge. The lookup returns
EUR 33,531,090.00 against a EUR 36,000,000.00 reference dated 2026-06-10, a signed
gap of EUR -2,468,910.00. These are fictional examples, not accuracy claims.

Ridge uses age, minutes, goals, assists and broad position. A single pipeline
scales numeric features and one-hot encodes position using training rows only,
then fits `Ridge(alpha=1)` to the log of positive EUR references. Predictions are
exponentiated and checked for finite, positive amounts representable in cents.
IDs, names, clubs, dates and reference amounts never enter the feature matrix.
The chronological split runs before preprocessing or model fitting, and holds
out the requested season in full. The implementation follows scikit-learn's
[pipeline and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Comparison reports paired keys, coverage, skipped reasons and MAE in EUR for
each model. Zero-minute rows and positions without historical training are
skipped by both models; lookups retain `zero_minutes` and
`no_training_for_position`. A comparison with no eligible pair reports zero
paired rows and null MAEs. Ridge requires at least two eligible historical rows
to fit, which is only a computational minimum. It does not establish sufficient
data for a useful real-world estimate. The initial alpha is fixed, and the
holdout is for evaluation, not tuning. Estimators stay in memory; no model files
or data exports are written.
