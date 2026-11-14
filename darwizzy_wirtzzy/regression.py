"""Optional historical Ridge comparison; estimators stay in memory."""

from dataclasses import dataclass
from decimal import Decimal, localcontext
import math

from .baseline import InputError, PlayerSeason, fit as fit_median, money, season_start, split

FEATURES = ("age_at_cutoff", "minutes", "goals", "assists", "position")
ALPHA = 1.0


def _pipeline():
    # Keep the default median workflow independent of the model extra.
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as error:
        raise InputError('Ridge requires the optional model dependencies; install with python -m pip install ".[model]"') from error
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", StandardScaler(), [0, 1, 2, 3]),
            ("position", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [4]),
        ])),
        ("ridge", Ridge(alpha=ALPHA, solver="svd")),
    ])


def feature_rows(rows: list[PlayerSeason]) -> list[list]:
    """Only pre-cutoff age, season totals and broad position enter the model."""
    result = []
    for row in rows:
        try:
            numeric = [float(getattr(row, field)) for field in FEATURES[:-1]]
        except OverflowError as error:
            raise InputError("Ridge features exceed the finite numeric range") from error
        if not all(math.isfinite(value) for value in numeric):
            raise InputError("Ridge features must be finite")
        result.append([*numeric, row.position])
    return result


@dataclass
class RidgeFit:
    pipeline: object
    positions: set[str]

    def predict(self, rows: list[PlayerSeason]) -> dict[tuple[str, str], Decimal]:
        eligible = [row for row in rows if row.minutes > 0 and row.position in self.positions]
        if not eligible:
            return {}
        try:
            logs = self.pipeline.predict(feature_rows(eligible))
        except InputError:
            raise
        except (ValueError, FloatingPointError) as error:
            raise InputError("Ridge could not produce finite estimates from these features") from error
        estimates = {}
        for row, log_value in zip(eligible, logs):
            try:
                amount = math.exp(float(log_value))
            except OverflowError as error:
                raise InputError("Ridge produced a nonfinite EUR estimate") from error
            if not math.isfinite(amount) or amount <= 0:
                raise InputError("Ridge produced a nonfinite or nonpositive EUR estimate")
            estimate = Decimal(str(amount))
            # Values that cannot be represented as positive cents are unusable.
            if estimate < Decimal("0.005"):
                raise InputError("Ridge estimate is outside the reportable positive EUR range")
            estimates[(row.player_id, row.season)] = estimate
        return estimates


def fit_training(training: list[PlayerSeason]) -> RidgeFit:
    if len(training) < 2:
        raise InputError("Ridge requires at least two eligible historical training rows")
    pipeline = _pipeline()
    targets = [math.log(float(row.reference_eur)) for row in training]
    try:
        pipeline.fit(feature_rows(training), targets)
    except InputError:
        raise
    except (ValueError, FloatingPointError) as error:
        raise InputError("Ridge could not fit these historical features") from error
    return RidgeFit(pipeline, {row.position for row in training})


def _coverage(training: list[PlayerSeason], holdout: list[PlayerSeason], season: str) -> dict:
    positions = {row.position for row in training}
    return {
        "evaluation_season": season,
        "training_seasons": sorted({row.season for row in training}),
        "training_row_count": len(training),
        "holdout_row_count": len(holdout),
        "skipped_zero_minutes": sum(row.minutes == 0 for row in holdout),
        "skipped_unseen_position": sum(row.minutes > 0 and row.position not in positions for row in holdout),
        "positions_without_training": sorted({row.position for row in holdout} - positions),
    }


def _mae(scored: list[PlayerSeason], estimates: dict[tuple[str, str], Decimal]) -> str | None:
    if not scored:
        return None
    with localcontext() as context:
        context.prec = 330  # Accommodate the full finite float range at EUR cents.
        errors = [abs(estimates[(row.player_id, row.season)] - row.reference_eur) for row in scored]
        return money(sum(errors) / len(errors))


def evaluate(rows: list[PlayerSeason], season: str) -> dict:
    training, holdout = split(rows, season)
    model = fit_training(training)
    estimates = model.predict(holdout)
    scored = [row for row in holdout if (row.player_id, row.season) in estimates]
    return {
        **_coverage(training, holdout, season),
        "model": "ridge", "ridge_alpha": ALPHA,
        "features": list(FEATURES), "target": "log_reference_eur",
        "evaluated_row_count": len(scored),
        "mean_absolute_error_eur": _mae(scored, estimates),
    }


def compare(rows: list[PlayerSeason], season: str) -> dict:
    medians, training, holdout = fit_median(rows, season)
    ridge = fit_training(training)
    ridge_estimates = ridge.predict(holdout)
    median_estimates = {(row.player_id, row.season): medians[row.position] for row in holdout
                        if row.minutes > 0 and row.position in medians}
    paired = [row for row in holdout if (row.player_id, row.season) in ridge_estimates
              and (row.player_id, row.season) in median_estimates]
    return {
        **_coverage(training, holdout, season),
        "model": "comparison",
        "paired_row_count": len(paired),
        "paired_player_seasons": [{"player_id": row.player_id, "season": row.season} for row in paired],
        "skipped_reasons": {
            "zero_minutes": sum(row.minutes == 0 for row in holdout),
            "no_training_for_position": sum(row.minutes > 0 and row.position not in medians for row in holdout),
        },
        "models": {
            "median": {"eligible_row_count": len(median_estimates), "paired_row_count": len(paired),
                       "mean_absolute_error_eur": _mae(paired, median_estimates)},
            "ridge": {"eligible_row_count": len(ridge_estimates), "paired_row_count": len(paired),
                      "mean_absolute_error_eur": _mae(paired, ridge_estimates), "alpha": ALPHA},
        },
    }


def lookup(rows: list[PlayerSeason], player_id: str, season: str) -> dict:
    season_start(season)
    row = next((row for row in rows if row.player_id == player_id and row.season == season), None)
    if row is None:
        raise InputError(f"Player-season not found: {player_id} / {season}")
    training, _ = split(rows, season)
    model = fit_training(training)
    estimate = model.predict([row]).get((row.player_id, row.season))
    status = "zero_minutes" if row.minutes == 0 else ("ok" if estimate is not None else "no_training_for_position")
    with localcontext() as context:
        context.prec = 330
        estimate_eur = money(estimate) if estimate is not None else None
        gap_eur = money(estimate - row.reference_eur) if estimate is not None else None
    return {
        "model": "ridge", "player_id": row.player_id, "player_name": row.player_name,
        "season": row.season, "club": row.club, "position": row.position, "status": status,
        "estimate_eur": estimate_eur,
        "reference_eur": money(row.reference_eur),
        "reference_valuation_date": row.reference_valuation_date.isoformat(),
        "signed_gap_eur": gap_eur,
    }
