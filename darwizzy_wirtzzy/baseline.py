"""Strict CSV input and a chronological median-by-position baseline."""

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re
from statistics import median

FIELDS = (
    "player_id", "player_name", "season", "club", "position", "age_at_cutoff",
    "minutes", "goals", "assists", "feature_cutoff_date", "reference_valuation_date",
    "reference_eur",
)
POSITIONS = {"Goalkeeper", "Defender", "Midfielder", "Forward"}


class InputError(ValueError):
    """A dataset or requested experiment cannot be used safely."""


def season_start(value: str) -> int:
    match = re.fullmatch(r"([0-9]{4})/([0-9]{2})", value)
    if not match or int(match[2]) != (int(match[1]) + 1) % 100:
        raise InputError(f"Invalid season {value!r}; expected YYYY/YY, such as 2025/26")
    year = int(match[1])
    if not 1 <= year <= 9998:
        raise InputError("Season year must be between 0001 and 9998")
    return year


@dataclass(frozen=True)
class PlayerSeason:
    player_id: str
    player_name: str
    season: str
    club: str
    position: str
    age_at_cutoff: int
    minutes: int
    goals: int
    assists: int
    feature_cutoff_date: date
    reference_valuation_date: date
    reference_eur: Decimal


def _integer(value: str, field: str) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise InputError(f"{field} must be a nonnegative integer")
    return int(value)


def _date(value: str, field: str) -> date:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise InputError(f"{field} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise InputError(f"{field} is not a valid date") from error


def load_csv(path: str | Path) -> list[PlayerSeason]:
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(FIELDS):
            raise InputError("CSV headers must contain each required field exactly once: " + ", ".join(FIELDS))
        for raw in reader:
            try:
                if None in raw or any(value is None or not value.strip() for value in raw.values()):
                    raise InputError("Every required field must have a value and each row must match the header")
                row = {field: raw[field].strip() for field in FIELDS}
                year = season_start(row["season"])
                key = (row["player_id"], row["season"])
                if key in seen:
                    raise InputError(f"Duplicate player-season key: {key[0]} / {key[1]}")
                if row["position"] not in POSITIONS:
                    raise InputError("position must be Goalkeeper, Defender, Midfielder or Forward")
                counts = {field: _integer(row[field], field) for field in ("age_at_cutoff", "minutes", "goals", "assists")}
                if counts["age_at_cutoff"] > 100:
                    raise InputError("age_at_cutoff must be between 0 and 100")
                cutoff = _date(row["feature_cutoff_date"], "feature_cutoff_date")
                reference_date = _date(row["reference_valuation_date"], "reference_valuation_date")
                if not date(year, 7, 1) <= cutoff <= date(year + 1, 6, 30):
                    raise InputError("feature_cutoff_date must fall within the July–June season")
                if reference_date <= cutoff:
                    raise InputError("reference_valuation_date must be strictly after feature_cutoff_date")
                try:
                    amount = Decimal(row["reference_eur"])
                except InvalidOperation as error:
                    raise InputError("reference_eur must be a positive finite EUR amount") from error
                if not amount.is_finite() or amount <= 0 or amount > Decimal("1000000000000") or amount != amount.quantize(Decimal("0.01")):
                    raise InputError("reference_eur must be positive, at most 1 trillion, and have at most two decimal places")
                rows.append(PlayerSeason(
                    row["player_id"], row["player_name"], row["season"], row["club"], row["position"],
                    **counts, feature_cutoff_date=cutoff, reference_valuation_date=reference_date, reference_eur=amount,
                ))
                seen.add(key)
            except InputError as error:
                raise InputError(f"CSV line {reader.line_num}: {error}") from error
    if not rows:
        raise InputError("CSV contains no player-season rows")
    return sorted(rows, key=lambda row: (row.season, row.player_id))


def money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")


def fit(rows: list[PlayerSeason], evaluation_season: str) -> tuple[dict[str, Decimal], list[PlayerSeason], list[PlayerSeason]]:
    year = season_start(evaluation_season)
    holdout = [row for row in rows if row.season == evaluation_season]
    if not holdout:
        raise InputError(f"No holdout rows for season {evaluation_season}")
    available_at = min(row.feature_cutoff_date for row in holdout)
    training = [row for row in rows if season_start(row.season) < year and row.minutes > 0
                and row.reference_valuation_date <= available_at]
    if not training:
        raise InputError("No positive-minute training rows from earlier seasons with references available at the evaluation cutoff")
    estimates = {position: median([row.reference_eur for row in training if row.position == position])
                 for position in sorted({row.position for row in training})}
    return estimates, training, holdout


def evaluate(rows: list[PlayerSeason], evaluation_season: str) -> dict:
    estimates, training, holdout = fit(rows, evaluation_season)
    scored = [row for row in holdout if row.minutes > 0 and row.position in estimates]
    errors = [abs(estimates[row.position] - row.reference_eur) for row in scored]
    return {
        "evaluation_season": evaluation_season,
        "training_seasons": sorted({row.season for row in training}),
        "training_row_count": len(training),
        "holdout_row_count": len(holdout),
        "evaluated_row_count": len(scored),
        "skipped_zero_minutes": sum(row.minutes == 0 for row in holdout),
        "skipped_unseen_position": sum(row.minutes > 0 and row.position not in estimates for row in holdout),
        "positions_without_training": sorted({row.position for row in holdout} - estimates.keys()),
        "position_medians_eur": {position: money(value) for position, value in estimates.items()},
        "mean_absolute_error_eur": money(sum(errors) / len(errors)) if errors else None,
    }


def lookup(rows: list[PlayerSeason], player_id: str, season: str) -> dict:
    season_start(season)
    row = next((row for row in rows if row.player_id == player_id and row.season == season), None)
    if row is None:
        raise InputError(f"Player-season not found: {player_id} / {season}")
    estimates, _, _ = fit(rows, season)
    estimate = estimates.get(row.position) if row.minutes > 0 else None
    status = "zero_minutes" if row.minutes == 0 else ("ok" if estimate is not None else "no_training_for_position")
    return {
        "player_id": row.player_id, "player_name": row.player_name, "season": row.season,
        "club": row.club, "position": row.position, "status": status,
        "estimate_eur": money(estimate) if estimate is not None else None,
        "reference_eur": money(row.reference_eur),
        "reference_valuation_date": row.reference_valuation_date.isoformat(),
        "signed_gap_eur": money(estimate - row.reference_eur) if estimate is not None else None,
    }
