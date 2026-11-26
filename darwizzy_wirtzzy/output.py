"""CLI output contracts over the existing computed results."""

import json

from .baseline import InputError

METRICS = ("mean_absolute_error_eur", "median_absolute_error_eur", "mean_signed_error_eur")


def legacy_result(result: dict) -> dict:
    """Keep the original unversioned CLI shape for existing invocations."""
    value = {key: item for key, item in result.items() if key not in METRICS[1:]}
    if "models" in value:
        value["models"] = {
            name: {key: item for key, item in model.items() if key not in METRICS[1:]}
            for name, model in value["models"].items()
        }
    return value


def structured_result(result: dict, command: str, model: str) -> dict:
    value = {"schema_version": 1, "command": command, "currency": "EUR", **result}
    value["model"] = result.get("model", model)
    if command == "evaluate":
        scored = result.get("paired_row_count", result.get("evaluated_row_count"))
        value["eligible_evaluation_row_count"] = (
            result["holdout_row_count"] - result["skipped_zero_minutes"] - result["skipped_unseen_position"]
        )
        value["scored_row_count"] = scored
        if "models" in result:
            value["models"] = {
                name: {**details, "scored_row_count": details["paired_row_count"]}
                for name, details in result["models"].items()
            }
        else:
            value["models"] = {
                value["model"]: {
                    "eligible_row_count": scored,
                    "scored_row_count": scored,
                    **{name: value.pop(name) for name in METRICS},
                }
            }
    return value


def encode_result(result: dict, command: str, model: str, output_format: str) -> str:
    """Serialize completely before writing stdout so encoding failures are atomic."""
    value = structured_result(result, command, model) if output_format == "json" else legacy_result(result)
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=output_format == "json",
                          allow_nan=False, indent=None if output_format == "json" else 2)
    except (TypeError, ValueError) as error:
        raise InputError("Cannot serialize the result as finite JSON") from error
