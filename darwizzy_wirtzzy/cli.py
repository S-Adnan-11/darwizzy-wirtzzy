"""Local commands. All input comes from the specified CSV."""

import argparse
import csv
import sys

from .baseline import InputError, evaluate, load_csv, lookup
from .output import encode_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwizzy-wirtzzy", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("evaluate", "lookup"):
        command = actions.add_parser(action)
        command.add_argument("--input", required=True, help="Player-season CSV path")
        command.add_argument("--season", required=True, help="Held-out season in YYYY/YY format")
        command.add_argument("--model", choices=("median", "ridge"), default="median", help="Estimator; Ridge needs the model extra")
        command.add_argument("--format", choices=("legacy", "json"), default="legacy",
                             help="Output contract: original unversioned JSON (default), or versioned JSON")
        if action == "evaluate":
            command.add_argument("--compare", action="store_true", help="Compare median and Ridge on the same holdout keys")
        if action == "lookup":
            command.add_argument("--player-id", required=True, help="Stable fictional player ID")
    args = parser.parse_args(argv)
    if args.action == "evaluate" and args.compare and args.model != "median":
        parser.error("--compare evaluates both models; omit --model")
    try:
        rows = load_csv(args.input)
        if args.model == "ridge" or (args.action == "evaluate" and args.compare):
            from . import regression
            if args.action == "lookup":
                result = regression.lookup(rows, args.player_id, args.season)
            else:
                result = regression.compare(rows, args.season) if args.compare else regression.evaluate(rows, args.season)
        else:
            result = evaluate(rows, args.season) if args.action == "evaluate" else lookup(rows, args.player_id, args.season)
        output = encode_result(result, args.action, args.model, args.format)
    except (InputError, OSError, UnicodeError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0
