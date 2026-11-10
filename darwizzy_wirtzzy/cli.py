"""Local commands. All input comes from the specified CSV."""

import argparse
import csv
import json
import sys

from .baseline import InputError, evaluate, load_csv, lookup


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="darwizzy-wirtzzy", description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for action in ("evaluate", "lookup"):
        command = actions.add_parser(action)
        command.add_argument("--input", required=True, help="Player-season CSV path")
        command.add_argument("--season", required=True, help="Held-out season in YYYY/YY format")
        if action == "lookup":
            command.add_argument("--player-id", required=True, help="Stable fictional player ID")
    args = parser.parse_args(argv)
    try:
        rows = load_csv(args.input)
        result = evaluate(rows, args.season) if args.action == "evaluate" else lookup(rows, args.player_id, args.season)
    except (InputError, OSError, UnicodeError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0
