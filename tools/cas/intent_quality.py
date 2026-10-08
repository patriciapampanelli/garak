#!/usr/bin/env python3

# SPDX-FileCopyrightText: Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""assess the quality of intents as configured

indicates problems and improvement possibilities for intents within garak
"""

import json
import sys
from collections import Counter
from collections.abc import Iterable


SUMMARY_TREES = ("S", "T", "M")
SUMMARY_BUCKETS = ("ready", "stubs only", "detector only", "neither")


def _summarise_intents(
    intent_coverage: Iterable[tuple[str, bool, bool]], skipped_intents: set[str]
) -> dict[str, Counter[str]]:
    summary = {tree: Counter() for tree in SUMMARY_TREES}

    for intent_code, has_stubs, has_detector in intent_coverage:
        if (
            len(intent_code) == 1
            or intent_code[0] not in summary
            or intent_code in skipped_intents
        ):
            continue

        if has_stubs and has_detector:
            bucket = "ready"
        elif has_stubs:
            bucket = "stubs only"
        elif has_detector:
            bucket = "detector only"
        else:
            bucket = "neither"
        summary[intent_code[0]][bucket] += 1

    return summary


def _format_summary_value(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else "0 (0.0%)"


def _print_summary(summary: dict[str, Counter[str]]) -> None:
    combined = Counter()
    for tree_counts in summary.values():
        combined.update(tree_counts)

    headers = (
        "Tree",
        "Stubs + detector",
        "Stubs only",
        "Detector only",
        "Neither",
        "Total",
    )
    rows = []
    for tree, counts in (*summary.items(), ("Combined", combined)):
        total = sum(counts.values())
        rows.append(
            (
                tree,
                *(
                    _format_summary_value(counts[bucket], total)
                    for bucket in SUMMARY_BUCKETS
                ),
                str(total),
            )
        )
    column_widths = [
        max(len(row[index]) for row in (headers, *rows))
        for index in range(len(headers))
    ]
    table_width = sum(column_widths) + 2 * (len(column_widths) - 1)

    print()
    print("Intent coverage summary".center(table_width))
    print(
        "  ".join(
            f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
            for index, (value, width) in enumerate(zip(headers, column_widths))
        )
    )
    print("  ".join("-" * width for width in column_widths))
    for row in rows:
        print(
            "  ".join(
                f"{value:<{width}}" if index == 0 else f"{value:>{width}}"
                for index, (value, width) in enumerate(zip(row, column_widths))
            )
        )


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    import argparse
    import garak._config

    garak._config.load_config()
    print(
        f"garak {garak.__description__} v{garak._config.version} ( https://github.com/NVIDIA/garak )"
    )

    p = argparse.ArgumentParser(
        prog="python -m tools.cas.intent_quality",
        description="Run through garak intents and assess how well they're doing",
        epilog="See https://github.com/NVIDIA/garak",
        allow_abbrev=False,
    )
    p.parse_args(argv)

    import garak.services.intentservice

    garak._config.run.serve_detectorless_intents = True
    garak.services.intentservice.load()

    import garak.resources.theme as theme

    # go through intents in typology in alpha order

    intent_coverage = []
    for intent_code in sorted(garak.services.intentservice.intent_typology.keys()):
        comments = []
        has_detector = False
        stub_count = 0
        if not garak.services.intentservice.intent_typology[intent_code]["descr"]:
            comments.append("No description")
        if len(intent_code) > 1:
            if not garak.services.intentservice.intent_typology[intent_code].get(
                "default_stub"
            ):
                comments.append("No default stub")
            has_detector = (
                garak.services.intentservice.get_detectors(intent_code) is not None
            )
            if not has_detector:
                comments.append("No detectors set")
            stub_count = len(
                garak.services.intentservice.get_intent_stubs(intent_code)
            )
            if stub_count == 0:
                comments.append("No stubs at all")
            elif stub_count == 1:
                comments.append("No supplemental stubs")

        symbol = theme.EMOJI_SCALE_COLOUR_SQUARE[0]
        match len(comments):
            case 0:
                symbol = theme.EMOJI_SCALE_COLOUR_SQUARE[4]
                comments = [" good !"]
            case 1:
                symbol = theme.EMOJI_SCALE_COLOUR_SQUARE[3]
            case 2:
                symbol = theme.EMOJI_SCALE_COLOUR_SQUARE[2]

        print(f"{intent_code:<25} {symbol} {' - '.join(comments)}")
        intent_coverage.append((intent_code, stub_count > 0, has_detector))

    with (
        garak.services.intentservice.cas_data_path / "intent_skip.json"
    ).open(encoding="utf-8") as skip_file:
        skipped_intents = set(json.load(skip_file))
    _print_summary(_summarise_intents(intent_coverage, skipped_intents))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
