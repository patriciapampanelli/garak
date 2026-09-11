#!/usr/bin/env python3
"""Update technique-to-probe coverage columns in an exported TSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


TECHNIQUE_COLUMNS = (
    "Technique Root",
    "Technique Family",
    "Technique",
)
COVERAGE_COLUMNS = (
    "IntentProbe Count",
    "IntentProbes",
    "Non-IntentProbe Count",
    "Non-IntentProbes",
)
EMPTY_PROBE_LIST = "—"
TECHNIQUE_TAG_PREFIX = "demon:"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
PROBE_LINE_RE = re.compile(
    r"^probes:\s+(?P<name>[A-Za-z_]\w*\.[A-Za-z_]\w*)(?:\s|$)"
)
METADATA_MARKER = "__GARAK_TECHNIQUE_COVERAGE__="

TechniqueKey = tuple[str, str, str]
ProbeGroups = dict[str, set[str]]
Coverage = dict[TechniqueKey, ProbeGroups]

_METADATA_SCRIPT = f"""
import importlib
import json
import sys

from garak import _plugins
from garak.probes.base import IntentProbe

metadata = {{}}
for plugin_name in json.load(sys.stdin):
    tags = [
        tag
        for tag in (_plugins.plugin_info(plugin_name).get("tags") or [])
        if isinstance(tag, str) and tag.startswith({TECHNIQUE_TAG_PREFIX!r})
    ]
    if not tags:
        continue

    module_name, class_name = plugin_name.rsplit(".", 1)
    module = importlib.import_module(f"garak.{{module_name}}")
    probe_class = getattr(module, class_name)
    if not isinstance(probe_class, type):
        raise TypeError(f"{{plugin_name}} does not resolve to a class")

    metadata[plugin_name] = {{
        "is_intent_probe": issubclass(probe_class, IntentProbe),
        "tags": tags,
    }}

print({METADATA_MARKER!r} + json.dumps(metadata, sort_keys=True))
"""


def _run_command(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    input_text: str | None = None,
) -> str:
    try:
        result = subprocess.run(
            command,
            input=input_text,
            capture_output=True,
            check=False,
            encoding="utf-8",
            env=environment,
        )
    except OSError as error:
        raise RuntimeError(f"could not run {' '.join(command)}: {error}") from error

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"{' '.join(command)} exited with {result.returncode}: {details}"
        )
    return result.stdout


def _parse_probe_listing(output: str) -> list[str]:
    probe_names = set()
    for raw_line in output.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        match = PROBE_LINE_RE.match(line)
        if match:
            probe_names.add(f"probes.{match.group('name')}")

    if not probe_names:
        raise ValueError("garak --list_probes returned no probe classes")
    return sorted(probe_names)


def _parse_metadata(output: str) -> dict[str, dict[str, Any]]:
    for line in reversed(output.splitlines()):
        if line.startswith(METADATA_MARKER):
            payload = line.removeprefix(METADATA_MARKER)
            try:
                metadata = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError("garak returned malformed probe metadata") from error
            if not isinstance(metadata, dict):
                raise ValueError("garak returned invalid probe metadata")
            return metadata
    raise ValueError("garak did not return probe metadata")


def _collect_probe_metadata(python_executable: str) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="garak-coverage-") as cache_directory:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["XDG_CACHE_HOME"] = cache_directory

        listing = _run_command(
            [python_executable, "-m", "garak", "--list_probes"],
            environment=environment,
        )
        probe_names = _parse_probe_listing(listing)
        metadata_output = _run_command(
            [python_executable, "-c", _METADATA_SCRIPT],
            environment=environment,
            input_text=json.dumps(probe_names),
        )
    return _parse_metadata(metadata_output)


def _technique_key(tag: str) -> TechniqueKey:
    parts = tag.split(":", 3)
    if len(parts) != 4 or parts[0] != TECHNIQUE_TAG_PREFIX.rstrip(":"):
        raise ValueError(f"invalid technique tag: {tag}")
    _, root, family, technique = parts
    if not all((root, family, technique)):
        raise ValueError(f"invalid technique tag: {tag}")
    return root, family, technique


def _build_coverage(metadata: Mapping[str, Mapping[str, Any]]) -> Coverage:
    coverage: defaultdict[TechniqueKey, ProbeGroups] = defaultdict(
        lambda: {"intent": set(), "non_intent": set()}
    )
    for full_name, probe_metadata in metadata.items():
        short_name = full_name.removeprefix("probes.")
        group = "intent" if probe_metadata["is_intent_probe"] else "non_intent"
        for tag in probe_metadata["tags"]:
            coverage[_technique_key(tag)][group].add(short_name)
    return dict(coverage)


def _find_header(rows: Sequence[Sequence[str]]) -> tuple[int, dict[str, int]]:
    required_columns = TECHNIQUE_COLUMNS + COVERAGE_COLUMNS
    for row_number, row in enumerate(rows):
        stripped_row = [cell.strip() for cell in row]
        if not all(column in stripped_row for column in required_columns):
            continue

        indexes = {}
        for column in required_columns:
            if stripped_row.count(column) != 1:
                raise ValueError(f"column {column!r} must occur exactly once")
            indexes[column] = stripped_row.index(column)
        return row_number, indexes
    raise ValueError(
        "could not find a header row containing the technique and coverage columns"
    )


def _format_probe_names(probe_names: set[str]) -> str:
    return "; ".join(sorted(probe_names)) if probe_names else EMPTY_PROBE_LIST


def _update_rows(
    rows: list[list[str]], coverage: Coverage
) -> tuple[int, list[TechniqueKey]]:
    header_number, indexes = _find_header(rows)
    maximum_index = max(indexes.values())
    seen_keys = set()
    updated_rows = 0

    for row_number, row in enumerate(rows[header_number + 1 :], header_number + 2):
        if len(row) <= max(indexes[column] for column in TECHNIQUE_COLUMNS):
            if any(cell.strip() for cell in row):
                raise ValueError(f"row {row_number} does not contain a technique key")
            continue

        key: TechniqueKey = (
            row[indexes["Technique Root"]].strip(),
            row[indexes["Technique Family"]].strip(),
            row[indexes["Technique"]].strip(),
        )
        if not any(key):
            continue
        if not all(key):
            raise ValueError(f"row {row_number} contains an incomplete technique key")
        if key in seen_keys:
            raise ValueError(f"row {row_number} duplicates technique {':'.join(key)}")
        seen_keys.add(key)

        if len(row) <= maximum_index:
            row.extend([""] * (maximum_index + 1 - len(row)))

        groups = coverage.get(key)
        intent_probes = groups["intent"] if groups is not None else set()
        non_intent_probes = groups["non_intent"] if groups is not None else set()
        row[indexes["IntentProbe Count"]] = str(len(intent_probes))
        row[indexes["IntentProbes"]] = _format_probe_names(intent_probes)
        row[indexes["Non-IntentProbe Count"]] = str(len(non_intent_probes))
        row[indexes["Non-IntentProbes"]] = _format_probe_names(non_intent_probes)
        updated_rows += 1

    unmatched_keys = sorted(set(coverage) - seen_keys)
    return updated_rows, unmatched_keys


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.reader(input_file, dialect="excel-tab"))


def _write_tsv(path: Path, rows: Sequence[Sequence[str]]) -> None:
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_name = output_file.name
            writer = csv.writer(
                output_file,
                dialect="excel-tab",
                lineterminator="\n",
            )
            writer.writerows(rows)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _output_path(
    input_path: Path, requested_output: Path | None, in_place: bool
) -> Path:
    if in_place:
        return input_path
    if requested_output is not None:
        return requested_output
    suffix = input_path.suffix or ".tsv"
    return input_path.with_name(f"{input_path.stem}.updated{suffix}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Update IntentProbe and non-IntentProbe coverage in a technique TSV "
            "using the installed garak probe inventory."
        )
    )
    parser.add_argument("input", type=Path, help="TSV exported from Google Sheets")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output TSV (default: <input>.updated.tsv)",
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="replace the input TSV atomically",
    )
    parser.add_argument(
        "--garak-python",
        default=sys.executable,
        help="Python interpreter containing the garak version to inventory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Update an exported technique TSV and return a process status code."""
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    output_path = _output_path(
        arguments.input, arguments.output, arguments.in_place
    )

    try:
        rows = _read_tsv(arguments.input)
        metadata = _collect_probe_metadata(arguments.garak_python)
        coverage = _build_coverage(metadata)
        updated_rows, unmatched_keys = _update_rows(rows, coverage)
        _write_tsv(output_path, rows)
    except (OSError, RuntimeError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")

    print(f"Updated {updated_rows} technique rows in {output_path}")
    if unmatched_keys:
        print(
            "Warning: garak has technique tags absent from the TSV:",
            file=sys.stderr,
        )
        for key in unmatched_keys:
            print(f"  {TECHNIQUE_TAG_PREFIX}{':'.join(key)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
