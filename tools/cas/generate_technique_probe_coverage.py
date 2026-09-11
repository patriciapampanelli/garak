#!/usr/bin/env python3
"""Generate a technique-to-probe coverage TSV from garak metadata.

Run from the garak repository root:

    python tools/cas/generate_technique_probe_coverage.py

The default output is ``./garak-technique-probe-coverage.tsv``. To also copy
the complete TSV to the system clipboard for pasting into Google Sheets:

    python tools/cas/generate_technique_probe_coverage.py --clipboard

Use ``--output PATH`` to choose another destination or ``--garak-python PATH``
to inventory garak from a different Python environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = (
    "Technique Root",
    "Technique Family",
    "Technique",
    "IntentProbe Count",
    "IntentProbes",
    "Non-IntentProbe Count",
    "Non-IntentProbes",
)
DEFAULT_OUTPUT = Path("garak-technique-probe-coverage.tsv")
EMPTY_PROBE_LIST = "—"
TECHNIQUE_TAG_PREFIX = "demon:"
PREVIEW_ROW_LIMIT = 10
PREVIEW_COLUMNS = (
    ("Technique Root", 16),
    ("Technique Family", 25),
    ("Technique", 26),
    ("Intent #", 8),
    ("Intent probes", 24),
    ("Non-intent #", 12),
    ("Non-intent probes", 36),
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
PROBE_LINE_RE = re.compile(
    r"^probes:\s+(?P<name>[A-Za-z_]\w*\.[A-Za-z_]\w*)(?:\s|$)"
)
INVENTORY_MARKER = "__GARAK_TECHNIQUE_COVERAGE__="

TechniqueKey = tuple[str, str, str]
ProbeGroups = dict[str, set[str]]
Coverage = dict[TechniqueKey, ProbeGroups]

_INVENTORY_SCRIPT = f"""
import csv
import importlib
import json
import sys

from garak import _config, _plugins
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

tag_file = _config.transient.package_dir / "data" / "tags.misp.tsv"
with tag_file.open("r", encoding="utf-8", newline="") as source:
    technique_tags = [
        row[0]
        for row in csv.reader(source, dialect="excel-tab")
        if row and row[0].startswith({TECHNIQUE_TAG_PREFIX!r})
    ]

inventory = {{"metadata": metadata, "technique_tags": technique_tags}}
print({INVENTORY_MARKER!r} + json.dumps(inventory, sort_keys=True))
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


def _parse_inventory(
    output: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    for line in reversed(output.splitlines()):
        if not line.startswith(INVENTORY_MARKER):
            continue
        try:
            inventory = json.loads(line.removeprefix(INVENTORY_MARKER))
        except json.JSONDecodeError as error:
            raise ValueError("garak returned malformed inventory data") from error

        if not isinstance(inventory, dict):
            raise ValueError("garak returned invalid inventory data")
        technique_tags = inventory.get("technique_tags")
        metadata = inventory.get("metadata")
        if not isinstance(technique_tags, list) or not all(
            isinstance(tag, str) for tag in technique_tags
        ):
            raise ValueError("garak returned invalid technique tags")
        if not isinstance(metadata, dict):
            raise ValueError("garak returned invalid probe metadata")
        return technique_tags, metadata
    raise ValueError("garak did not return inventory data")


def _collect_inventory(
    python_executable: str,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="garak-coverage-") as cache_directory:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["XDG_CACHE_HOME"] = cache_directory

        listing = _run_command(
            [python_executable, "-m", "garak", "--list_probes"],
            environment=environment,
        )
        probe_names = _parse_probe_listing(listing)
        inventory_output = _run_command(
            [python_executable, "-c", _INVENTORY_SCRIPT],
            environment=environment,
            input_text=json.dumps(probe_names),
        )
    return _parse_inventory(inventory_output)


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


def _ordered_techniques(
    technique_tags: Sequence[str], coverage: Coverage
) -> tuple[list[TechniqueKey], list[TechniqueKey]]:
    ordered_keys = []
    registered_keys = set()
    for tag in technique_tags:
        key = _technique_key(tag)
        if key in registered_keys:
            raise ValueError(f"duplicate technique tag: {tag}")
        registered_keys.add(key)
        ordered_keys.append(key)

    unregistered_keys = sorted(set(coverage) - registered_keys)
    ordered_keys.extend(unregistered_keys)
    indexed_keys = list(enumerate(ordered_keys))
    indexed_keys.sort(
        key=lambda item: (
            item[1][0].casefold(),
            -sum(len(names) for names in coverage.get(item[1], {}).values()),
            item[0],
        )
    )
    return [key for _, key in indexed_keys], unregistered_keys


def _format_probe_names(probe_names: set[str]) -> str:
    return "; ".join(sorted(probe_names)) if probe_names else EMPTY_PROBE_LIST


def _build_rows(
    technique_tags: Sequence[str], coverage: Coverage
) -> tuple[list[list[str]], list[TechniqueKey]]:
    technique_keys, unregistered_keys = _ordered_techniques(
        technique_tags, coverage
    )
    rows = [list(OUTPUT_COLUMNS)]
    for root, family, technique in technique_keys:
        groups = coverage.get((root, family, technique))
        intent_probes = groups["intent"] if groups is not None else set()
        non_intent_probes = groups["non_intent"] if groups is not None else set()
        rows.append(
            [
                root,
                family,
                technique,
                str(len(intent_probes)),
                _format_probe_names(intent_probes),
                str(len(non_intent_probes)),
                _format_probe_names(non_intent_probes),
            ]
        )
    return rows, unregistered_keys


def _serialise_tsv(rows: Sequence[Sequence[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(
        output,
        dialect="excel-tab",
        lineterminator="\n",
    )
    writer.writerows(rows)
    return output.getvalue()


def _write_tsv(path: Path, contents: str) -> None:
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
            output_file.write(contents)
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _clipboard_commands() -> tuple[tuple[str, ...], ...]:
    if sys.platform == "darwin":
        return (("pbcopy",),)
    if os.name == "nt":
        return (("clip",),)
    return (
        ("wl-copy",),
        ("xclip", "-selection", "clipboard"),
        ("xsel", "--clipboard", "--input"),
        ("clip.exe",),
    )


def _copy_with_terminal_escape(contents: str) -> None:
    payload = base64.b64encode(contents.encode("utf-8")).decode("ascii")
    sequence = f"\033]52;c;{payload}\a"
    if "TMUX" in os.environ:
        sequence = f"\033Ptmux;\033{sequence}\033\\"
    sys.stdout.write(sequence)
    sys.stdout.flush()


def _copy_to_clipboard(contents: str) -> str:
    failures = []
    for command in _clipboard_commands():
        if shutil.which(command[0]) is None:
            continue
        try:
            result = subprocess.run(
                command,
                input=contents,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            failures.append(f"{command[0]}: {error}")
            continue
        if result.returncode == 0:
            return command[0]
        failures.append(f"{command[0]}: {result.stderr.strip()}")

    if sys.stdout.isatty():
        _copy_with_terminal_escape(contents)
        return "terminal OSC 52"
    if failures:
        raise RuntimeError("clipboard copy failed: " + "; ".join(failures))
    raise RuntimeError(
        "no clipboard utility found; install wl-copy, xclip, or xsel"
    )


def _preview_cell(value: str, width: int) -> str:
    if len(value) > width:
        return f"{value[: width - 1]}…"
    return value


def _print_preview(rows: Sequence[Sequence[str]]) -> None:
    data_rows = rows[1:]
    preview_rows = data_rows[:PREVIEW_ROW_LIMIT]
    widths = [width for _, width in PREVIEW_COLUMNS]

    def render(row: Sequence[str]) -> str:
        cells = [
            _preview_cell(value, width).ljust(width)
            for value, width in zip(row, widths)
        ]
        return " | ".join(cells)

    headers = [name for name, _ in PREVIEW_COLUMNS]
    separator = "-+-".join("-" * width for width in widths)
    print(f"\nPreview ({len(preview_rows)} of {len(data_rows)} techniques):")
    print(render(headers))
    print(separator)
    for row in preview_rows:
        print(render(row))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IntentProbe and non-IntentProbe technique coverage "
            "from the installed garak probe inventory."
        )
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output TSV (default: ./{DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--garak-python",
        default=sys.executable,
        help="Python interpreter containing the garak version to inventory",
    )
    parser.add_argument(
        "--clipboard",
        "--copy",
        action="store_true",
        help="copy the complete TSV to the system clipboard",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate the coverage TSV and return a process status code."""
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        technique_tags, metadata = _collect_inventory(arguments.garak_python)
        coverage = _build_coverage(metadata)
        rows, unregistered_keys = _build_rows(technique_tags, coverage)
        tsv_contents = _serialise_tsv(rows)
        _write_tsv(arguments.output, tsv_contents)
    except (OSError, RuntimeError, UnicodeError, ValueError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")

    print(f"Wrote {len(rows) - 1} technique rows to {arguments.output}")
    _print_preview(rows)
    if arguments.clipboard:
        try:
            clipboard_utility = _copy_to_clipboard(tsv_contents)
        except RuntimeError as error:
            parser.exit(
                2,
                f"error: TSV was generated, but could not be copied: {error}\n",
            )
        print(f"\nCopied complete TSV to clipboard using {clipboard_utility}")
    if unregistered_keys:
        print(
            "Warning: included probe tags absent from garak's tag taxonomy:",
            file=sys.stderr,
        )
        for key in unregistered_keys:
            print(f"  {TECHNIQUE_TAG_PREFIX}{':'.join(key)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
