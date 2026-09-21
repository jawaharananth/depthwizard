"""Generates DECISIONS.md by extracting rationale straight out of the code.

WHY THIS EXISTS INSTEAD OF A HAND-WRITTEN ADR FOLDER

A conventional architecture-decision-record set lives in its own directory,
written once when a choice is made and then never touched again -- which
means it can say "we chose X because Y" long after the code has moved to Z,
with nothing forcing the two back into agreement. This script has no
separate source of truth to drift from: it scans the actual comments sitting
directly above the code they justify, so a decision record is either current
or the comment it was extracted from no longer exists. Re-running this after
any change is how it stays honest, the same property CAPABILITIES.md already
has via scripts/verify_capabilities.py -- this is that same idea applied to
"why", not just "what's wired up".

METHOD

Walks every tracked .py file, looking for:
  1. Module/function/class docstrings (via ast, so this is a real parse,
     not a comment-shaped regex guess).
  2. Contiguous '#' comment blocks of 3+ lines immediately preceding a
     def/class/top-level assignment.
A block is kept only if it contains a real decision-rationale signal (see
SIGNAL_WORDS) -- otherwise this would just dump every comment in the repo,
which is noise, not a decision log. This is a precision choice, not a
completeness one: a block explaining WHAT something does with no stated WHY
is correctly left out.

Run: python scripts/extract_decisions.py
"""
from __future__ import annotations

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "DECISIONS.md")

SKIP_DIRS = {
    "venv", "checkpoints", "dfc2019_data", "cache", "final_out", "scenes",
    ".git", "__pycache__", "node_modules", "gamus_data", "gamus_probe",
    "buildings", "colab", "colab_subset", "colab - Copy", "mumbai_bmc",
    "mumbai_bmc_big", "mumbai_dataset", "mumbai_dataset_full",
    "ramp_karnataka", "iccd_extract", "dashboard", "dist", "docs",
    "packaging", "src-tauri", "tools", "webapp", "india", "verify_out",
    "demo_out", "dfc_out", "moge_out",
}

SIGNAL_WORDS = re.compile(
    r"\b(measured|because|chosen over|rejected|deliberately|found that|"
    r"discovered|instead of|rather than|turns out|the reason|not just|"
    r"real bug|confirmed by|verified against|the mistake|failure mode)\b",
    re.IGNORECASE,
)

MIN_BLOCK_LINES = 3


def _iter_py_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _docstring_entries(path: str, source: str) -> list[dict]:
    entries = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return entries
    nodes = [tree] + [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    for node in nodes:
        doc = ast.get_docstring(node)
        if not doc or not SIGNAL_WORDS.search(doc):
            continue
        name = getattr(node, "name", "(module)")
        line = getattr(node, "lineno", 1)
        entries.append({"name": name, "line": line, "text": doc.strip()})
    return entries


def _comment_block_entries(path: str, lines: list[str]) -> list[dict]:
    entries = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            start = i
            block = []
            while i < n and lines[i].strip().startswith("#"):
                block.append(lines[i].strip().lstrip("#").strip())
                i += 1
            if len(block) >= MIN_BLOCK_LINES:
                text = "\n".join(block)
                if SIGNAL_WORDS.search(text):
                    # Attach to the next real code line, if one follows shortly.
                    j = i
                    while j < n and not lines[j].strip():
                        j += 1
                    name = "(top of file)" if start == 0 else "(inline)"
                    if j < n:
                        m = re.match(r"\s*(def|class)\s+(\w+)", lines[j])
                        if m:
                            name = m.group(2)
                        else:
                            m2 = re.match(r"\s*([A-Z_][A-Z0-9_]*)\s*=", lines[j])
                            if m2:
                                name = m2.group(1)
                    entries.append({"name": name, "line": start + 1, "text": text})
        else:
            i += 1
    return entries


def main() -> None:
    by_file: dict[str, list[dict]] = {}
    for path in sorted(_iter_py_files()):
        try:
            with open(path, encoding="utf-8") as f:
                source = f.read()
        except Exception:
            continue
        lines = source.splitlines()
        entries = _docstring_entries(path, source) + _comment_block_entries(path, lines)
        if entries:
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            entries.sort(key=lambda e: e["line"])
            by_file[rel] = entries

    total = sum(len(v) for v in by_file.values())
    lines_out = [
        "# DECISIONS.md",
        "",
        f"Auto-generated by `scripts/extract_decisions.py` -- {total} rationale "
        f"entries extracted from {len(by_file)} files' own comments and "
        "docstrings, not hand-written. Re-run after any change; CI fails the "
        "build if this file is stale relative to the code (see "
        "`.github/workflows/ci.yml`). Do not hand-edit -- edit the source "
        "comment the entry came from and regenerate.",
        "",
    ]
    for rel in sorted(by_file):
        lines_out.append(f"## `{rel}`")
        lines_out.append("")
        for e in by_file[rel]:
            first_line = e["text"].splitlines()[0][:200]
            lines_out.append(f"- **{e['name']}** ({rel}:{e['line']}) -- {first_line}")
        lines_out.append("")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out))
    print(f"wrote {OUT_PATH}: {total} entries across {len(by_file)} files")


if __name__ == "__main__":
    main()
