# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — tests.test_no_hmm_terminology                            ║
# ║  « regression guard for the HMM → Markov rename »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  All three reviewers flagged the HMM-vs-first-order-Markov       ║
# ║  terminology confusion.  Sprint 0.5 renamed every occurrence     ║
# ║  in source, docs, and citation metadata to "Markov".  This       ║
# ║  test re-greps the tree on every CI run so the rename does not   ║
# ║  silently regress through future edits.                          ║
# ║                                                                  ║
# ║  Allowlist exists for legitimate references (e.g. a one-line     ║
# ║  history note explaining the rename to readers coming from the   ║
# ║  manuscript draft).  Add file paths to ALLOWED_FILES if you      ║
# ║  intentionally introduce a new mention.                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Regression test: no "HMM" or "Hidden Markov" outside an allowlist."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories to scan and the file extensions inside them.
SCAN_TARGETS: dict[Path, tuple[str, ...]] = {
    REPO_ROOT / "stag":  (".py",),
    REPO_ROOT / "docs":  (".rst", ".py"),
    REPO_ROOT / "tests": (".py",),
}

# Top-level files to scan unconditionally.
TOP_LEVEL_FILES: tuple[Path, ...] = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "CITATION.cff",
    REPO_ROOT / "pyproject.toml",
)

# Files explicitly allowed to contain "HMM" / "Hidden Markov" — extend
# only with a comment explaining why the mention is legitimate.
ALLOWED_FILES: set[Path] = {
    Path(__file__).resolve(),  # this regression test contains the strings by design
}

PATTERN = re.compile(r"HMM|Hidden Markov|hidden.markov|hidden-markov", re.IGNORECASE)


def _iter_candidate_files() -> list[Path]:
    candidates: list[Path] = []
    for directory, extensions in SCAN_TARGETS.items():
        if not directory.is_dir():
            continue
        for ext in extensions:
            candidates.extend(directory.rglob(f"*{ext}"))
    candidates.extend(p for p in TOP_LEVEL_FILES if p.exists())
    return candidates


def test_no_hmm_terminology_in_source() -> None:
    """Fail if any file outside the allowlist mentions HMM / Hidden Markov."""
    violations: list[tuple[Path, int, str]] = []
    for path in _iter_candidate_files():
        resolved = path.resolve()
        if resolved in ALLOWED_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PATTERN.search(line):
                violations.append((resolved.relative_to(REPO_ROOT), lineno, line.strip()))

    if violations:
        bullets = "\n".join(
            f"  {p}:{ln}  {snippet}" for p, ln, snippet in violations
        )
        raise AssertionError(
            "HMM / Hidden Markov terminology reintroduced.  All three "
            "reviewers flagged this; rename to 'first-order Markov "
            "transition model' or extend the test allowlist with a "
            "comment justifying the new mention.\n\n"
            f"Offending lines:\n{bullets}"
        )
