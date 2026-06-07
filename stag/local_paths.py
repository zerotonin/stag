# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — local_paths                                              ║
# ║  « env > local_paths.json > committed default — every path,      ║
# ║    every script »                                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Single resolver for every machine-specific path STAG needs.     ║
# ║  Tracked code never contains a developer's username or drive     ║
# ║  label; instead it asks ``get_path("data_root")`` (or one of the ║
# ║  other keys catalogued below) and the resolver picks the right   ║
# ║  value via this priority:                                        ║
# ║                                                                  ║
# ║    1. Environment variable           (deployment override)       ║
# ║    2. ``local_paths.json``           (per-machine config, .gitignored)║
# ║    3. ``default`` keyword            (caller-supplied fallback)  ║
# ║    4. Raise ``LocalPathNotConfiguredError`` (no silent failure)  ║
# ║                                                                  ║
# ║  Reviewers fill in their copy of ``local_paths.json`` by         ║
# ║  copying ``local_paths.template.json`` and editing the           ║
# ║  ``<placeholder>`` strings.  See README.md > Installation > Local║
# ║  paths.                                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Machine-specific path resolver — env > local_paths.json > default."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_LOCAL_PATHS_FILE: Path = _PROJECT_ROOT / "local_paths.json"
_TEMPLATE_FILE: Path = _PROJECT_ROOT / "local_paths.template.json"

# JSON-key -> environment-variable name.  Env var wins when set.
_ENV_VAR_NAMES: dict[str, str] = {
    "data_root":                "STAG_DATA_DIR",
    "hcs_source":               "STAG_HCS_DIR",
    "aoraki_data_root":         "STAG_AORAKI_DATA_ROOT",
    "aoraki_raw_data":          "STAG_AORAKI_RAW_DATA",
    "aoraki_deer_codes":        "STAG_AORAKI_DEER_CODES",
    "aoraki_merged_signals":    "STAG_AORAKI_MERGED_SIGNALS",
    "aoraki_merged_signals_v2": "STAG_AORAKI_MERGED_SIGNALS_V2",
    "aoraki_plot_dir":          "STAG_AORAKI_PLOT_DIR",
    "aoraki_plot_dir_v2":       "STAG_AORAKI_PLOT_DIR_V2",
    "aoraki_log_dir":           "STAG_AORAKI_LOG_DIR",
    "aoraki_log_dir_v2":        "STAG_AORAKI_LOG_DIR_V2",
    "aoraki_db_folder":         "STAG_AORAKI_DB_FOLDER",
    "aoraki_quality_control_dir": "STAG_AORAKI_QC_DIR",
    "aoraki_correlations_file": "STAG_AORAKI_CORRELATIONS_FILE",
    "aoraki_loc_and_tort_dir":  "STAG_AORAKI_LOC_AND_TORT_DIR",
    "deer_db_url":              "STAG_DEER_DB_URL",
    "deer_db_url_legacy":       "STAG_DEER_DB_URL_LEGACY",
    "alex_home":                "STAG_ALEX_HOME",
}

_PLACEHOLDER_SENTINEL: str = "<"  # all template values start with `<...>`

_cache: dict[str, Any] | None = None


class LocalPathNotConfiguredError(RuntimeError):
    """Raised when a local-path key cannot be resolved.

    Triggered when neither the matching env var, ``local_paths.json``,
    nor an explicit ``default`` kwarg supplies a value.
    """


def _load_json() -> dict[str, Any]:
    """Read ``local_paths.json`` once and memoise; empty dict if missing."""
    global _cache
    if _cache is not None:
        return _cache
    if _LOCAL_PATHS_FILE.exists():
        _cache = json.loads(
            _LOCAL_PATHS_FILE.read_text(encoding="utf-8"),
        )
    else:
        _cache = {}
    return _cache


def _is_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(_PLACEHOLDER_SENTINEL)


def get_path(key: str, *, default: str | None = None) -> str:
    """Resolve ``key`` to a concrete path string.

    Priority order:
      1. Environment variable named in :data:`_ENV_VAR_NAMES` for ``key``.
      2. The same-named field in ``local_paths.json``.
      3. ``default`` keyword argument.
      4. :class:`LocalPathNotConfiguredError`.

    Placeholder values in ``local_paths.json`` (strings starting with
    ``<``) are ignored — they signal "still to be filled in" and the
    resolver falls through to the default or raises.
    """
    env_name = _ENV_VAR_NAMES.get(key)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]

    data = _load_json()
    if key in data and not _is_placeholder(data[key]):
        return str(data[key])

    if default is not None:
        return default

    hint = (
        f"copy {_TEMPLATE_FILE.name} to local_paths.json and edit the "
        f"{key!r} value"
    )
    if env_name:
        hint = f"either export {env_name} or {hint}"
    raise LocalPathNotConfiguredError(
        f"Local path {key!r} is not configured; {hint}.",
    )


def get_path_obj(key: str, *, default: str | None = None) -> Path:
    """Same as :func:`get_path` but returns a :class:`Path`."""
    return Path(get_path(key, default=default))


def reset_cache() -> None:
    """Drop the memoised ``local_paths.json`` content (test-helper)."""
    global _cache
    _cache = None
