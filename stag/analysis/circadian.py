# ╔══════════════════════════════════════════════════════════════════╗
# ║  STAG — analysis.circadian                                       ║
# ║  « hourly proportions, day/night tests, per-animal time budgets»║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Sprint 3 deliverables that depend on the per-sample (deer_id,  ║
# ║  timestamp) cache:                                              ║
# ║                                                                  ║
# ║    - hourly_proportions(idx, ts) — diurnal panel data           ║
# ║    - split_by_day(idx, ts) — day-1 vs day-2 consistency panel   ║
# ║      (R1 + R3 asked for the second 24 h to be analysed too)     ║
# ║    - ear_flick_day_night_test(idx, ts, deer_ids, ear_pms,       ║
# ║      activity_pms) — R1 #10: paired Wilcoxon per animal,        ║
# ║      day-rate / night-rate, normalised to overall activity      ║
# ║    - per_animal_time_budget(idx, deer_ids) — R2 #8 individual   ║
# ║      variability table                                           ║
# ║                                                                  ║
# ║  Day/night is astral-based (sunrise/sunset for the Waikato      ║
# ║  recording site on 2018-11-12 → 2018-12-01), not a fixed 06/18  ║
# ║  clock cut.  Crepuscular margins are excluded by default (15    ║
# ║  min on each side of the solar event) — adjustable.             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Circadian summaries, day/night Wilcoxon, and per-animal time budgets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as _date

import numpy as np
import pandas as pd
from astral import LocationInfo
from astral.sun import sun
from scipy.stats import wilcoxon

# ─────────────────────────────────────────────────────────────────
#  Recording site (from per-animal-means median GPS, see Sprint 3
#  cache step — Tokanui-area paddock, Waikato, NZ).
# ─────────────────────────────────────────────────────────────────

RECORDING_SITE: LocationInfo = LocationInfo(
    name="Tokanui_area",
    region="Waikato, NZ",
    timezone="Pacific/Auckland",
    latitude=-38.11015,
    longitude=175.49841,
)


# ─────────────────────────────────────────────────────────────────
#  Day/night classification
# ─────────────────────────────────────────────────────────────────


def _solar_events_table(
    dates: Sequence[_date],
    site: LocationInfo = RECORDING_SITE,
    crepuscular_margin_minutes: float = 15.0,
) -> pd.DataFrame:
    """Return per-date sunrise / sunset plus padded day-window edges.

    The padded edges shrink the "day" window by ``crepuscular_margin_minutes``
    on each side so the test compares well-resolved day to well-resolved
    night and ignores the ambiguous twilight periods.  Pass ``0.0`` to
    use the raw solar events.
    """
    rows = []
    margin_ns = int(crepuscular_margin_minutes * 60 * 1e9)
    for d in dates:
        s = sun(site.observer, date=d, tzinfo=site.timezone)
        # The cache step writes NZ_DateTime as naive wall-clock
        # nanoseconds — "12:00 NZDT" is stored as the int value of
        # "12:00 UTC".  Match that convention here by stripping the
        # tz info from astral's tz-aware events (keeps the wall-clock
        # numbers, treats them as UTC for int conversion).
        sunrise_ns = pd.Timestamp(s["sunrise"].replace(tzinfo=None)).value
        sunset_ns  = pd.Timestamp(s["sunset"].replace(tzinfo=None)).value
        rows.append({
            "date":          d,
            "sunrise_ns":    sunrise_ns,
            "sunset_ns":     sunset_ns,
            "day_start_ns":  sunrise_ns + margin_ns,
            "day_end_ns":    sunset_ns  - margin_ns,
        })
    return pd.DataFrame(rows)


def classify_day_night(
    timestamps_ns: np.ndarray,
    site: LocationInfo = RECORDING_SITE,
    crepuscular_margin_minutes: float = 15.0,
) -> np.ndarray:
    """Per-sample 0/1/-1 = night / day / crepuscular for ``timestamps_ns``.

    Args:
        timestamps_ns:                int64 nanoseconds since the Unix
                                      epoch, NZ local time (what the
                                      cache writes).
        site:                         Astral LocationInfo (default Waikato).
        crepuscular_margin_minutes:   Minutes excluded around sunrise /
                                      sunset.  Default 15 min.

    Returns:
        int8 array, same length as ``timestamps_ns``:
          - 1 = day (between sunrise+margin and sunset-margin)
          - 0 = night
          - -1 = crepuscular (excluded from day/night comparisons)
    """
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    if ts.size == 0:
        return np.zeros(0, dtype=np.int8)

    dates = pd.to_datetime(ts).date
    unique_dates = sorted(set(dates))
    table = _solar_events_table(unique_dates, site=site,
                                crepuscular_margin_minutes=crepuscular_margin_minutes)
    by_date = {row["date"]: row for _, row in table.iterrows()}

    out = np.zeros(ts.size, dtype=np.int8)  # default night
    # Vectorise per-date using contiguous spans — much faster than
    # per-sample lookup on 200 M rows.
    df = pd.DataFrame({"ts": ts, "date": dates})
    for d, group in df.groupby("date", sort=False):
        ev = by_date[d]
        block = group.index.to_numpy()
        sub_ts = ts[block]
        is_day = (sub_ts >= ev["day_start_ns"]) & (sub_ts < ev["day_end_ns"])
        # Crepuscular margins: [sunrise, day_start) ∪ [day_end, sunset)
        is_crep = (
            ((sub_ts >= ev["sunrise_ns"]) & (sub_ts < ev["day_start_ns"]))
            |
            ((sub_ts >= ev["day_end_ns"])   & (sub_ts < ev["sunset_ns"]))
        )
        out[block[is_day]]  =  1
        out[block[is_crep]] = -1
    return out


# ─────────────────────────────────────────────────────────────────
#  Hourly proportions
# ─────────────────────────────────────────────────────────────────


def hourly_proportions(
    idx: np.ndarray,
    timestamps_ns: np.ndarray,
    pm_ids: Sequence[int],
) -> pd.DataFrame:
    """Per-hour proportion of each PM across the cohort.

    Args:
        idx:           Per-sample cluster IDs.
        timestamps_ns: int64 ns timestamps aligned with ``idx``.
        pm_ids:        PMs to report (columns of the result).

    Returns:
        ``DataFrame`` indexed by ``hour_of_day`` (0..23) with one
        column per requested PM, plus ``n_samples``.  Rows sum to 1
        across PM columns when all PMs in ``idx`` are listed.
    """
    idx = np.asarray(idx)
    ts = pd.to_datetime(np.asarray(timestamps_ns, dtype=np.int64))
    hour = ts.hour

    df = pd.DataFrame({"hour": hour, "pm": idx})
    counts = df.groupby("hour")["pm"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=list(pm_ids), fill_value=0)
    n_per_hour = counts.sum(axis=1).rename("n_samples")
    proportions = counts.div(n_per_hour, axis=0)
    proportions["n_samples"] = n_per_hour
    return proportions


# ─────────────────────────────────────────────────────────────────
#  Day-split for replication panels
# ─────────────────────────────────────────────────────────────────


def split_by_day(
    timestamps_ns: np.ndarray,
    deer_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Per-sample "recording day" index (0 = first 24 h, 1 = next, …).

    Day boundaries are local-time midnights of the *first* timestamp
    in the array (per animal, when ``deer_ids`` is provided).  This
    is the index reviewers asked for in the day-1 vs day-2 panel
    (R1 + R3 want to see the second day replicates the first).

    Args:
        timestamps_ns: int64 ns timestamps.
        deer_ids:      Optional per-sample deer_id; when given,
                       day-indexing restarts at each animal's first
                       timestamp.

    Returns:
        int8 array of day indices (0, 1, 2, …), length matching input.
    """
    ts = pd.to_datetime(np.asarray(timestamps_ns, dtype=np.int64))
    out = np.zeros(ts.size, dtype=np.int8)
    if deer_ids is None:
        groups = [(None, np.arange(ts.size))]
    else:
        deer_ids = np.asarray(deer_ids)
        groups = [
            (int(d), np.flatnonzero(deer_ids == d))
            for d in np.unique(deer_ids)
        ]
    for _, idx_block in groups:
        if idx_block.size == 0:
            continue
        block_ts = ts[idx_block]
        start = block_ts[0].floor("D")
        out[idx_block] = ((block_ts - start) // pd.Timedelta(days=1)).astype("int8").to_numpy()
    return out


# ─────────────────────────────────────────────────────────────────
#  Ear-flick day/night rate test
# ─────────────────────────────────────────────────────────────────


def ear_flick_day_night_test(
    idx: np.ndarray,
    timestamps_ns: np.ndarray,
    deer_ids: np.ndarray,
    ear_flick_pms: Sequence[int],
    activity_pms: Sequence[int],
    site: LocationInfo = RECORDING_SITE,
    crepuscular_margin_minutes: float = 15.0,
) -> dict:
    """Paired Wilcoxon test of ear-flick rate, day vs night, per animal.

    R1 #10 — "test whether ear flicks are diurnal".  The rate is
    normalised to overall activity (any PM in ``activity_pms``) so
    a higher rate during the day is not just driven by the animal
    being awake.

    For each animal we compute:
        rate_day   = (# ear-flick samples during day)   / (# activity samples during day)
        rate_night = (# ear-flick samples during night) / (# activity samples during night)

    and run a paired Wilcoxon signed-rank test on ``(rate_day, rate_night)``
    across animals.  Crepuscular samples are excluded.

    Returns:
        Dict with the per-animal table and the test statistics:
          ``per_animal``:  ``DataFrame`` with rate_day / rate_night.
          ``W``:           Wilcoxon test statistic.
          ``p_value``:     Two-sided p.
          ``median_ratio_day_over_night``: across animals.
          ``q025_ratio``, ``q975_ratio``: 2.5/97.5 bootstrap of ratio.
    """
    idx = np.asarray(idx)
    deer_ids = np.asarray(deer_ids)
    day_label = classify_day_night(
        timestamps_ns, site=site,
        crepuscular_margin_minutes=crepuscular_margin_minutes,
    )

    ear_set = np.isin(idx, list(ear_flick_pms))
    act_set = np.isin(idx, list(activity_pms))

    rows = []
    for d in np.unique(deer_ids):
        mask = deer_ids == d
        if not mask.any():
            continue
        ear  = ear_set[mask]
        act  = act_set[mask]
        dl   = day_label[mask]
        day_act   = int(((dl ==  1) & act).sum())
        night_act = int(((dl ==  0) & act).sum())
        day_ear   = int(((dl ==  1) & ear).sum())
        night_ear = int(((dl ==  0) & ear).sum())
        rows.append({
            "deer_id":    int(d),
            "day_act":    day_act,
            "night_act":  night_act,
            "day_ear":    day_ear,
            "night_ear":  night_ear,
            "rate_day":   day_ear   / day_act   if day_act   > 0 else float("nan"),
            "rate_night": night_ear / night_act if night_act > 0 else float("nan"),
        })
    per_animal = pd.DataFrame(rows).set_index("deer_id")
    finite = per_animal.dropna(subset=["rate_day", "rate_night"])

    if len(finite) < 6:
        return {
            "per_animal": per_animal,
            "n_animals_in_test": int(len(finite)),
            "W": float("nan"), "p_value": float("nan"),
            "median_ratio_day_over_night": float("nan"),
            "q025_ratio": float("nan"), "q975_ratio": float("nan"),
        }

    W, p = wilcoxon(finite["rate_day"].to_numpy(),
                    finite["rate_night"].to_numpy())
    ratios = finite["rate_day"] / finite["rate_night"].replace(0, np.nan)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "per_animal":                 per_animal,
        "n_animals_in_test":          int(len(finite)),
        "W":                          float(W),
        "p_value":                    float(p),
        "median_ratio_day_over_night": float(ratios.median()),
        "q025_ratio":                 float(ratios.quantile(0.025)) if len(ratios) else float("nan"),
        "q975_ratio":                 float(ratios.quantile(0.975)) if len(ratios) else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────
#  Per-animal time budget
# ─────────────────────────────────────────────────────────────────


def per_animal_time_budget(
    idx: np.ndarray,
    deer_ids: np.ndarray,
    pm_ids: Sequence[int],
) -> pd.DataFrame:
    """Per-animal proportion of time in each PM (rows = animal, cols = PM).

    Used by the R2 #8 individual-variability supplementary figure
    (stacked bar of PM proportions per stag, ordered by inactive
    proportion).
    """
    idx = np.asarray(idx)
    deer_ids = np.asarray(deer_ids)
    df = pd.DataFrame({"deer_id": deer_ids, "pm": idx})
    counts = df.groupby("deer_id")["pm"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=list(pm_ids), fill_value=0)
    n_per_animal = counts.sum(axis=1).rename("n_samples")
    proportions = counts.div(n_per_animal, axis=0)
    proportions["n_samples"] = n_per_animal
    return proportions
