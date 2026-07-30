"""
Daily refresh of the current season's slice of the app snapshot.

The deployed app reads a committed Parquet snapshot (data/snapshots/*.parquet).
Historical seasons never change, so only the current season needs refreshing.
This module pulls fresh current-season Statcast (via pybaseball) and active-spin
data, rebuilds just that season's rows of the four snapshot tables, and splices
them into the committed snapshot in place of the stale rows — leaving prior
seasons untouched.

Intended to run daily in CI (see .github/workflows/refresh.yml); can also be run
locally:

    python -m src.refresh        # run from the repo root
"""
import datetime as dt
import json
import os

import pandas as pd

from src.data import (
    load_statcast_live, clean_statcast, build_pitch_type_summ,
    download_spin_files, load_spin_data, build_spin_features,
    build_pitcher_summ, build_pitch_type_views,
)

SNAPSHOT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'snapshots')
SNAPSHOT_KEYS = ['pitcher_summ_r', 'pitcher_summ_l', 'pitch_type_r', 'pitch_type_l']


def current_season():
    """The MLB season year to refresh — a season runs Mar–Nov, so it never
    crosses a calendar year and is simply the current year."""
    return dt.date.today().year


def build_current_season(season=None, start_dt=None, end_dt=None):
    """
    Build the four snapshot tables for a single season from live data.

    Parameters:
        season   : season year to build
        start_dt : Statcast pull start date (defaults to Jan 1 of the season)
        end_dt   : Statcast pull end date (defaults to today, capped at Dec 31)
    Returns:
        (tables, data_through) where tables is a dict keyed by SNAPSHOT_KEYS, and
        data_through is the latest game date seen (YYYY-MM-DD). Returns (None, None)
        if the season has no regular-season games yet.
    """
    season   = season or current_season()
    start_dt = start_dt or f'{season}-01-01'
    end_dt   = end_dt or min(dt.date.today(), dt.date(season, 12, 31)).isoformat()

    statcast_raw   = load_statcast_live(start_dt=start_dt, end_dt=end_dt)
    statcast_clean = clean_statcast(statcast_raw)
    if statcast_clean.empty:
        return None, None

    pitch_type_summ = build_pitch_type_summ(statcast_clean)

    # Active-spin is a season-level Savant leaderboard; pull the current season fresh.
    download_spin_files(years=[season])
    spin_raw     = load_spin_data(years=[season])
    spin_df_join = build_spin_features(spin_raw)

    pitcher_summ = build_pitcher_summ(statcast_clean, pitch_type_summ, spin_df_join)
    pitch_type_r, pitch_type_l = build_pitch_type_views(pitch_type_summ)

    tables = {
        'pitcher_summ_r': pitcher_summ[pitcher_summ['p_throws'] == 'R'].copy(),
        'pitcher_summ_l': pitcher_summ[pitcher_summ['p_throws'] == 'L'].copy(),
        'pitch_type_r':   pitch_type_r,
        'pitch_type_l':   pitch_type_l,
    }
    data_through = str(pd.to_datetime(statcast_clean['game_date']).max().date())
    return tables, data_through


def refresh_snapshot(season=None, snapshot_dir=None):
    """
    Rebuild `season`'s rows in the committed snapshot from live data, in place.

    Prior-season rows are kept exactly as-is; only rows for `season` are swapped
    for freshly built ones. Fresh tables are reindexed to the existing column
    schema so the splice stays schema-stable (and fails loudly on any drift).
    """
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    season       = season or current_season()

    fresh, data_through = build_current_season(season)
    if fresh is None:
        print(f'No {season} regular-season games available yet; snapshot unchanged.')
        return

    for key in SNAPSHOT_KEYS:
        path     = os.path.join(snapshot_dir, f'{key}.parquet')
        existing = pd.read_parquet(path)
        kept     = existing[existing['game_year'] != season]
        combined = pd.concat([kept, fresh[key][existing.columns]], ignore_index=True)
        combined.to_parquet(path, index=False)
        print(f'  {key:18s} {len(kept):>6,} kept + {len(fresh[key]):>5,} fresh = {len(combined):>6,} rows')

    with open(os.path.join(snapshot_dir, 'meta.json'), 'w') as f:
        json.dump({'data_through': data_through}, f)
    print(f'  data through {data_through}')
    print(f'Refreshed {season} rows in {snapshot_dir}')


if __name__ == '__main__':
    refresh_snapshot()
