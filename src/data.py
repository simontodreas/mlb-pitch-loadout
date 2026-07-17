import os

import pandas as pd
import numpy as np
import requests

# ── Constants ────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPIN_DIR = os.path.join(_REPO_ROOT, 'data', 'spin/')

SPIN_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

SPIN_CSV_URL = 'https://baseballsavant.mlb.com/leaderboard/active-spin?year={year}_spin-based&min=1&hand=&csv=true'

PITCH_TYPE_COLUMNS = [
    'pitch_type', 'pitcher', 'player_name', 'release_speed', 'release_pos_x',
    'release_pos_z', 'p_throws', 'pfx_x', 'pfx_z', 'release_spin_rate',
    'release_extension', 'release_pos_y', 'spin_axis', 'arm_angle', 'game_year'
]

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_statcast_local(paths):
    """
    Load Statcast data from local CSVs and concatenate into one DataFrame.

    Parameters:
        paths : dict with keys 'statcast_25', 'statcast_2124', and 'statcast_26'
    Returns:
        Raw combined statcast DataFrame
    """
    statcast_25   = pd.read_csv(paths['statcast_25'])
    statcast_2124 = pd.read_csv(paths['statcast_2124'])
    statcast_26   = pd.read_csv(paths['statcast_26'])
    return pd.concat([statcast_2124, statcast_25, statcast_26], ignore_index=True)


def load_statcast_live(start_dt='2025-01-01', end_dt='2026-12-31'):
    """
    Pull Statcast data live using pybaseball and save to the default CSV paths.
    Filters to regular season games only.

    Parameters:
        start_dt : start date string (YYYY-MM-DD)
        end_dt   : end date string (YYYY-MM-DD)
    Returns:
        Raw statcast DataFrame
    """
    from pybaseball import statcast  # imported lazily; only needed for live pulls
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    df = df[df['game_type'] == 'R']
    return df


def load_statcast(live=False, paths=None, start_dt='2025-01-01', end_dt='2026-12-31'):
    """
    Entry point for loading Statcast data.

    Parameters:
        live     : if True, pull from pybaseball; if False, load from local CSVs
        paths    : dict of local CSV paths (used when live=False)
        start_dt : start date for live pull
        end_dt   : end date for live pull
    Returns:
        Raw combined statcast DataFrame
    """
    if live:
        return load_statcast_live(start_dt=start_dt, end_dt=end_dt)
    if paths is None:
        raise ValueError("live=False requires `paths` to local Statcast CSVs "
                        "(the raw files are not part of this repo)")
    return load_statcast_local(paths)


def download_spin_files(years=None, spin_dir=None):
    """
    Download active-spin leaderboard CSVs from Baseball Savant into spin_dir. Existing files are overwritten.

    Parameters:
        years    : list of years to fetch (defaults to SPIN_YEARS)
        spin_dir : destination folder (defaults to SPIN_DIR)
    """
    years    = years or SPIN_YEARS
    spin_dir = spin_dir or SPIN_DIR
    os.makedirs(spin_dir, exist_ok=True)
    for year in years:
        response = requests.get(SPIN_CSV_URL.format(year=year), timeout=60)
        response.raise_for_status()
        path = os.path.join(spin_dir, f'active-spin_{str(year)[-2:]}.csv')
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f'  downloaded active-spin_{str(year)[-2:]}.csv ({len(response.content):,} bytes)')


def load_spin_data(spin_dir=None):
    """
    Load and combine active spin CSVs across years.

    Parameters:
        spin_dir : path to folder containing active-spin_YY.csv files
    Returns:
        Combined spin DataFrame with 'year' column
    """
    spin_dir = spin_dir or SPIN_DIR
    frames = []
    for year in SPIN_YEARS:
        yy = str(year)[-2:]
        df = pd.read_csv(f'{spin_dir}active-spin_{yy}.csv')
        df['year'] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# ── Cleaning & Feature Construction ──────────────────────────────────────────

def canonicalize_player_names(df):
    """
    Ensure each pitcher has a single player_name. Handles name changes across seasons (e.g. Lou Trevino -> Lou Trevino III).

    Parameters:
        df : statcast DataFrame with 'pitcher' and 'player_name' columns
    Returns:
        DataFrame with player_name made consistent per pitcher (copy)
    """
    df = df.copy()
    order_col = 'game_date' if 'game_date' in df.columns else 'game_year'
    latest_names = (
        df.dropna(subset=['player_name'])
          .sort_values(order_col)
          .groupby('pitcher')['player_name']
          .last()
    )
    df['player_name'] = df['pitcher'].map(latest_names)
    return df


def clean_statcast(statcast_raw):
    """
    Apply base cleaning rules to raw Statcast data.

    Parameters:
        statcast_raw : raw combined statcast DataFrame
    Returns:
        Cleaned statcast DataFrame
    """
    statcast_clean = statcast_raw[
        (statcast_raw['release_speed'] >= 70) &
        (statcast_raw['pitch_type'] != 'PO')
    ].copy()
    return canonicalize_player_names(statcast_clean)


def build_pitch_type_summ(statcast_clean):
    """
    Aggregate Statcast data to pitch-type level per pitcher per year.

    Parameters:
        statcast_clean : cleaned statcast DataFrame
    Returns:
        pitch_type_summ DataFrame
    """
    summ = (
        statcast_clean[PITCH_TYPE_COLUMNS]
        .groupby(['pitch_type', 'pitcher', 'player_name', 'p_throws', 'game_year'])
        .mean()
        .reset_index()
    )
    counts = (
        statcast_clean
        .groupby(['pitch_type', 'pitcher', 'player_name', 'p_throws', 'game_year'])
        .size()
        .reset_index(name='n')
    )
    return summ.merge(counts, on=['pitch_type', 'pitcher', 'player_name', 'p_throws', 'game_year'])


def build_spin_features(spin_raw):
    """
    Derive active spin fastball and FB type columns from raw spin data.

    Parameters:
        spin_raw : combined spin DataFrame from load_spin_data()
    Returns:
        Slim spin DataFrame ready to merge (pitcher, active_spin_fastball, FB_type, year)
    """
    df = spin_raw.copy()
    conditions   = [
        ~df['active_spin_fourseam'].isna(),
        df['active_spin_fourseam'].isna() & ~df['active_spin_sinker'].isna()
    ]
    choices      = [df['active_spin_fourseam'], df['active_spin_sinker']]
    choices_type = ['FF', 'SI']

    df['active_spin_fastball'] = np.select(conditions, choices, default=df['active_spin_cutter'])
    df['FB_type']              = np.select(conditions, choices_type, default='FC')
    df['pitcher']              = df['entity_id']

    return df[['pitcher', 'active_spin_fastball', 'FB_type', 'year']]


def build_pitcher_summ(statcast_clean, pitch_type_summ, spin_df_join):
    """
    Aggregate to pitcher level and merge in pitch characteristics and spin data.

    Parameters:
        statcast_clean   : cleaned statcast DataFrame
        pitch_type_summ  : output of build_pitch_type_summ()
        spin_df_join     : output of build_spin_features()
    Returns:
        pitcher_summ DataFrame
    """
    pitcher_summ = (
        statcast_clean
        .groupby(['pitcher', 'p_throws', 'player_name', 'game_year'])
        .agg(
            release_pos_x    =('release_pos_x',    'mean'),
            release_pos_z    =('release_pos_z',    'mean'),
            release_extension=('release_extension','mean'),
            arm_angle        =('arm_angle',        'mean'),
            n                =('pitcher',          'size')
        )
        .reset_index()
    )

    pitch_chars = (
        pitch_type_summ
        .groupby(['pitcher', 'p_throws', 'game_year'])
        .agg(max_velo=('release_speed','max'), max_spin=('release_spin_rate','max'))
        .reset_index()
    )

    fastball_counts = (
        pitch_type_summ[pitch_type_summ['pitch_type'].isin(['FF', 'SI'])]
        .sort_values('n', ascending=False)
        .groupby(['pitcher', 'game_year'])
        .first()
        .reset_index()
        [['pitcher', 'game_year', 'pitch_type', 'pfx_x', 'n']]
        .rename(columns={'pfx_x': 'fb_pfx_x', 'pitch_type': 'pri_fb', 'n': 'fb_n'})
    )

    pitcher_summ = pitcher_summ.merge(pitch_chars,     on=['pitcher', 'p_throws', 'game_year'])
    pitcher_summ = pitcher_summ.merge(fastball_counts, on=['pitcher', 'game_year'], how='left')
    pitcher_summ['pri_fb_cd'] = (pitcher_summ['pri_fb'] == 'FF').astype(int)
    pitcher_summ = pitcher_summ.merge(
        spin_df_join, left_on=['pitcher', 'game_year'], right_on=['pitcher', 'year'], how='inner' # Changed to inner join to ensure we only keep pitchers with spin data
    ).drop(columns='year')

    return pitcher_summ


def build_pitch_type_views(pitch_type_summ):
    """
    Build the handedness-split pitch type DataFrames used by the similarity pipeline.

    Parameters:
        pitch_type_summ : output of build_pitch_type_summ()
    Returns:
        pitch_type_r, pitch_type_l DataFrames
    """
    cols = ['pitcher', 'player_name', 'game_year', 'pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'n']
    pitch_type_r = pitch_type_summ[pitch_type_summ['p_throws'] == 'R'][cols].copy()
    pitch_type_l = pitch_type_summ[pitch_type_summ['p_throws'] == 'L'][cols].copy()
    return pitch_type_r, pitch_type_l

# ── Top-level Pipeline ────────────────────────────────────────────────────────

def build_all(live=False, paths=None, spin_dir=None):
    """
    Run the full data preparation pipeline and return all DataFrames needed by downstream modules.

    Parameters:
        live            : if True, pull Statcast live; if False, load from CSVs
        paths           : dict of local CSV paths (used when live=False)
        spin_dir        : path to spin CSV folder
    Returns:
        dict with keys:
            statcast_clean, pitch_type_summ,
            pitcher_summ, pitcher_summ_r, pitcher_summ_l,
            pitch_type_z_r, pitch_type_z_l
    """
    statcast_raw    = load_statcast(live=live, paths=paths)
    statcast_clean  = clean_statcast(statcast_raw)
    pitch_type_summ = build_pitch_type_summ(statcast_clean)
    spin_raw        = load_spin_data(spin_dir=spin_dir)
    spin_df_join    = build_spin_features(spin_raw)
    pitcher_summ    = build_pitcher_summ(statcast_clean, pitch_type_summ, spin_df_join)

    pitcher_summ_r = pitcher_summ[pitcher_summ['p_throws'] == 'R'].copy()
    pitcher_summ_l = pitcher_summ[pitcher_summ['p_throws'] == 'L'].copy()

    pitch_type_r, pitch_type_l = build_pitch_type_views(pitch_type_summ)

    return {
        'statcast_clean':  statcast_clean,
        'pitch_type_summ': pitch_type_summ,
        'pitcher_summ':    pitcher_summ,
        'pitcher_summ_r':  pitcher_summ_r,
        'pitcher_summ_l':  pitcher_summ_l,
        'pitch_type_r':  pitch_type_r,
        'pitch_type_l':  pitch_type_l,
    }