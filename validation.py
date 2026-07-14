"""Helpers for `2026 Validation.ipynb`.

Compare `suggest_pitches` output against the novel pitches MLB pitchers actually
threw in 2026: an app-style break plot with the actual pitch overlaid, a
side-by-side arsenal/suggestion table, and a per-pitcher pipeline diagnostic.

These were previously defined inline in the notebook; they live here so the
notebook stays a thin driver. Functions take the notebook's frames/lookups as
arguments rather than reaching for globals.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from IPython.display import display

from distances import compute_euclidean_distances
from pitch_suggestions import (
    suggest_pitches, make_cluster_fig, _full_name, hb_in, vb_in,
    BIOMECH_FEATURES, PITCH_CHAR_FEATURES,
    _tag_novelty, _find_target, _find_biomech_comps, _collect_pitches,
)

# Shared suggest_pitches parameters used across the validation notebook.
PARAMS = dict(min_pitches=20, biomech_distance_threshold=1.5,
              novelty_distance_threshold=1.2, min_comp_usage_pct=0.01)


def make_validation_fig(result, actual, is_righty):
    """The app's break plot (via make_cluster_fig), with the actual 2026 novel
    pitch overlaid as a red star — the only difference from what the app shows."""
    fig = make_cluster_fig(result, is_righty)

    fig.add_trace(go.Scatter(                            # the actually-thrown 2026 novel pitch
        x=hb_in(actual['pfx_x']), y=vb_in(actual['pfx_z']), mode='markers+text',
        name='Actual 2026 Novel',
        marker=dict(symbol='star', size=24, color='red', line=dict(color='black', width=1.5)),
        text=[_full_name(p) for p in actual['pitch_type']],
        textposition='bottom center', textfont=dict(size=16, color='red'),
        customdata=np.column_stack([actual['pitch_type'].map(_full_name), actual['release_speed'], actual['min_dist_to_target']]),
        hovertemplate=(
            '<b>ACTUAL 2026: %{customdata[0]}</b><br>'
            '%{customdata[1]:.1f} mph<br>'
            'HBreak: %{x:.1f} in<br>'
            'IVBreak: %{y:.1f} in<br>'
            'Novelty: %{customdata[2]:.2f}'
            '<extra></extra>'
        ),
    ))
    return fig


def validate_pitcher(name, novel_2026, pools, throws, pitch_pools):
    """App-style profile + Arsenal/Suggestions info, with the actual 2026 novel pitch alongside.

    Parameters
    ----------
    name        : player_name present in `novel_2026`
    novel_2026  : frame of actual 2026 novel pitches (one+ rows per pitcher)
    pools       : {'L'/'R': handedness pitcher-summary pool} for suggest_pitches
    throws      : {pitcher_id: 'L'/'R'} handedness lookup
    pitch_pools : {'L'/'R': handedness pitch-type summary frame} for suggest_pitches,
                  mirroring the app's per-hand frames
    """
    actual = novel_2026[novel_2026['player_name'] == name]
    pit_id = int(actual['pitcher'].iloc[0])
    hand   = throws[pit_id]
    res    = suggest_pitches(pit_id, pools[hand], pitch_pools[hand], **PARAMS)
    is_r   = hand == 'R'

    if res['status'] != 'ok':
        print(f"No suggestions for {name} (status: {res['status']}).")
        display(actual)
        return

    make_validation_fig(res, actual, is_r).show()

    t, s = res['target_pitches'], res['suggestions']
    table = pd.concat([
        pd.DataFrame({'Kind': 'Current', 'Pitch': t['pitch_type'].map(_full_name).values,
                      'Usage': (t['n'] / t['n'].sum()).values, 'MPH': t['release_speed'].values,
                      'HBreak (in)': hb_in(t['pfx_x'].values), 'IVBreak (in)': vb_in(t['pfx_z'].values), '# Comps': np.nan}),
        pd.DataFrame({'Kind': 'Suggested', 'Pitch': s['cluster_label'].values, 'Usage': np.nan,
                      'MPH': s['wavg_release_speed'].values, 'HBreak (in)': hb_in(s['wavg_pfx_x'].values),
                      'IVBreak (in)': vb_in(s['wavg_pfx_z'].values), '# Comps': s['n_comps'].values.astype(float)}),
        pd.DataFrame({'Kind': 'ACTUAL 2026', 'Pitch': actual['pitch_type'].map(_full_name).values, 'Usage': np.nan,
                      'MPH': actual['release_speed'].values, 'HBreak (in)': hb_in(actual['pfx_x'].values),
                      'IVBreak (in)': vb_in(actual['pfx_z'].values), '# Comps': np.nan}),
    ], ignore_index=True)
    display(table.style.format({'Usage': '{:.1%}', 'MPH': '{:.1f}', 'HBreak (in)': '{:.1f}',
                                'IVBreak (in)': '{:.1f}', '# Comps': '{:.0f}'}, na_rep=''))


def diagnose(pit_id, pools, throws, pitch_pools):
    """Walk suggest_pitches' pipeline for one pitcher and explain where (and why) it stops.

    Returns (status, n_comps, n_comp_pitches, n_novel, cause).
    """
    hand = throws[pit_id]
    pool, pitch_type_summ = pools[hand], pitch_pools[hand]
    status = suggest_pitches(pit_id, pool, pitch_type_summ, **PARAMS)['status']

    target_row, target_year = _find_target(pool, pit_id)
    if target_row is None:
        return status, np.nan, np.nan, np.nan, \
            f"pitcher id absent from the {throws[pit_id]}-handed 2021-2025 pool"

    comps = _find_biomech_comps(pool, pit_id, target_year, BIOMECH_FEATURES,
                                PARAMS['biomech_distance_threshold'], PARAMS['min_pitches'])
    if comps.empty:
        bd = compute_euclidean_distances(pool, features=BIOMECH_FEATURES,
                                         label_cols=['pitcher', 'game_year'],
                                         min_pitches=PARAMS['min_pitches'])
        m = (((bd['pitcher1'] == pit_id) & (bd['game_year1'] == target_year)) |
             ((bd['pitcher2'] == pit_id) & (bd['game_year2'] == target_year)))
        nearest = bd[m & (bd['pitcher1'] != bd['pitcher2'])]['distance'].min()
        return status, 0, np.nan, np.nan, \
            f"no biomech comp within {PARAMS['biomech_distance_threshold']} " \
            f"(nearest other pitcher = {nearest:.2f})"

    tp, cp = _collect_pitches(pitch_type_summ, pit_id, target_year, comps,
                              PITCH_CHAR_FEATURES, PARAMS['min_comp_usage_pct'], PARAMS['min_pitches'])
    gs = StandardScaler().fit(pitch_type_summ[PITCH_CHAR_FEATURES].dropna().values)

    _, nov = _tag_novelty(tp, cp, PITCH_CHAR_FEATURES, PARAMS['novelty_distance_threshold'], gs)
    return status, len(comps), len(cp), len(nov), \
        f"only {len(nov)} of {len(cp)} comp pitches are novel (need >= 4 to cluster)"
