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
    suggest_pitches, _full_name, hb_in, vb_in,
    BIOMECH_FEATURES, PITCH_CHAR_FEATURES,
    _tag_novelty, _find_target, _find_biomech_comps, _collect_pitches,
)

# Shared suggest_pitches parameters used across the validation notebook.
PARAMS = dict(min_pitches=20, biomech_distance_threshold=1.5,
              novelty_distance_threshold=1.2, min_comp_usage_pct=0.01)

_MARKERS = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
            'x', 'triangle-down', 'triangle-left', 'triangle-right', 'hexagon']


def make_validation_fig(result, actual, is_righty):
    """App-style Pitcher-View break plot with the actual 2026 novel pitch overlaid as a red star."""
    comp   = result['comp_pitches'].reset_index(drop=True)
    target = result['target_pitches']
    name   = target['player_name'].iloc[0]
    keys   = sorted(comp[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None))
    vmin, vmax = comp['release_speed'].min(), comp['release_speed'].max()

    fig = go.Figure()
    for i, (label, cid) in enumerate(keys):                                  # comp pitches, colored by velo
        g = comp[(comp['cluster_label'] == label) & (comp['cluster'] == cid)]
        fig.add_trace(go.Scatter(
            x=hb_in(g['pfx_x']), y=vb_in(g['pfx_z']), mode='markers', name=f'Possible ({_full_name(label)})',
            marker=dict(symbol=_MARKERS[i % len(_MARKERS)], size=8, color=g['release_speed'],
                        colorscale='plasma', cmin=vmin, cmax=vmax, opacity=0.7, showscale=(i == 0),
                        colorbar=dict(title=dict(text='Velo (mph)', side='right'), x=1.02, thickness=15, len=0.75) if i == 0 else None),
            customdata=np.column_stack([g['player_name'], g['pitch_type'].map(_full_name), g['release_speed']]),
            hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]:.1f} mph<extra></extra>'))

    cen = comp.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean().reset_index()
    for i, (label, cid) in enumerate(keys):                                  # suggestion centroids
        r = cen[(cen['cluster_label'] == label) & (cen['cluster'] == cid)].iloc[0]
        fig.add_trace(go.Scatter(
            x=[hb_in(r['pfx_x'])], y=[vb_in(r['pfx_z'])], mode='markers', name='Suggestion Centroid',
            showlegend=(i == 0), legendgroup='centroid',
            marker=dict(symbol=_MARKERS[i % len(_MARKERS)], size=16, color=[r['release_speed']],
                        colorscale='plasma', cmin=vmin, cmax=vmax, line=dict(color='black', width=2)),
            hovertemplate=f'<b>Suggestion: {_full_name(label)}</b><br>HB %{{x:.1f}} / IVB %{{y:.1f}} in<extra></extra>'))

    fig.add_trace(go.Scatter(                                                # existing arsenal
        x=hb_in(target['pfx_x']), y=vb_in(target['pfx_z']), mode='markers+text', name='Existing Pitch',
        marker=dict(symbol='diamond', size=15, color='black'),
        text=[_full_name(p) for p in target['pitch_type']], textposition='top right', textfont=dict(size=11, color='black'),
        hovertemplate='Existing: %{text}<extra></extra>'))

    fig.add_trace(go.Scatter(                                                # the actually-thrown 2026 novel pitch
        x=hb_in(actual['pfx_x']), y=vb_in(actual['pfx_z']), mode='markers+text', name='Actual 2026 Novel',
        marker=dict(symbol='star', size=24, color='red', line=dict(color='black', width=1.5)),
        text=[_full_name(p) for p in actual['pitch_type']], textposition='bottom center', textfont=dict(size=12, color='red'),
        customdata=np.column_stack([actual['pitch_type'].map(_full_name), actual['release_speed'], actual['min_dist_to_target']]),
        hovertemplate='<b>ACTUAL 2026: %{customdata[0]}</b><br>%{customdata[1]:.1f} mph<br>HB %{x:.1f} / IVB %{y:.1f} in<br>novelty %{customdata[2]:.2f}<extra></extra>'))

    ang = result['target_info']['arm_angle']                                # arm-angle ray (mirror RHP)
    rad, d = np.radians(ang), (-1 if is_righty else 1)
    fig.add_trace(go.Scatter(x=[0, hb_in(d * 1.5 * np.cos(rad))], y=[0, vb_in(1.5 * np.sin(rad))], mode='lines',
        name='Arm Angle', line=dict(color='rgba(50,50,50,0.3)', width=6),
        hovertemplate=f'Arm Angle {ang:.1f}°<extra></extra>'))

    grid = dict(showgrid=True, gridcolor='lightgrey', zeroline=True, zerolinecolor='darkgrey', range=[-25.2, 25.2], constrain='domain')
    fig.update_layout(
        title=dict(text=f'Actual 2026 Novel Pitch vs. Suggestions — {name}<br><sup>Pitcher View</sup>', x=0.5, xanchor='center'),
        xaxis_title='Horizontal Break (in)', yaxis_title='Induced Vertical Break (in)',
        xaxis=grid, yaxis=dict(**grid, scaleanchor='x', scaleratio=1),
        legend=dict(x=1.18, y=1, xanchor='left'), height=600, margin=dict(r=220))
    return fig


def validate_pitcher(name, novel_2026, pools, throws, pitch_type_summ):
    """App-style profile + Arsenal/Suggestions info, with the actual 2026 novel pitch alongside.

    Parameters
    ----------
    name            : player_name present in `novel_2026`
    novel_2026      : frame of actual 2026 novel pitches (one+ rows per pitcher)
    pools           : {'L'/'R': handedness pitcher-summary pool} for suggest_pitches
    throws          : {pitcher_id: 'L'/'R'} handedness lookup
    pitch_type_summ : pitch-type summary frame passed to suggest_pitches
    """
    actual = novel_2026[novel_2026['player_name'] == name]
    pit_id = int(actual['pitcher'].iloc[0])
    res    = suggest_pitches(pit_id, pools[throws[pit_id]], pitch_type_summ, **PARAMS)
    is_r   = throws[pit_id] == 'R'

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


def diagnose(pit_id, pools, throws, pitch_type_summ):
    """Walk suggest_pitches' pipeline for one pitcher and explain where (and why) it stops.

    Returns (status, n_comps, n_comp_pitches, n_novel, cause).
    """
    pool = pools[throws[pit_id]]
    status = suggest_pitches(pit_id, pool, pitch_type_summ, **PARAMS)['status']

    target_row, target_year = _find_target(pool, pit_id)
    if target_row is None:
        return status, np.nan, np.nan, np.nan, \
            f"pitcher id absent from the {throws[pit_id]}-handed 2021-2025 pool"

    comps = _find_biomech_comps(pool, pit_id, target_year, BIOMECH_FEATURES,
                                PARAMS['biomech_distance_threshold'], PARAMS['min_pitches'])
    if comps.empty:
        bd = compute_euclidean_distances(pool, features=BIOMECH_FEATURES,
                                         label_cols=['pitcher', 'game_year'], min_pitches=20)
        m = (((bd['pitcher1'] == pit_id) & (bd['game_year1'] == target_year)) |
             ((bd['pitcher2'] == pit_id) & (bd['game_year2'] == target_year)))
        nearest = bd[m & (bd['pitcher1'] != bd['pitcher2'])]['distance'].min()
        return status, 0, np.nan, np.nan, \
            f"no biomech comp within 1.5 (nearest other pitcher = {nearest:.2f})"

    tp, cp = _collect_pitches(pitch_type_summ, pit_id, target_year, comps,
                              PITCH_CHAR_FEATURES, PARAMS['min_comp_usage_pct'], PARAMS['min_pitches'])
    gs = StandardScaler().fit(pitch_type_summ[PITCH_CHAR_FEATURES].dropna().values)

    _, nov = _tag_novelty(tp, cp, PITCH_CHAR_FEATURES, PARAMS['novelty_distance_threshold'], gs)
    return status, len(comps), len(cp), len(nov), \
        f"only {len(nov)} of {len(cp)} comp pitches are novel (need >= 4 to cluster)"
