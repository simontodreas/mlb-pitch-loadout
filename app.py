import os

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from pitch_suggestions import (suggest_pitches, make_cluster_fig, _full_name,
                               BIOMECH_FEATURES, PITCH_CHAR_FEATURES, hb_in, vb_in)

SNAPSHOT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
SNAPSHOT_KEYS = ['pitcher_summ_r', 'pitcher_summ_l', 'pitch_type_r', 'pitch_type_l']

st.set_page_config(page_title="MLB Pitch Loadout", layout="wide")

st.markdown("""
<style>
/* Bump the app's base type ~2pt larger. rem-based Streamlit text (titles,
   captions, widget labels, dataframes) scales with this. The Plotly chart is
   immune (its text is SVG with explicit px sizes) and the Pitcher Profile panel
   is pinned in px below, so both keep their current size. */
html { font-size: 20px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Prefer the prebuilt Parquet snapshot; fall back to building from CSVs."""
    if all(os.path.exists(os.path.join(SNAPSHOT_DIR, f'{k}.parquet')) for k in SNAPSHOT_KEYS):
        return {k: pd.read_parquet(os.path.join(SNAPSHOT_DIR, f'{k}.parquet')) for k in SNAPSHOT_KEYS}
    from data import build_all  # heavy path: only imported if the snapshot is missing
    data = build_all(live=False)
    return {k: data[k] for k in SNAPSHOT_KEYS}


@st.cache_data(show_spinner=False)
def run_suggest(pitcher_id, is_righty, season, biomech_thr, novelty_thr, min_usage, min_pitches):
    data = load_data()
    pitcher_summ   = data['pitcher_summ_r']  if is_righty else data['pitcher_summ_l']
    pitch_type_summ = data['pitch_type_r']   if is_righty else data['pitch_type_l']
    # Restrict both pools to the selected season and earlier, so a 2025 query never pulls
    # comps or pitches from a later season. _find_target anchors on the most recent year
    # remaining in the pool, so this also makes the target season resolve to `season`.
    pitcher_summ    = pitcher_summ[pitcher_summ['game_year'] <= season]
    pitch_type_summ = pitch_type_summ[pitch_type_summ['game_year'] <= season]
    return suggest_pitches(
        target_pitcher_id=pitcher_id,
        pitcher_summ=pitcher_summ,
        pitch_type_summ=pitch_type_summ,
        biomech_distance_threshold=biomech_thr,
        novelty_distance_threshold=novelty_thr,
        min_comp_usage_pct=min_usage,
        min_pitches=min_pitches,
        mask=True
    )


@st.cache_data(show_spinner="Calibrating similarity scales...")
def distance_percentile_refs(min_pitches):
    """Global reference distributions (quantile grids) for the two distance scores.

    Built from ALL pairs league-wide — including pitchers and pitches that fail
    the comp/novelty cutoffs — so a percentile reads against the full population,
    not just the pre-filtered survivors.
    """
    data = load_data()
    q = np.linspace(0, 1, 10001)

    # Biomech: every pitcher-season pair, per hand (comps only ever come from
    # the same-handed pool), z-scored the same way _find_biomech_comps does.
    biomech_dists = []
    for k in ('pitcher_summ_r', 'pitcher_summ_l'):
        pool = data[k]
        pool = pool[pool['n'] >= min_pitches].dropna(subset=BIOMECH_FEATURES)
        X = StandardScaler().fit_transform(pool[BIOMECH_FEATURES].values)
        dm = cdist(X, X)
        biomech_dists.append(dm[np.triu_indices(len(X), k=1)])
    biomech_ref = np.quantile(np.concatenate(biomech_dists), q)

    # Novelty: the displayed statistic is a pitch's distance to the NEAREST pitch
    # in the target's arsenal, so the reference must be that same statistic
    # league-wide — pitch -> nearest pitch of another pitcher's arsenal — not raw
    # pairwise gaps (which cross-type gaps like FF-vs-CU would inflate). Sampled
    # with a fixed seed for tractability.
    rng = np.random.default_rng(0)
    pitch_dists = []
    for k in ('pitch_type_r', 'pitch_type_l'):
        pool = data[k].dropna(subset=PITCH_CHAR_FEATURES).reset_index(drop=True)
        X = StandardScaler().fit_transform(pool[PITCH_CHAR_FEATURES].values)
        codes = pd.factorize(pool['pitcher'].astype(str) + '_' + pool['game_year'].astype(str))[0]

        q_idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
        arsenals = rng.choice(np.unique(codes), size=min(400, codes.max() + 1), replace=False)

        a_mask  = np.isin(codes, arsenals)
        order   = np.argsort(codes[a_mask], kind='stable')
        Xa      = X[a_mask][order]
        a_codes = codes[a_mask][order]
        starts  = np.flatnonzero(np.r_[True, a_codes[1:] != a_codes[:-1]])

        # min distance from each query pitch to each sampled arsenal
        mins = np.minimum.reduceat(cdist(X[q_idx], Xa), starts, axis=1)

        # a pitch is trivially 0 away from its own arsenal; mask those cells
        col_of_code = {c: i for i, c in enumerate(a_codes[starts])}
        for row, qc in enumerate(codes[q_idx]):
            if qc in col_of_code:
                mins[row, col_of_code[qc]] = np.nan
        pitch_dists.append(mins[~np.isnan(mins)])
    pitch_ref = np.quantile(np.concatenate(pitch_dists), q)

    return biomech_ref, pitch_ref


def _pctile(values, ref):
    """Percentile (0-100) of each value within a quantile-grid reference."""
    return np.searchsorted(ref, np.asarray(values, dtype=float)) / (len(ref) - 1) * 100




# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    data = load_data()

pitcher_summ_r = data['pitcher_summ_r']
pitcher_summ_l = data['pitcher_summ_l']

# Global velocity color band: every pitcher's plot uses the same top and bottom
# of the scale, so colors are comparable across pitchers.
_all_speeds = pd.concat([data['pitch_type_r']['release_speed'], data['pitch_type_l']['release_speed']])
VELO_MIN, VELO_MAX = float(_all_speeds.min()), float(_all_speeds.max())

# Search is by name, but identity is the `pitcher` id. Build label -> (id, is_righty)
# options from the selectable seasons, disambiguating duplicate names by id.
SEASONS = [2025, 2026]
_pool = pd.concat([
    pitcher_summ_r[pitcher_summ_r['game_year'].isin(SEASONS)][['pitcher', 'player_name', 'game_year']].assign(is_righty=True),
    pitcher_summ_l[pitcher_summ_l['game_year'].isin(SEASONS)][['pitcher', 'player_name', 'game_year']].assign(is_righty=False),
], ignore_index=True)
# Seasons each pitcher actually appears in, so the year selector only offers valid years.
pitcher_seasons = _pool.groupby('pitcher')['game_year'].apply(lambda s: sorted(s.unique())).to_dict()

_ids = _pool.drop_duplicates(subset='pitcher')
_name_counts = _ids['player_name'].value_counts()
pitcher_options = {
    (row['player_name'] if _name_counts[row['player_name']] == 1 else f"{row['player_name']} (id {row['pitcher']})"):
        (int(row['pitcher']), bool(row['is_righty']))
    for _, row in _ids.iterrows()
}
all_pitchers = sorted(pitcher_options)

# ── Pitcher & season selectors (top of page) ─────────────────────────────────
st.title("MLB Pitch Loadout")

# snapshot.py records the max game date in the data at build time.
_meta_path = os.path.join(SNAPSHOT_DIR, 'meta.json')
if os.path.exists(_meta_path):
    import json
    with open(_meta_path) as f:
        _through = pd.to_datetime(json.load(f)['data_through'])
    st.caption(f"Data through {_through.strftime('%B')} {_through.day}, {_through.year}")
sel_col, yr_col = st.columns([3, 1])
with sel_col:
    selected = st.selectbox("Select Pitcher", all_pitchers, placeholder="Search for a pitcher…")
selected_id, is_righty = pitcher_options[selected]

# Only offer the seasons this pitcher actually appears in; default to the most recent.
avail_seasons = pitcher_seasons.get(selected_id, SEASONS)
with yr_col:
    season = st.selectbox("Season", avail_seasons, index=len(avail_seasons) - 1)

# ── Analysis parameters (fixed) ───────────────────────────────────────────────
biomech_thr  = 1.5   # max biomechanical distance to qualify as a comp
novelty_thr  = 1.2   # min pitch-char distance to count as novel vs. target
min_usage    = 0.01  # minimum usage share a comp pitch must have
min_pitches  = 20    # minimum pitch count to include a pitcher

# Global percentile scales for displaying the raw distances as 0-100 scores.
biomech_ref, pitch_ref = distance_percentile_refs(min_pitches)

# ── Main ──────────────────────────────────────────────────────────────────────
throws = "RHP" if is_righty else "LHP"

with st.spinner("Running analysis..."):
    result = run_suggest(selected_id, is_righty, season, biomech_thr, novelty_thr, min_usage, min_pitches)

status = result['status']

STATUS_MESSAGES = {
    'pitcher_not_found': "Pitcher not found in the dataset.",
    'no_comps':          "No biomechanically similar comps found for this pitcher.",
    'no_comp_pitches':   "Comps found but no usable pitch data.",
    'no_novel_pitches':  "No novel pitches found for this pitcher.",
}

if status != 'ok':
    _n_novel = len(result.get('comp_pitches', []))
    if status == 'no_novel_pitches' and _n_novel > 0:
        # too-few-to-cluster case: novel pitches exist but suggestions need >= 4
        st.warning(
            f"Only {_n_novel} novel {'pitch' if _n_novel == 1 else 'pitches'} found — at least 4 "
            "are needed to cluster into suggestions. See Novel Comparable Pitches below."
        )
    else:
        st.warning(STATUS_MESSAGES.get(status, f"Status: {status}"))

    # Statuses that still have pitch data render the normal page below the
    # warning; only statuses with nothing to show stop here.
    if status not in ('no_novel_pitches', 'no_comp_pitches'):
        st.stop()

# ── Bio panel | Pitch Plot ────────────────────────────────────────────────────
info = result['target_info']
# Spacer column pushes the chart column right so the square plot (packed
# against the legend) starts near the column's left edge, keeping the headings
# above it aligned with the plot.
bio_col, _gap_col, plot_col = st.columns([1, 0.1, 1.9])

with bio_col:
    st.subheader("Pitcher Profile")

    # MLB static headshot by pitcher id; the d_ param falls back to a generic
    # silhouette when no photo exists.
    _headshot = (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "d_people:generic:headshot:silo:current.png,q_auto:best,f_auto,w_180/"
        f"v1/people/{selected_id}/headshot/67/current"
    )
    _last, _, _first = str(info['player_name']).partition(', ')
    _display_name = f"{_first} {_last}".strip()
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:14px; padding:10px 14px;
                background:#f0efec; border:1px solid rgba(11,11,11,0.10);
                border-radius:12px; margin-bottom:12px;">
    <img src="{_headshot}" width="72" alt="{_display_name}"
        style="border-radius:50%; background:#fcfcfb; border:1px solid rgba(11,11,11,0.10);"/>
    <div>
        <div style="font-size:1.15rem; font-weight:700; color:#0b0b0b;">{_display_name}</div>
        <div style="color:#52514e; font-size:0.9rem;">{throws} &middot; {int(info['game_year'])} Season</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    profile_stats = [
        ('Arm Angle',         f"{info['arm_angle']:.1f}°"),
        ('Extension',         f"{info['release_extension']:.2f} ft"),
        ('Max Avg. Velocity', f"{info['max_velo']:.1f} mph"),
        ('Active FB Spin',    f"{info['active_spin_fastball']:.1f}%"),
        ('Total Pitches',     f"{int(info['n']):,}"),
        ('# Comp Pitchers',   f"{len(result['comps']):,}"),
    ]
    # HTML rather than st.dataframe: no height cap (so no scroll) and full
    # control of text size to fill the column beside the plot.
    _last_row = len(profile_stats) - 1
    # Streamlit injects its own CSS into every st.markdown table (margin-bottom
    # on the table, borders on tr/td, rem-based so it shifts with the html
    # font-size). Every property it touches is pinned inline here so the card
    # renders identically regardless of theme or Streamlit version: the trapped
    # table margin was drawing an empty strip inside the card's bottom edge.
    # The row and column separators wanted here are re-declared explicitly
    # after each border:none reset.
    _stat_rows = ''.join(
        # Separators sit *below* each row except the last, so the final row's
        # padding isn't left uncapped (which read as a blank half-row).
        f'<tr style="border:none;{"" if i == _last_row else " border-bottom:1px solid #d5dbe5;"}">'
        f'<td style="border:none; border-right:1px solid #d5dbe5; '
        f'padding:16px 14px; color:#52514e; font-size:24px;">{s}</td>'
        f'<td style="border:none; padding:16px 14px; text-align:right; font-weight:600; '
        f'font-size:32px; color:#0b0b0b;">{v}</td></tr>'
        for i, (s, v) in enumerate(profile_stats)
    )
    # No internal newlines/indentation: whitespace text nodes between the tags
    # otherwise render as an extra blank line box inside the card.
    st.markdown(
        '<div style="border:1px solid #d5dbe5; border-radius:8px; background:#fcfcfb; overflow:hidden;">'
        f'<table style="width:100%; border-collapse:collapse; margin:0; border:none;"><tbody>{_stat_rows}</tbody></table>'
        '</div>',
        unsafe_allow_html=True,
    )

with plot_col:
    st.subheader("Potential Pitch Plot")
    view_mode = st.segmented_control(
        "Pitch filter",
        ["Both", "Existing Only", "Recommended Only"],
        default="Both",
        label_visibility="collapsed",
    ) or "Both"  # deselecting the control returns None; treat it as Both
    show_existing  = view_mode != "Recommended Only"
    show_suggested = view_mode != "Existing Only"
    st.caption("Click, box-, or lasso-select pitches to highlight them in the tables below.")
    fig = make_cluster_fig(result, is_righty, VELO_MIN, VELO_MAX, show_existing, show_suggested)
    plot_event = st.plotly_chart(
        fig,
        width='stretch',
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key=f"pitch_plot_{selected}_{view_mode}",
        config={'scrollZoom': False, 'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d']},
    )

# ── Arsenal & Suggestions ─────────────────────────────────────────────────
st.subheader("Arsenal & Suggestions")

target = result['target_pitches'].copy()
total_n = target['n'].sum()

current_rows = pd.DataFrame({
    'Pitch':                       target['pitch_type'].map(_full_name).values,
    'Current Usage':               (target['n'] / total_n).values,
    'MPH':                         target['release_speed'].values,
    'Horizontal Break (in)':       hb_in(target['pfx_x'].values),
    'Induced Vertical Break (in)': vb_in(target['pfx_z'].values),
    '# Comparison Pitches':        np.nan,
})

sugg = result['suggestions']
if sugg is None or sugg.empty:
    # no-suggestion statuses: show the current arsenal alone
    sugg_rows = current_rows.iloc[0:0].copy()
else:
    sugg_rows = pd.DataFrame({
        'Pitch':                       sugg['cluster_label'].values + ' (Suggested)',
        'Current Usage':               np.nan,
        'MPH':                         sugg['wavg_release_speed'].values,
        'Horizontal Break (in)':       hb_in(sugg['wavg_pfx_x'].values),
        'Induced Vertical Break (in)': vb_in(sugg['wavg_pfx_z'].values),
        '# Comparison Pitches':        sugg['n_comps'].values.astype(float),
    })

current_rows['_is_sugg'] = False
sugg_rows['_is_sugg']    = True
combined = (
    pd.concat([current_rows, sugg_rows], ignore_index=True)
    .sort_values(['Current Usage', '# Comparison Pitches'], ascending=[False, False], na_position='last')
    .reset_index(drop=True)
)
sugg_mask = combined['_is_sugg'].tolist()
combined = combined.drop(columns='_is_sugg')

# st.dataframe renders null cells as "None" regardless of Styler na_rep, so the
# NA text has to be baked in as a string before the column reaches the grid.
combined['# Comparison Pitches'] = combined['# Comparison Pitches'].apply(
    lambda v: 'NA' if pd.isna(v) else f'{v:.0f}'
)

ARSENAL_SUGG_HL = 'background-color: #dbe8ff'  # highlight for suggested pitches

def _style_arsenal(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    for i, is_sugg in enumerate(sugg_mask):
        if is_sugg:
            styles.iloc[i] = ARSENAL_SUGG_HL
    return styles

st.dataframe(
    combined.style
        .format({
            'Current Usage':               '{:.1%}',
            'MPH':                         '{:.1f}',
            'Horizontal Break (in)':       '{:.1f}',
            'Induced Vertical Break (in)': '{:.1f}',
        })
        .apply(_style_arsenal, axis=None),
    width='stretch',
    hide_index=True,
)

st.markdown("---")

# ── Plot selection → table highlighting ───────────────────────────────────────
# Comp pitch traces carry customdata [player_name, pitch_full_name, row_id]; centroid/
# arm-angle/target traces don't, so filter to points that have all three fields.
selected_row_ids = []
for _pt in (plot_event.get("selection", {}).get("points", []) if plot_event else []):
    _cd = _pt.get("customdata")
    if _cd and len(_cd) >= 3:
        selected_row_ids.append(int(_cd[2]))

_comp_pitches_indexed = result['comp_pitches'].reset_index(drop=True)
selected_pitchers = set(_comp_pitches_indexed.loc[selected_row_ids, 'player_name']) if selected_row_ids else set()

SELECT_HL = 'background-color: #fff2a8'  # highlight for plot-selected rows

# Sequential ramp for Biomech Similarity (light -> dark blue).
from matplotlib.colors import LinearSegmentedColormap
_SIM_CMAP = LinearSegmentedColormap.from_list('sim_blue', ['#f2f7fd', '#5598e7'])

# ── Detail tables ─────────────────────────────────────────────────────────────
st.subheader("Comparable Pitchers")
_sim_cutoff = 100 - _pctile([biomech_thr], biomech_ref)[0]
st.caption(
    "Biomechanical Similarity is a percentile of biomechanical closeness measured against every "
    f"pitcher-season pair league-wide (100 = closest). Only pitchers at or above {_sim_cutoff:.0f} "
    "qualify as comps."
)
pitcher_summ = data['pitcher_summ_r'] if is_righty else data['pitcher_summ_l']
# comp_pitcher is a pitcher id; join on it to recover the display name + biomech features.
comps_display = (
    result['comps']
    .merge(
        pitcher_summ[['pitcher', 'game_year', 'player_name'] + BIOMECH_FEATURES],
        left_on=['comp_pitcher', 'comp_year'],
        right_on=['pitcher', 'game_year'],
        how='left',
    )
    .rename(columns={
        'player_name':          'Pitcher',
        'comp_year':            'Year',
        'distance':             'Biomechanical Similarity',
        'release_extension':    'Extension (ft)',
        'arm_angle':            'Arm Angle (°)',
        'max_velo':             'Max Avg. Velocity (mph)',
        'active_spin_fastball': 'Fastball Active Spin (%)',
    })
    [['Pitcher', 'Year', 'Biomechanical Similarity', 'Extension (ft)', 'Arm Angle (°)',
      'Max Avg. Velocity (mph)', 'Fastball Active Spin (%)']]
)
# Distance -> global percentile score (100 = closest pair league-wide).
comps_display['Biomechanical Similarity'] = 100 - _pctile(comps_display['Biomechanical Similarity'], biomech_ref)

# Raise plot-selected pitchers to the top (just under the pinned target row).
if selected_pitchers:
    _sel = comps_display['Pitcher'].isin(selected_pitchers)
    comps_display = pd.concat([comps_display[_sel], comps_display[~_sel]]).reset_index(drop=True)

_ti = result['target_info']
target_row_df = pd.DataFrame([{
    'Pitcher':                  _ti['player_name'],
    'Year':                     int(_ti['game_year']),
    'Biomechanical Similarity': np.nan,
    'Extension (ft)':           _ti['release_extension'],
    'Arm Angle (°)':            _ti['arm_angle'],
    'Max Avg. Velocity (mph)':  _ti['max_velo'],
    'Fastball Active Spin (%)': _ti['active_spin_fastball'],
}])
comps_display = pd.concat([target_row_df, comps_display], ignore_index=True)

def _style_comps(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    styles.iloc[0] = 'font-weight: bold; background-color: #dbe8ff'  # pinned target row
    for i in range(1, len(df)):
        if df.iloc[i]['Pitcher'] in selected_pitchers:
            styles.iloc[i] = SELECT_HL
    return styles

st.dataframe(
    comps_display.style
        .format({
            'Year':                     '{:.0f}',
            'Biomechanical Similarity': '{:.0f}',
            'Extension (ft)':           '{:.1f}',
            'Arm Angle (°)':            '{:.0f}',
            'Max Avg. Velocity (mph)':  '{:.1f}',
            'Fastball Active Spin (%)': '{:.1f}',
        }, na_rep='')
        .background_gradient(cmap=_SIM_CMAP, subset=['Biomechanical Similarity'],
                             vmin=_sim_cutoff, vmax=100)
        .apply(_style_comps, axis=None),
    width='stretch',
    hide_index=True,
)

st.subheader("Novel Comparable Pitches")
_nov_cutoff = _pctile([novelty_thr], pitch_ref)[0]
st.caption(
    "Novelty Score is the percentile of a pitch's distance from the nearest pitch in the current "
    "arsenal, measured against how far pitches league-wide sit from other pitchers' arsenals "
    f"(100 = most novel). Only pitches at or above {_nov_cutoff:.0f} qualify as novel."
)
cp = result['comp_pitches'].reset_index(drop=True)
display_cols = ['player_name', 'game_year', 'pitch_type',
                'usage_pct', 'release_speed', 'pfx_x', 'pfx_z',
                'cluster_label']
display_cols = [c for c in display_cols if c in cp.columns]
cp = cp[display_cols]

# Show breaks in inches (pitcher's view) and pitch codes as full names, matching the plot.
if 'pfx_x' in cp.columns:
    cp['pfx_x'] = hb_in(cp['pfx_x'])
if 'pfx_z' in cp.columns:
    cp['pfx_z'] = vb_in(cp['pfx_z'])
for _c in ('pitch_type', 'cluster_label'):
    if _c in cp.columns:
        cp[_c] = cp[_c].map(_full_name)
cp = cp.rename(columns={
    'player_name':          'Pitcher',
    'game_year':            'Year',
    'pitch_type':           'Pitch',
    'release_speed':        'Velocity (mph)',
    'pfx_x':                'Horizontal Break (in)',
    'pfx_z':                'Induced Vertical Break (in)',
    'usage_pct':            'Usage',
    'cluster_label':        'Suggested Pitch',
})

# Raise the exact plot-selected pitch rows to the top (row ids align with cp's index).
if selected_row_ids:
    _sel_idx = [i for i in selected_row_ids if i in cp.index]
    _rest    = [i for i in cp.index if i not in selected_row_ids]
    cp = cp.loc[_sel_idx + _rest]
    _novel_sel_mask = [i in selected_row_ids for i in cp.index]
else:
    _novel_sel_mask = [False] * len(cp)

def _style_novel(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    for i, on in enumerate(_novel_sel_mask):
        if on:
            styles.iloc[i] = SELECT_HL
    return styles

_novel_formats = {
    'Year':                        '{:.0f}',
    'Velocity (mph)':              '{:.1f}',
    'Horizontal Break (in)':       '{:.1f}',
    'Induced Vertical Break (in)': '{:.1f}',
    'Usage':                       '{:.1%}',
}
if cp.empty:
    st.info("No novel comparable pitches to display for this pitcher.")
else:
    st.dataframe(
        cp.style
            .format({k: v for k, v in _novel_formats.items() if k in cp.columns}, na_rep='')
            .apply(_style_novel, axis=None),
        width='stretch',
        hide_index=True,
    )

