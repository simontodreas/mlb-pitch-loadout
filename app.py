import sys, os
sys.path.insert(0, '/Users/kids/Pitcher Similarity')

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from pitch_suggestions import suggest_pitches, _full_name, BIOMECH_FEATURES, hb_in, vb_in

SNAPSHOT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
SNAPSHOT_KEYS = ['pitcher_summ_r', 'pitcher_summ_l', 'pitch_type_r', 'pitch_type_l']

st.set_page_config(page_title="Pitch Suggestions", layout="wide")

st.markdown("""
<style>
.metric-label { font-size: 0.8rem; color: #888; }
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
def run_suggest(pitcher_id, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches):
    data = load_data()
    pitcher_summ   = data['pitcher_summ_r']  if is_righty else data['pitcher_summ_l']
    pitch_type_summ = data['pitch_type_r']   if is_righty else data['pitch_type_l']
    return suggest_pitches(
        target_pitcher_id=pitcher_id,
        pitcher_summ=pitcher_summ,
        pitch_type_summ=pitch_type_summ,
        biomech_distance_threshold=biomech_thr,
        novelty_distance_threshold=novelty_thr,
        min_comp_usage_pct=min_usage,
        min_pitches=min_pitches,
    )


def _label_position(x, z, limit=25.2, margin=7.2):
    """Pick textposition that keeps labels inside the plot boundary."""
    if x > limit - margin:
        return 'top left'
    if z > limit - margin:
        return 'bottom right'
    return 'top right'


def _wrap_label(name):
    """Insert <br> near the midpoint of names longer than 10 chars."""
    if len(name) <= 10:
        return name
    mid = len(name) // 2
    left  = name.rfind(' ', 0, mid)
    right = name.find(' ', mid)
    if left == -1 and right == -1:
        return name
    if left == -1:
        split = right
    elif right == -1:
        split = left
    else:
        split = left if (mid - left) <= (right - mid) else right
    return name[:split] + '<br>' + name[split + 1:]


def make_cluster_fig(result, is_righty):
    comp_pitches   = result['comp_pitches'].reset_index(drop=True)
    target_pitches = result['target_pitches']
    pitcher_name   = target_pitches['player_name'].iloc[0]

    plotly_markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
                      'x', 'triangle-down', 'triangle-left', 'triangle-right', 'hexagon']
    cluster_keys = sorted(
        comp_pitches[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None)
    )
    cluster_key_index = {(label, cid): idx for idx, (label, cid) in enumerate(cluster_keys)}

    arm_angle_deg  = result['target_info']['arm_angle']
    arm_angle_rad  = np.radians(arm_angle_deg)

    vmin = comp_pitches['release_speed'].min()
    vmax = comp_pitches['release_speed'].max()

    fig = go.Figure()

    for i, (label, cid) in enumerate(cluster_keys):
        grp = comp_pitches[(comp_pitches['cluster_label'] == label) & (comp_pitches['cluster'] == cid)]
        fig.add_trace(go.Scatter(
            x=hb_in(grp['pfx_x']),
            y=vb_in(grp['pfx_z']),
            mode='markers',
            name=f'Possible Pitch ({_full_name(label)})',
            marker=dict(
                symbol=plotly_markers[i % len(plotly_markers)],
                size=8,
                color=grp['release_speed'],
                colorscale='plasma',
                cmin=vmin,
                cmax=vmax,
                opacity=0.7,
                showscale=(i == 0),
                colorbar=dict(
                    title=dict(text='Velocity (mph)', side='right'),
                    x=1.02,
                    thickness=15,
                    len=0.75,
                ) if i == 0 else None,
            ),
            customdata=np.column_stack([
                grp['player_name'].values,
                grp['pitch_type'].map(_full_name).values,
                grp.index.values,                       # stable row id → maps to table rows
            ]),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Pitch: %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    centroids = comp_pitches.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean().reset_index()
    for _, row in centroids.iterrows():
        idx = cluster_key_index.get((row['cluster_label'], row['cluster']), 0)
        label = row['cluster_label']
        fig.add_trace(go.Scatter(
            x=[hb_in(row['pfx_x'])],
            y=[vb_in(row['pfx_z'])],
            mode='markers',
            name='Cluster Centroid',
            showlegend=(idx == 0),
            legendgroup='centroid',
            marker=dict(
                symbol=plotly_markers[idx % len(plotly_markers)],
                size=16,
                color=[row['release_speed']],
                colorscale='plasma',
                cmin=vmin,
                cmax=vmax,
                line=dict(color='black', width=2),
                showscale=False,
            ),
            hovertemplate=(
                f'<b>Centroid: {_full_name(label)}</b><br>'
                'HBreak: %{x:.1f} in<br>'
                'IVBreak: %{y:.1f} in'
                '<extra></extra>'
            ),
        ))

    if target_pitches is not None and not target_pitches.empty:
        fig.add_trace(go.Scatter(
            x=hb_in(target_pitches['pfx_x']),
            y=vb_in(target_pitches['pfx_z']),
            mode='markers+text',
            name='Existing Pitch',
            marker=dict(symbol='diamond', size=16, color='black'),
            text=[_wrap_label(_full_name(pt)) for pt in target_pitches['pitch_type']],
            textposition=[_label_position(x, z) for x, z in zip(hb_in(target_pitches['pfx_x']), vb_in(target_pitches['pfx_z']))],
            textfont=dict(size=14, color='black'),
            customdata=np.column_stack([target_pitches['player_name'].values, target_pitches['pitch_type'].map(_full_name).values]),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Pitch: %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    # ── Arm angle (drawn on the main plot, pivoting at the origin) ─────────────
    ARM_LEN = 1.5
    arm_dir = -1 if is_righty else 1  # mirror righties so the arm enters from the correct side
    # Transform the arm endpoint with the same break helpers so it tracks the
    # flipped (pitcher's-perspective), inches-scaled pitch coordinates.
    ax_x = hb_in(arm_dir * ARM_LEN * np.cos(arm_angle_rad))
    ax_y = vb_in(ARM_LEN * np.sin(arm_angle_rad))

    fig.add_trace(go.Scatter(
        x=[0, ax_x], y=[0, ax_y], mode='lines',
        name='Arm Angle',
        line=dict(color='rgba(50,50,50,0.30)', width=6),
        hovertemplate=f'Arm Angle: {arm_angle_deg:.1f}°<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[ax_x], y=[ax_y], mode='markers',
        showlegend=False, legendgroup='Arm Angle',
        marker=dict(size=20, color='rgba(80,80,80,0.30)',
                    line=dict(color='rgba(50,50,50,0.35)', width=2)),
        hovertemplate=f'Arm Angle: {arm_angle_deg:.1f}°<extra></extra>',
    ))

    axis_range = [-25.2, 25.2]
    grid_style = dict(
        showgrid=True, gridcolor='lightgrey', gridwidth=1,
        zeroline=True, zerolinecolor='darkgrey', zerolinewidth=1.5,
        range=axis_range, constrain='domain',
    )
    fig.update_layout(
        title=dict(text=f'Potential Arsenal — {pitcher_name}<br><sup>Pitcher View</sup>', x=0.5, xanchor='center'),
        xaxis_title='Horizontal Break (in)',
        yaxis_title='Induced Vertical Break (in)',
        xaxis=grid_style,
        yaxis=dict(**grid_style, scaleanchor='x', scaleratio=1),
        dragmode='select',
        legend=dict(x=1.22, y=1, xanchor='left'),
        height=560,
        margin=dict(r=200),
    )
    return fig


# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    data = load_data()

pitcher_summ_r = data['pitcher_summ_r']
pitcher_summ_l = data['pitcher_summ_l']

# Search is by name, but identity is the `pitcher` id. Build label -> (id, is_righty)
# options from the 2025 pool, disambiguating duplicate names by id.
_pool = pd.concat([
    pitcher_summ_r[pitcher_summ_r['game_year'] == 2025][['pitcher', 'player_name']].assign(is_righty=True),
    pitcher_summ_l[pitcher_summ_l['game_year'] == 2025][['pitcher', 'player_name']].assign(is_righty=False),
], ignore_index=True).drop_duplicates(subset='pitcher')
_name_counts = _pool['player_name'].value_counts()
pitcher_options = {
    (row['player_name'] if _name_counts[row['player_name']] == 1 else f"{row['player_name']} (id {row['pitcher']})"):
        (int(row['pitcher']), bool(row['is_righty']))
    for _, row in _pool.iterrows()
}
all_pitchers = sorted(pitcher_options)

# ── Pitcher selector (top of page) ───────────────────────────────────────────
st.title("Pitch Suggestions")
selected = st.selectbox("Select Pitcher", all_pitchers, placeholder="Search for a pitcher…")
selected_id, is_righty = pitcher_options[selected]

# ── Sidebar parameters ────────────────────────────────────────────────────────
st.sidebar.subheader("Parameters")

biomech_thr  = st.sidebar.slider("Biomech Distance Threshold",  0.5, 3.0, 1.5, 0.1,
                                  help="Max biomechanical distance to qualify as a comp")
novelty_thr  = st.sidebar.slider("Novelty Distance Threshold",  0.5, 3.0, 1.2, 0.1,
                                  help="Min pitch-char distance to count as novel vs. target")
min_usage    = st.sidebar.slider("Min Comp Usage %",            0.01, 0.10, 0.01, 0.01,
                                  help="Minimum usage share a comp pitch must have")
min_pitches  = st.sidebar.slider("Min Pitches",                 10, 50, 20, 5,
                                  help="Minimum pitch count to include a pitcher")

# ── Main ──────────────────────────────────────────────────────────────────────
throws = "RHP" if is_righty else "LHP"

with st.spinner("Running analysis..."):
    result = run_suggest(selected_id, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches)

status = result['status']

STATUS_MESSAGES = {
    'pitcher_not_found': "Pitcher not found in the dataset.",
    'no_comps':          "No biomechanically similar comps found. Try raising the Biomech Distance Threshold.",
    'no_comp_pitches':   "Comps found but no usable pitch data. Try lowering Min Comp Usage % or Min Pitches.",
    'no_novel_pitches':  "No novel pitches found. Try raising the Novelty Distance Threshold.",
}

if status != 'ok':
    st.warning(STATUS_MESSAGES.get(status, f"Status: {status}"))

    if result.get('comps') is not None and not result['comps'].empty:
        with st.expander("Similar Pitchers Found"):
            st.dataframe(result['comps'], use_container_width=True)
    st.stop()

# ── Bio panel | Pitch Plot ────────────────────────────────────────────────────
info = result['target_info']
bio_col, plot_col = st.columns([1, 2])

with bio_col:
    st.subheader("Pitcher Profile")
    st.metric("Season",        int(info['game_year']))
    st.metric("Throws",        throws)
    st.metric("Arm Angle",     f"{info['arm_angle']:.1f}°")
    st.metric("Extension",     f"{info['release_extension']:.2f} ft")
    st.metric("Max Velocity",  f"{info['max_velo']:.1f} mph")
    st.metric("Primary FB",    info.get('pri_fb', 'N/A'))
    st.metric("Active Spin",   f"{info['active_spin_fastball']:.1f}%")
    st.metric("Total Pitches", f"{int(info['n']):,}")
    st.metric("Comps Found",   len(result['comps']))

with plot_col:
    st.subheader("Potential Pitch Plot")
    st.caption("Click, box-, or lasso-select pitches to highlight them in the tables below.")
    fig = make_cluster_fig(result, is_righty)
    plot_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key=f"pitch_plot_{selected}",
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
        '# Comps':                     np.nan,
    })

    sugg = result['suggestions']
    sugg_rows = pd.DataFrame({
        'Pitch':                       sugg['cluster_label'].values,
        'Current Usage':               np.nan,
        'MPH':                         sugg['wavg_release_speed'].values,
        'Horizontal Break (in)':       hb_in(sugg['wavg_pfx_x'].values),
        'Induced Vertical Break (in)': vb_in(sugg['wavg_pfx_z'].values),
        '# Comps':                     sugg['n_comps'].values.astype(float),
    })

    current_rows['_is_sugg'] = False
    sugg_rows['_is_sugg']    = True
    combined = (
        pd.concat([current_rows, sugg_rows], ignore_index=True)
        .sort_values(['Current Usage', '# Comps'], ascending=[False, False], na_position='last')
        .reset_index(drop=True)
    )
    sugg_mask = combined['_is_sugg'].tolist()
    combined = combined.drop(columns='_is_sugg')

    def _style_suggestions(df):
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        pitch_col = df.columns.get_loc('Pitch')
        for i, is_sugg in enumerate(sugg_mask):
            if is_sugg:
                styles.iloc[i, pitch_col] = 'font-style: italic'
        return styles

    st.dataframe(
        combined.style
            .format({
                'Current Usage':               '{:.1%}',
                'MPH':                         '{:.1f}',
                'Horizontal Break (in)':       '{:.1f}',
                'Induced Vertical Break (in)': '{:.1f}',
                '# Comps':                     '{:.0f}',
            }, na_rep='')
            .apply(_style_suggestions, axis=None),
        use_container_width=True,
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

# ── Detail tables ─────────────────────────────────────────────────────────────
st.subheader("Comp Pitchers")
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
        'distance':             'Biomech Distance',
        'release_extension':    'Extension (ft)',
        'arm_angle':            'Arm Angle (°)',
        'max_velo':             'Max Velo (mph)',
        'active_spin_fastball': 'Fastball Active Spin (%)',
    })
    [['Pitcher', 'Year', 'Biomech Distance', 'Extension (ft)', 'Arm Angle (°)',
      'Max Velo (mph)', 'Fastball Active Spin (%)']]
)

# Raise plot-selected pitchers to the top (just under the pinned target row).
if selected_pitchers:
    _sel = comps_display['Pitcher'].isin(selected_pitchers)
    comps_display = pd.concat([comps_display[_sel], comps_display[~_sel]]).reset_index(drop=True)

_ti = result['target_info']
target_row_df = pd.DataFrame([{
    'Pitcher':                  _ti['player_name'],
    'Year':                     int(_ti['game_year']),
    'Biomech Distance':         np.nan,
    'Extension (ft)':           _ti['release_extension'],
    'Arm Angle (°)':            _ti['arm_angle'],
    'Max Velo (mph)':           _ti['max_velo'],
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
            'Biomech Distance':         '{:.3f}',
            'Extension (ft)':           '{:.1f}',
            'Arm Angle (°)':            '{:.0f}',
            'Max Velo (mph)':           '{:.1f}',
            'Fastball Active Spin (%)': '{:.1f}',
        }, na_rep='')
        .apply(_style_comps, axis=None),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Novel Comp Pitches")
cp = result['comp_pitches'].reset_index(drop=True)
display_cols = ['player_name', 'game_year', 'pitch_type', 'release_speed',
                'pfx_x', 'pfx_z', 'usage_pct', 'min_dist_to_target',
                'closest_target_pitch', 'cluster_label', 'biomech_distance']
display_cols = [c for c in display_cols if c in cp.columns]
cp = cp[display_cols]

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

st.dataframe(
    cp.style.apply(_style_novel, axis=None),
    use_container_width=True,
    hide_index=True,
)

