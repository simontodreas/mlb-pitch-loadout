import sys, os
sys.path.insert(0, '/Users/kids/Pitcher Similarity')

import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

from pitch_suggestions import suggest_pitches

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
def run_suggest(pitcher_name, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches):
    data = load_data()
    pitcher_summ   = data['pitcher_summ_r']  if is_righty else data['pitcher_summ_l']
    pitch_type_summ = data['pitch_type_r']   if is_righty else data['pitch_type_l']
    return suggest_pitches(
        target_pitcher=pitcher_name,
        pitcher_summ=pitcher_summ,
        pitch_type_summ=pitch_type_summ,
        biomech_distance_threshold=biomech_thr,
        novelty_distance_threshold=novelty_thr,
        min_comp_usage_pct=min_usage,
        min_pitches=min_pitches,
    )


def make_cluster_fig(result):
    comp_pitches   = result['comp_pitches']
    target_pitches = result['target_pitches']
    pitcher_name   = target_pitches['player_name'].iloc[0]

    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h']
    cluster_keys = sorted(
        comp_pitches[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None)
    )

    vmin = comp_pitches['release_speed'].min()
    vmax = comp_pitches['release_speed'].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (label, cid) in enumerate(cluster_keys):
        grp    = comp_pitches[(comp_pitches['cluster_label'] == label) & (comp_pitches['cluster'] == cid)]
        marker = markers[i % len(markers)]
        ax.scatter(
            grp['pfx_x'], grp['pfx_z'],
            c=grp['release_speed'], cmap=cmap, norm=norm,
            marker=marker, s=60, alpha=0.7, zorder=2,
        )

    centroids = comp_pitches.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean()
    for i, ((label, cid), row) in enumerate(centroids.iterrows()):
        marker = markers[i % len(markers)]
        ax.scatter(
            row['pfx_x'], row['pfx_z'],
            c=[[cmap(norm(row['release_speed']))]],
            marker=marker, s=250, zorder=4,
            edgecolors='black', linewidths=1.5,
        )

    if target_pitches is not None and not target_pitches.empty:
        first = True
        for pt, grp in target_pitches.groupby('pitch_type'):
            ax.scatter(
                grp['pfx_x'], grp['pfx_z'],
                label='Existing Pitch' if first else '_nolegend_',
                color='black', s=80, zorder=3, marker='D',
            )
            for _, r in grp.iterrows():
                ax.annotate(pt, (r['pfx_x'], r['pfx_z']),
                            textcoords='offset points', xytext=(6, 4), fontsize=8, color='black')
            first = False

    legend_handles = []
    for i, (label, cid) in enumerate(cluster_keys):
        legend_handles.append(
            mlines.Line2D([], [], color='grey', marker=markers[i % len(markers)],
                          linestyle='None', markersize=7, label=label)
        )
    legend_handles.append(
        mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                      markersize=12, markeredgecolor='black', markeredgewidth=1.5,
                      label='Cluster Centroid')
    )
    legend_handles.append(
        mlines.Line2D([], [], color='black', marker='D', linestyle='None',
                      markersize=7, label='Existing Pitch')
    )

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Release speed (mph)')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Horizontal Break (ft)')
    ax.set_ylabel('Induced Vertical Break (ft)')
    ax.set_title(f'Pitch Recommendations — {pitcher_name}')
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig


# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    data = load_data()

pitcher_summ_r = data['pitcher_summ_r']
pitcher_summ_l = data['pitcher_summ_l']

pitchers_r = set(pitcher_summ_r[pitcher_summ_r['game_year'] == 2025]['player_name'].unique())
pitchers_l = set(pitcher_summ_l[pitcher_summ_l['game_year'] == 2025]['player_name'].unique())
all_pitchers = sorted(pitchers_r | pitchers_l)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Pitch Suggestions")

search = st.sidebar.text_input("Search pitcher", placeholder="e.g. Bello")
filtered = [p for p in all_pitchers if search.lower() in p.lower()] if search else all_pitchers

if not filtered:
    st.sidebar.warning("No pitchers match that search.")
    st.stop()

selected = st.sidebar.selectbox("Select pitcher", filtered)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

biomech_thr  = st.sidebar.slider("Biomech Distance Threshold",  0.5, 3.0, 1.5, 0.1,
                                  help="Max biomechanical distance to qualify as a comp")
novelty_thr  = st.sidebar.slider("Novelty Distance Threshold",  0.5, 3.0, 1.5, 0.1,
                                  help="Min pitch-char distance to count as novel vs. target")
min_usage    = st.sidebar.slider("Min Comp Usage %",            0.01, 0.10, 0.01, 0.01,
                                  help="Minimum usage share a comp pitch must have")
min_pitches  = st.sidebar.slider("Min Pitches",                 10, 50, 20, 5,
                                  help="Minimum pitch count to include a pitcher")

# ── Handedness ────────────────────────────────────────────────────────────────
is_righty = selected in pitchers_r

# ── Main ──────────────────────────────────────────────────────────────────────
throws = "RHP" if is_righty else "LHP"
st.title(f"Pitch Suggestions — {selected}")
st.caption(f"{throws}")

with st.spinner("Running analysis..."):
    result = run_suggest(selected, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches)

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

# ── Target info metrics ───────────────────────────────────────────────────────
info = result['target_info']
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Arm Angle",  f"{info['arm_angle']:.1f}°")
c2.metric("Extension",  f"{info['release_extension']:.2f} ft")
c3.metric("Max Velo",   f"{info['max_velo']:.1f} mph")
c4.metric("Primary FB", info.get('pri_fb', 'N/A'))
c5.metric("Comps Found", len(result['comps']))

st.markdown("---")

# ── Current arsenal ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6])

with left_col:
    st.subheader("Current Arsenal")
    arsenal_cols = ['pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'n']
    st.dataframe(
        result['target_pitches'][arsenal_cols].rename(columns={
            'pitch_type':    'Pitch',
            'release_speed': 'Velo',
            'pfx_x':         'HBreak',
            'pfx_z':         'IVBreak',
            'n':             'Count',
        }).style.format({'Velo': '{:.1f}', 'HBreak': '{:.2f}', 'IVBreak': '{:.2f}'}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Suggestions")
    sugg_cols = ['cluster_label', 'n_comps', 'wavg_release_speed', 'wavg_pfx_x', 'wavg_pfx_z',
                 '_sil', 'pitch_types_in_cluster']
    st.dataframe(
        result['suggestions'][sugg_cols].rename(columns={
            'cluster_label':          'Label',
            'n_comps':                '# Comps',
            'wavg_release_speed':     'Wtd Velo',
            'wavg_pfx_x':             'Wtd HBreak',
            'wavg_pfx_z':             'Wtd IVBreak',
            '_sil':                   'Silhouette',
            'pitch_types_in_cluster': 'Pitch Types',
        }).style.format({'Wtd Velo': '{:.1f}', 'Wtd HBreak': '{:.2f}', 'Wtd IVBreak': '{:.2f}',
                         'Silhouette': '{:.3f}'}),
        use_container_width=True,
        hide_index=True,
    )

with right_col:
    st.subheader("Cluster Plot")
    fig = make_cluster_fig(result)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("---")

# ── Detail tables ─────────────────────────────────────────────────────────────
with st.expander("Comp Pitchers"):
    st.dataframe(result['comps'], use_container_width=True, hide_index=True)

with st.expander("Novel Comp Pitches"):
    cp = result['comp_pitches'].copy()
    display_cols = ['player_name', 'game_year', 'pitch_type', 'release_speed',
                    'pfx_x', 'pfx_z', 'usage_pct', 'min_dist_to_target',
                    'closest_target_pitch', 'cluster_label', 'biomech_distance']
    display_cols = [c for c in display_cols if c in cp.columns]
    st.dataframe(cp[display_cols], use_container_width=True, hide_index=True)

