from distances import compute_euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# Constants
BIOMECH_FEATURES    = ['release_extension', 'arm_angle', 'max_velo', 'active_spin_fastball']
PITCH_CHAR_FEATURES = ['release_speed', 'pfx_x', 'pfx_z']

# Pitch-break display helpers. Statcast pfx_x/pfx_z are in feet from the catcher's
# (batter's) perspective. Pitch plots show break in inches from the pitcher's
# perspective, so horizontal break is scaled to inches and sign-flipped, while
# vertical break is only scaled to inches.
FT_TO_IN = 12.0

def hb_in(pfx_x):
    """Horizontal break in inches, from the pitcher's perspective."""
    return -pfx_x * FT_TO_IN

def vb_in(pfx_z):
    """Induced vertical break in inches."""
    return pfx_z * FT_TO_IN

PITCH_FULL_NAMES = {
    'FF': 'Four-Seam Fastball',
    'SI': 'Sinker',
    'FC': 'Cutter',
    'SL': 'Slider',
    'ST': 'Sweeper',
    'CU': 'Curveball',
    'CH': 'Changeup',
    'KC': 'Knuckle Curve',
    'CS': 'Slow Curve',
    'FS': 'Splitter',
    'FA': 'Fastball',
    'SV': 'Slurve',
    'FO': 'Other',
    'KN': 'Knuckleball',
    'EP': 'Eephus',
    'SC': 'Screwball',
    'PO': 'Pitch Out',
}

def _full_name(abbrev):
    return PITCH_FULL_NAMES.get(abbrev, abbrev)


def _find_target(pitcher_summ, target_pitcher_id):
    """Returns (target_row, target_year) or (None, None) if the pitcher id is not found.

    Identity is the unique `pitcher` id (not `player_name`), so the most recent
    season for that id anchors the suggestion.
    """
    rows = pitcher_summ[pitcher_summ['pitcher'] == target_pitcher_id]
    if rows.empty:
        return None, None
    row = rows.loc[rows['game_year'].idxmax()]
    return row, row['game_year']


def _find_biomech_comps(pitcher_summ, target_pitcher_id, target_year,
                        biomech_features, biomech_distance_threshold, min_pitches):
    """
    Returns a DataFrame (comp_pitcher, comp_year, distance) of biomechanically
    similar pitchers (`comp_pitcher` is a `pitcher` id), deduplicated to the
    closest year per comp.
    """
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['pitcher', 'game_year'],
        min_pitches=min_pitches,
    )
    target_mask = (
        ((biomech_dist['pitcher1'] == target_pitcher_id) & (biomech_dist['game_year1'] == target_year)) |
        ((biomech_dist['pitcher2'] == target_pitcher_id) & (biomech_dist['game_year2'] == target_year))
    )
    dists   = biomech_dist[target_mask].copy()
    is_left = dists['pitcher1'] == target_pitcher_id
    dists['comp_pitcher'] = np.where(is_left, dists['pitcher2'], dists['pitcher1'])
    dists['comp_year']    = np.where(is_left, dists['game_year2'], dists['game_year1'])
    return (
        dists[['comp_pitcher', 'comp_year', 'distance']]
        .query('distance <= @biomech_distance_threshold and comp_pitcher != @target_pitcher_id')
        .sort_values('distance')
        .drop_duplicates(subset='comp_pitcher', keep='first')
        .reset_index(drop=True)
    )


def _collect_pitches(pitch_type_summ, target_pitcher_id, target_year, target_dists,
                     pitch_features, min_comp_usage_pct, min_pitches):
    """Returns (target_pitches, comp_pitches) with usage filtering applied to comp_pitches."""
    target_pitches = (
        pitch_type_summ[
            (pitch_type_summ['pitcher']   == target_pitcher_id) &
            (pitch_type_summ['game_year'] == target_year)
        ]
        .dropna(subset=pitch_features)
        .copy()
        .reset_index(drop=True)
    )
    comp_year_keys = target_dists[['comp_pitcher', 'comp_year']].rename(
        columns={'comp_pitcher': 'pitcher', 'comp_year': 'game_year'}
    )
    comp_pitches = (
        pitch_type_summ
        .merge(comp_year_keys, on=['pitcher', 'game_year'], how='inner')
        .dropna(subset=pitch_features)
        .copy()
    )
    totals = comp_pitches.groupby('pitcher')['n'].sum().rename('total_n')
    comp_pitches = comp_pitches.merge(totals, on='pitcher')
    comp_pitches['usage_pct'] = comp_pitches['n'] / comp_pitches['total_n']
    comp_pitches = comp_pitches[
        (comp_pitches['usage_pct'] >= min_comp_usage_pct) &
        (comp_pitches['n'] >= min_pitches)
    ]
    return target_pitches, comp_pitches


def _tag_novelty(target_pitches, comp_pitches, pitch_features, novelty_distance_threshold,
                 global_scaler):
    """
    Uses global_scaler to transform pitches, tags each comp pitch with its minimum distance
    to any target pitch, and returns (comp_pitches_with_dist_cols, novel_subset).
    """
    X_target = global_scaler.transform(target_pitches[pitch_features].values)
    X_comp   = global_scaler.transform(comp_pitches[pitch_features].values)

    dist_matrix = cdist(X_comp, X_target, metric='euclidean')
    closest_idx = dist_matrix.argmin(axis=1)

    comp_pitches = comp_pitches.copy().reset_index(drop=True)
    comp_pitches['min_dist_to_target']   = dist_matrix.min(axis=1)
    comp_pitches['closest_target_pitch'] = target_pitches['pitch_type'].iloc[closest_idx].values

    novel = comp_pitches[comp_pitches['min_dist_to_target'] >= novelty_distance_threshold].copy()
    return comp_pitches, novel


def _trim_cluster_outliers(novel, X_novel, mask=True, mad_multiplier=3):
    """
    Removes points more than mad_multiplier*MAD from their cluster centroid.
    Writes _dist_to_centroid, _cluster_median_dist, _cluster_mad, _outlier_threshold
    onto novel (surviving rows only) so the caller can inspect and tune mad_multiplier.
    Returns (novel, X_novel, trimmed_any).
    """
    novel       = novel.copy()
    keep_mask   = np.ones(len(novel), dtype=bool)
    trimmed_any = False

    dist_to_centroid   = np.empty(len(novel))
    cluster_median     = np.empty(len(novel))
    cluster_mad_vals   = np.empty(len(novel))
    outlier_thresholds = np.empty(len(novel))

    for cid in novel['cluster'].unique():
        cluster_mask = novel['cluster'].values == cid
        X_clust      = X_novel[cluster_mask]
        centroid     = X_clust.mean(axis=0)
        dists        = np.linalg.norm(X_clust - centroid, axis=1)
        median_d     = np.median(dists)
        mad          = np.median(np.abs(dists - median_d))
        threshold    = np.max([median_d + mad_multiplier * mad, 1])
        outliers     = dists > threshold
        if outliers.any() and mask:
            trimmed_any = True
            keep_mask[np.where(cluster_mask)[0][outliers]] = False

        dist_to_centroid[cluster_mask]   = dists
        cluster_median[cluster_mask]     = median_d
        cluster_mad_vals[cluster_mask]   = mad
        outlier_thresholds[cluster_mask] = threshold

    novel['_dist_to_centroid']   = dist_to_centroid
    novel['_cluster_median_dist'] = cluster_median
    novel['_cluster_mad']         = cluster_mad_vals
    novel['_outlier_threshold']   = outlier_thresholds

    novel   = novel[keep_mask].copy()
    X_novel = X_novel[keep_mask]
    return novel, X_novel, trimmed_any


def _cluster_novel(novel, scaler, pitch_features, mad_multiplier=3, mask=True, **kwargs):
    """
    Iteratively clusters novel pitches (best silhouette k), drops low-cohesion clusters,
    and trims centroid outliers until stable. Returns novel with cluster and _sil columns.
    mad_multiplier controls the outlier threshold passed to _trim_cluster_outliers.
    """
    X_novel    = scaler.transform(novel[pitch_features].values)
    MAX_ITERATIONS = 5
    for _ in range(MAX_ITERATIONS):
        best_k, best_score, best_labels = 1, -1, np.zeros(len(novel), dtype=int)
        for k in range(2, min(9, len(novel))):
            labels = KMeans(n_clusters=k, random_state=0, n_init='auto').fit_predict(X_novel)
            score  = silhouette_score(X_novel, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        novel['cluster'] = best_labels
        novel   = novel.reset_index(drop=True)
        X_novel = scaler.transform(novel[pitch_features].values)
        novel, X_novel, trimmed_any = _trim_cluster_outliers(novel, X_novel, mask=mask, mad_multiplier=mad_multiplier)

        if not trimmed_any:
            break
    
    cluster_labels = (
        novel.groupby('cluster')['pitch_type']
        .agg(lambda x: x.value_counts().index[0])
        .rename('cluster_label')
    )
    novel = novel.join(cluster_labels, on='cluster')

    return novel


def _build_suggestions(novel, target_dists):
    """Aggregates clustered novel pitches into a suggestions DataFrame."""

    dist_lookup              = target_dists.set_index('comp_pitcher')['distance']
    novel['biomech_distance'] = novel['pitcher'].map(dist_lookup)
    novel['sim_weight']       = 1 / (novel['biomech_distance'] + 1e-6)

    def summarise(grp):
        total_sim = grp['sim_weight'].sum()
        return pd.Series({
            'n_comps':                grp['pitcher'].nunique(),
            'avg_release_speed':      round(grp['release_speed'].mean(), 1),
            'avg_pfx_x':              round(grp['pfx_x'].mean(), 2),
            'avg_pfx_z':              round(grp['pfx_z'].mean(), 2),
            'avg_min_dist_to_target': round(grp['min_dist_to_target'].mean(), 2),
            'wavg_release_speed':     round((grp['release_speed'] * grp['sim_weight']).sum() / total_sim, 1),
            'wavg_pfx_x':             round((grp['pfx_x'] * grp['sim_weight']).sum() / total_sim, 2),
            'wavg_pfx_z':             round((grp['pfx_z'] * grp['sim_weight']).sum() / total_sim, 2),
            'pitch_types_in_cluster': ', '.join(_full_name(p) for p in sorted(grp['pitch_type'].unique())),
            'comp_pitchers':          ', '.join(sorted(grp['player_name'].unique())),
        })

    result = (
        novel.groupby(['cluster_label', 'cluster'])
        .apply(summarise, include_groups=False)
        .reset_index()
        .sort_values('n_comps', ascending=False)
        .reset_index(drop=True)
    )
    result['cluster_label'] = result['cluster_label'].apply(lambda x: _full_name(x) + '*')
    return result


# Suggest pitches for a target pitcher based on biomechanical similarity to comps and novelty of pitch characteristics
def suggest_pitches(
    target_pitcher_id,
    pitcher_summ,
    pitch_type_summ,
    biomech_distance_threshold=2.0,
    novelty_distance_threshold=1.5,
    min_comp_usage_pct=0.05,
    min_pitches=20,
    biomech_features=BIOMECH_FEATURES,
    pitch_features=PITCH_CHAR_FEATURES,
    **kwargs,  # forwarded to _cluster_novel
):
    """`target_pitcher_id` is the `pitcher` id (not the player name)."""
    target_row, target_year = _find_target(pitcher_summ, target_pitcher_id)
    if target_row is None:
        return {
            'status':         'pitcher_not_found',
            'target_info':    None,
            'comps':          pd.DataFrame(),
            'comp_pitches':   pd.DataFrame(),
            'suggestions':    pd.DataFrame(),
            'target_pitches': pd.DataFrame(),
        }

    target_dists = _find_biomech_comps(
        pitcher_summ, target_pitcher_id, target_year,
        biomech_features, biomech_distance_threshold, min_pitches,
    )
    if target_dists.empty:
        return {
            'status':         'no_comps',
            'target_info':    target_row,
            'comps':          target_dists,
            'comp_pitches':   pd.DataFrame(),
            'suggestions':    pd.DataFrame(),
            'target_pitches': pd.DataFrame(),
        }

    target_pitches, comp_pitches = _collect_pitches(
        pitch_type_summ, target_pitcher_id, target_year, target_dists,
        pitch_features, min_comp_usage_pct, min_pitches,
    )
    if comp_pitches.empty:
        return {
            'status':         'no_comp_pitches',
            'target_info':    target_row,
            'comps':          target_dists,
            'comp_pitches':   pd.DataFrame(),
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    global_scaler = StandardScaler().fit(
        pitch_type_summ[pitch_features].dropna().values
    )
    comp_pitches, novel = _tag_novelty(
        target_pitches, comp_pitches, pitch_features, novelty_distance_threshold,
        global_scaler,
    )
    if len(novel) < 4:
        return {
            'status':         'no_novel_pitches',
            'target_info':    target_row,
            'comps':          target_dists,
            'comp_pitches':   novel,
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    novel = _cluster_novel(novel, global_scaler, pitch_features, **kwargs)
    if novel.empty:
        return {
            'status':         'no_novel_pitches',
            'target_info':    target_row,
            'comps':          novel,
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    suggestions = _build_suggestions(novel, target_dists)
    return {
        'status':         'ok',
        'target_info':    target_row,
        'comps':          target_dists,
        'comp_pitches':   novel,
        'suggestions':    suggestions,
        'target_pitches': target_pitches,
    }


def plot_pitch_clusters(result):
    """
    Scatter plot of comp pitches in pfx_x / pfx_z space.
    Color encodes release_speed, marker shape encodes cluster.
    Cluster centroids are overlaid as large markers.
    Target pitches are overlaid in grey, labeled by pitch_type.

    Parameters
    ----------
    result         : dict returned by suggest_pitches (must include 'comp_pitches', 'target_pitches')
    """
    comp_pitches   = result['comp_pitches']
    target_pitches = result['target_pitches']

    pitcher_name = result['target_pitches']['player_name'][0]

    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h']
    cluster_keys = sorted(comp_pitches[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None))

    vmin = comp_pitches['release_speed'].min()
    vmax = comp_pitches['release_speed'].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.plasma

    fig, ax = plt.subplots(figsize=(8, 6))

    # ── Comp pitches: color = velocity, shape = cluster ───────────────────
    for i, (label, cid) in enumerate(cluster_keys):
        grp    = comp_pitches[(comp_pitches['cluster_label'] == label) & (comp_pitches['cluster'] == cid)]
        marker = markers[i % len(markers)]
        sc = ax.scatter(
            hb_in(grp['pfx_x']), vb_in(grp['pfx_z']),
            c=grp['release_speed'], cmap=cmap, norm=norm,
            marker=marker, s=60, alpha=0.7, zorder=2,
        )

    # ── Cluster centroids ─────────────────────────────────────────────────
    centroids = comp_pitches.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean()
    for i, ((label, cid), row) in enumerate(centroids.iterrows()):
        marker = markers[i % len(markers)]
        ax.scatter(
            hb_in(row['pfx_x']), vb_in(row['pfx_z']),
            c=[[cmap(norm(row['release_speed']))]],
            marker=marker, s=250, zorder=4,
            edgecolors='black', linewidths=1.5,
        )

    # ── Target pitches ────────────────────────────────────────────────────
    if target_pitches is not None and not target_pitches.empty:
        first = True
        for pitch_type, grp in target_pitches.groupby('pitch_type'):
            ax.scatter(
                hb_in(grp['pfx_x']), vb_in(grp['pfx_z']),
                label='Existing Pitch' if first else '_nolegend_',
                color='black', s=80, zorder=3, marker='D',
            )
            for _, row in grp.iterrows():
                ax.annotate(
                    _full_name(pitch_type), (hb_in(row['pfx_x']), vb_in(row['pfx_z'])),
                    textcoords='offset points', xytext=(5, 5), fontsize=7,
                )
            first = False

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = []
    for i, (label, cid) in enumerate(cluster_keys):
        legend_handles.append(
            mlines.Line2D([], [], color='grey', marker=markers[i % len(markers)],
                        linestyle='None', markersize=7, label=_full_name(label))
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

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, label='Velocity (mph)',
    )
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlim(-24, 24)
    ax.set_ylim(-24, 24)
    ax.set_xlabel('Horizontal Break (in)')
    ax.set_ylabel('Induced Vertical Break (in)')
    ax.set_title(f'Potential Pitch Plot — {pitcher_name}')
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()
