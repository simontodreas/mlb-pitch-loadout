from distances import compute_euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import silhouette_samples


# Constants
BIOMECH_FEATURES    = ['release_extension', 'arm_angle', 'max_velo', 'active_spin_fastball']
PITCH_CHAR_FEATURES = ['release_speed', 'pfx_x', 'pfx_z']

# Suggest pitches for a target pitcher based on biomechanical similarity to comps and novelty of pitch characteristics
def suggest_pitches(
    target_pitcher,
    pitcher_summ,
    pitch_type_summ,
    biomech_distance_threshold=2.0,
    novelty_distance_threshold=1.5,
    min_comp_usage_pct=0.05,
    min_pitches=20,
    biomech_features=BIOMECH_FEATURES,
    pitch_features=PITCH_CHAR_FEATURES,
):
    # ── 1. Identify target's most recent year ─────────────────────────────────
    # CHANGED: filter to most recent game_year instead of just player_name
    target_rows = pitcher_summ[pitcher_summ['player_name'] == target_pitcher]
    if target_rows.empty:
        return {
            'status':         'pitcher_not_found',
            'target_info':    None,
            'comps':          pd.DataFrame(),
            'comp_pitches':   pd.DataFrame(),
            'suggestions':    pd.DataFrame(),
            'target_pitches': pd.DataFrame(),
        }
    target_year = target_rows['game_year'].max()
    target_row  = target_rows.loc[target_rows['game_year'].idxmax()]

    # ── 2. Biomechanical distances on full multi-year pitcher_summ ────────────
    # CHANGED: label_cols now includes game_year so each row is a pitcher-year
    # pair, making the distance matrix unambiguous
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )

    # ── 3. Filter to rows involving the target's most recent year only ─────────
    # CHANGED: match on both player_name and game_year to exclude the target's
    # own prior years and anchor distances to the current version of the pitcher
    target_mask = (
        (
            (biomech_dist['player_name1'] == target_pitcher) &
            (biomech_dist['game_year1']   == target_year)
        ) | (
            (biomech_dist['player_name2'] == target_pitcher) &
            (biomech_dist['game_year2']   == target_year)
        )
    )
    target_dists = biomech_dist[target_mask].copy()

    # Normalise so comp is always in comp_pitcher / comp_year columns
    is_left = (target_dists['player_name1'] == target_pitcher)
    target_dists['comp_pitcher'] = np.where(
        is_left, target_dists['player_name2'], target_dists['player_name1']
    )
    target_dists['comp_year'] = np.where(
        is_left, target_dists['game_year2'], target_dists['game_year1']
    )
    target_dists = (
        target_dists[['comp_pitcher', 'comp_year', 'distance']]
        .query('distance <= @biomech_distance_threshold')
        .reset_index(drop=True)
    )

    # ── 4. Deduplicate comps: keep the year closest to the target ─────────────
    # CHANGED: a comp pitcher may appear in multiple years; we keep only the
    # year with the smallest biomechanical distance to the current target
    target_dists = (
        target_dists
        .sort_values('distance')
        .drop_duplicates(subset='comp_pitcher', keep='first')
        .reset_index(drop=True)
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

    # ── 5. Collect pitches using year-specific lookups ────────────────────────
    # CHANGED: target pitches use target_year only; comp pitches use each
    # comp's best year from the lookup built in step 4

    # Target pitches: most recent year only
    target_pitches = (
        pitch_type_summ[
            (pitch_type_summ['player_name'] == target_pitcher) &
            (pitch_type_summ['game_year']   == target_year)
        ]
        .dropna(subset=pitch_features)
        .copy()
        .reset_index(drop=True)
    )

    # Comp pitches: merge on (player_name, game_year) to get each comp's best year
    comp_year_keys = target_dists[['comp_pitcher', 'comp_year']].rename(
        columns={'comp_pitcher': 'player_name', 'comp_year': 'game_year'}
    )
    comp_pitches = (
        pitch_type_summ
        .merge(comp_year_keys, on=['player_name', 'game_year'], how='inner')
        .dropna(subset=pitch_features)
        .copy()
    )

    # ── Steps 4–7: unchanged from original ───────────────────────────────────
    totals = comp_pitches.groupby('player_name')['n'].sum().rename('total_n')
    comp_pitches = comp_pitches.merge(totals, on='player_name')
    comp_pitches['usage_pct'] = comp_pitches['n'] / comp_pitches['total_n']
    comp_pitches = comp_pitches[
        (comp_pitches['usage_pct'] >= min_comp_usage_pct) &
        (comp_pitches['n'] >= min_pitches)
    ]

    if comp_pitches.empty:
        return {
            'status':         'no_comp_pitches',
            'target_info':    target_row,
            'comps':          target_dists,
            'comp_pitches':   pd.DataFrame(),
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    all_pitches  = pd.concat([target_pitches, comp_pitches], ignore_index=True)
    scaler       = StandardScaler().fit(all_pitches[pitch_features])
    X_target     = scaler.transform(target_pitches[pitch_features].values)
    X_comp       = scaler.transform(comp_pitches[pitch_features].values)

    dist_matrix  = cdist(X_comp, X_target, metric='euclidean')
    closest_idx  = dist_matrix.argmin(axis=1)

    comp_pitches = comp_pitches.copy().reset_index(drop=True)
    comp_pitches['min_dist_to_target']   = dist_matrix.min(axis=1)
    comp_pitches['closest_target_pitch'] = target_pitches['pitch_type'].iloc[closest_idx].values

    novel = comp_pitches[
        comp_pitches['min_dist_to_target'] >= novelty_distance_threshold
    ].copy()

    if len(novel) < 4:
        return {
            'status':         'no_novel_pitches',
            'target_info':    target_row,
            'comps':          target_dists,
            'comp_pitches':   novel,
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    X_novel = scaler.transform(novel[pitch_features].values)

    # ── Cluster novel pitches, trimming centroid outliers, then re-cluster ────
    MAX_ITERATIONS = 5
    for _ in range(MAX_ITERATIONS):
        best_k, best_score, best_labels = 1, -1, np.zeros(len(novel), dtype=int)
        for k in range(2, min(9, len(novel))):
            labels = KMeans(n_clusters=k, random_state=0, n_init='auto').fit_predict(X_novel)
            score  = silhouette_score(X_novel, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        novel = novel.copy().reset_index(drop=True)
        novel['cluster'] = best_labels

        # ── Drop low-cohesion clusters ────────────────────────────────────────
        if best_k > 1:
            sample_scores = silhouette_samples(X_novel, best_labels)
            novel['_sil'] = sample_scores
            cluster_mean_sil = novel.groupby('cluster')['_sil'].mean()
            MIN_CLUSTER_SIL = 0
            keep_clusters = cluster_mean_sil[cluster_mean_sil >= MIN_CLUSTER_SIL].index
            novel = novel[novel['cluster'].isin(keep_clusters)].copy()
        else:
            novel['_sil'] = 0.0

        # ── Trim centroid outliers within each cluster ────────────────────────
        novel = novel.reset_index(drop=True)
        X_novel = scaler.transform(novel[pitch_features].values)
        trimmed_any = False
        keep_mask = np.ones(len(novel), dtype=bool)

        for cid in novel['cluster'].unique():
            mask = novel['cluster'].values == cid
            X_clust = X_novel[mask]
            centroid = X_clust.mean(axis=0)
            dists = np.linalg.norm(X_clust - centroid, axis=1)
            median_dist = np.median(dists)
            mad = np.median(np.abs(dists - median_dist))
            threshold = median_dist + 3 * mad
            outlier_mask = dists > threshold
            if outlier_mask.any():
                trimmed_any = True
                global_indices = np.where(mask)[0]
                keep_mask[global_indices[outlier_mask]] = False

        novel = novel[keep_mask].copy()
        X_novel = scaler.transform(novel[pitch_features].values)

        if not trimmed_any:
            break

    if novel.empty:
        return {
            'status':         'no_novel_pitches',
            'target_info':    target_row,
            'comps':          novel,
            'suggestions':    pd.DataFrame(),
            'target_pitches': target_pitches,
        }

    cluster_labels = (
        novel.groupby('cluster')['pitch_type']
        .agg(lambda x: x.value_counts().index[0])
        .rename('cluster_label')
    )
    novel = novel.join(cluster_labels, on='cluster')

    dist_lookup           = target_dists.set_index('comp_pitcher')['distance']
    novel['biomech_distance'] = novel['player_name'].map(dist_lookup)
    novel['sim_weight']       = 1 / (novel['biomech_distance'] + 1e-6)

    def summarise(grp):
        total_sim = grp['sim_weight'].sum()
        return pd.Series({
            'n_comps':                len(grp['player_name'].unique()),
            'avg_release_speed':      round(grp['release_speed'].mean(), 1),
            'avg_pfx_x':              round(grp['pfx_x'].mean(), 2),
            'avg_pfx_z':              round(grp['pfx_z'].mean(), 2),
            'avg_min_dist_to_target': round(grp['min_dist_to_target'].mean(), 2),
            'wavg_release_speed':     round((grp['release_speed'] * grp['sim_weight']).sum() / total_sim, 1),
            'wavg_pfx_x':             round((grp['pfx_x'] * grp['sim_weight']).sum() / total_sim, 2),
            'wavg_pfx_z':             round((grp['pfx_z'] * grp['sim_weight']).sum() / total_sim, 2),
            '_sil':                   round(grp['_sil'].mean(), 3),
            'pitch_types_in_cluster': ', '.join(sorted(grp['pitch_type'].unique())),
            'comp_pitchers':          ', '.join(sorted(grp['player_name'].unique())),
        })

    suggestions = (
        novel.groupby(['cluster_label', 'cluster'])
        .apply(summarise, include_groups=False)
        .reset_index()
        .sort_values('n_comps', ascending=False)
        .reset_index(drop=True)
    )

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
            grp['pfx_x'], grp['pfx_z'],
            c=grp['release_speed'], cmap=cmap, norm=norm,
            marker=marker, s=60, alpha=0.7, zorder=2,
        )

    # ── Cluster centroids ─────────────────────────────────────────────────
    centroids = comp_pitches.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean()
    for i, ((label, cid), row) in enumerate(centroids.iterrows()):
        marker = markers[i % len(markers)]
        ax.scatter(
            row['pfx_x'], row['pfx_z'],
            c=[[cmap(norm(row['release_speed']))]],
            marker=marker, s=250, zorder=4,
            edgecolors='black', linewidths=1.5,
        )

    # ── Target pitches ────────────────────────────────────────────────────
    if target_pitches is not None and not target_pitches.empty:
        first = True
        for pt, grp in target_pitches.groupby('pitch_type'):
            ax.scatter(
                grp['pfx_x'], grp['pfx_z'],
                label='Existing Pitch' if first else '_nolegend_',
                color='black', s=80, zorder=3, marker='D',
            )
            first = False

    # ── Legend ────────────────────────────────────────────────────────────
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
    

    plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, label='Release speed (mph)',
    )
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Horizontal Break (ft)')
    ax.set_ylabel('Induced Vertical Break (ft)')
    ax.set_title(f'Pitch Recommendations — {pitcher_name}')
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.25, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()


def run_suggest_pitches_bulk(
    pitcher_summ,              # R or L
    pitch_type_summ,
    min_pitches=20,
    **kwargs,                  # forwarded to suggest_pitches
):
    """
    Run suggest_pitches for every qualifying pitcher and return:
        suggestions_df : flat DataFrame of all suggestions, with target_pitcher column
        diagnostics_df : one row per pitcher with status and basic counts
    """
    # build pool from that
    pool = pitcher_summ[(pitcher_summ['n'] >= min_pitches) & (pitcher_summ['game_year'] == 2025)]

    all_suggestions = []
    diag_rows       = []

    for name in pool['player_name']:
        try:
            result = suggest_pitches(
                name, pitcher_summ, pitch_type_summ,
                min_pitches=min_pitches, **kwargs
            )
            status    = result['status']
            n_comps   = len(result['comps'])
            n_suggest = len(result['suggestions'])

            if status == 'ok' and n_suggest > 0:
                sdf = result['suggestions'].copy()
                sdf.insert(0, 'target_pitcher', name)
                all_suggestions.append(sdf)

        except Exception as e:
            status    = f'exception: {e}'
            n_comps   = 0
            n_suggest = 0

        diag_rows.append({
            'target_pitcher': name,
            'status':         status,
            'n_comps':        n_comps,
            'n_suggestions':  n_suggest,
        })

    suggestions_df = (
        pd.concat(all_suggestions, ignore_index=True)
        if all_suggestions else pd.DataFrame()
    )
    diagnostics_df = pd.DataFrame(diag_rows)

    return suggestions_df, diagnostics_df