from distances import compute_euclidean_distances
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import plotly.graph_objects as go


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


def _pick_cluster_threshold(dists, floor):
    """
    Adaptive per-cluster outlier threshold.

    Walks median+3*mad -> median+2*mad -> median+1*mad, taking the first band that is
    strictly above `floor` AND actually flags at least one point as an outlier. Once a band
    would fall to/below the floor, the additive cascade stops and the floor itself is used
    (the floor is a hard minimum radius: points closer than `floor` to the centroid are never
    trimmed). Returns the chosen threshold.
    """
    median_d = np.median(dists)
    mad      = np.median(np.abs(dists - median_d))
    for mult in (3, 2, 1):
        threshold = median_d + mult * mad
        if threshold <= floor:
            break                              # this + tighter bands are below the floor
        if (dists > threshold).any():
            return threshold                   # most lenient band that flags the worst outliers
    return floor


def _trim_cluster_outliers(novel, X_novel, mask=True, floor=1.2):
    """
    Removes points beyond an adaptive per-cluster threshold (see _pick_cluster_threshold):
    the most lenient of median+{3,2,1}*MAD that stays above `floor` and flags outliers, else
    the `floor` itself. Because outliers drag the centroid, each call trims the worst outliers
    and the caller re-centers and re-trims until a pass removes nothing.
    Writes _dist_to_centroid, _cluster_median_dist, _cluster_mad, _outlier_threshold
    onto novel (surviving rows only) so the caller can inspect the trimming.
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
        threshold    = _pick_cluster_threshold(dists, floor)
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


def _cluster_novel(novel, scaler, pitch_features, floor=1.2, mask=True, **kwargs):
    """
    Iteratively clusters novel pitches (best silhouette k) and trims centroid outliers,
    re-centering after each trim, until a pass removes nothing. Returns novel with cluster
    and cluster_label columns. `floor` is the hard minimum outlier radius passed to
    _trim_cluster_outliers.
    """
    X_novel = scaler.transform(novel[pitch_features].values)
    MAX_ITERATIONS = 25
    for _ in range(MAX_ITERATIONS):
        best_k, best_score, best_labels = 1, -1, np.zeros(len(novel), dtype=int)
        for k in range(2, min(5, len(novel))):
            labels = KMeans(n_clusters=k, random_state=0, n_init='auto').fit_predict(X_novel)
            score  = silhouette_score(X_novel, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        novel['cluster'] = best_labels
        novel   = novel.reset_index(drop=True)
        X_novel = scaler.transform(novel[pitch_features].values)
        novel, X_novel, trimmed_any = _trim_cluster_outliers(novel, X_novel, mask=mask, floor=floor)

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
    result['cluster_label'] = result['cluster_label'].apply(lambda x: _full_name(x))
    return result


# Suggest pitches for a target pitcher based on biomechanical similarity to comps and novelty of pitch characteristics
def suggest_pitches(
    target_pitcher_id,
    pitcher_summ,
    pitch_type_summ,
    biomech_distance_threshold=1.5,
    novelty_distance_threshold=1.2,
    min_comp_usage_pct=0.01,
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
            'comps':          target_dists,
            'comp_pitches':   novel,
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


def make_cluster_fig(result, is_righty, vmin=None, vmax=None, show_existing=True, show_suggested=True):
    """The app's Pitcher-View break plot. Single source of truth: the Streamlit
    app, plot_pitch_clusters, and the validation overlay all render through here.

    vmin/vmax fix the velocity color band (the app passes a league-wide band so
    colors compare across pitchers); left as None they span this result's pitches.
    """
    comp_pitches = result.get('comp_pitches')
    if comp_pitches is None or 'cluster_label' not in comp_pitches.columns:
        # No-suggestion statuses (no novel pitches / too few to cluster) still
        # render the plot — existing pitches only.
        comp_pitches = pd.DataFrame({
            'cluster_label': pd.Series(dtype=str),
            'cluster':       pd.Series(dtype=int),
            'pfx_x':         pd.Series(dtype=float),
            'pfx_z':         pd.Series(dtype=float),
            'release_speed': pd.Series(dtype=float),
            'player_name':   pd.Series(dtype=str),
            'pitch_type':    pd.Series(dtype=str),
        })
    else:
        comp_pitches = comp_pitches.reset_index(drop=True)
    if not show_suggested:
        # Emptying the frame drops every comp/suggested trace (scatter + centroids)
        # since they are all derived from it.
        comp_pitches = comp_pitches.iloc[0:0]
    target_pitches = result['target_pitches']
    pitcher_name   = target_pitches['player_name'].iloc[0]

    if vmin is None or vmax is None:
        speeds = pd.concat([comp_pitches['release_speed'], target_pitches['release_speed']])
        vmin = float(speeds.min()) if vmin is None else vmin
        vmax = float(speeds.max()) if vmax is None else vmax

    plotly_markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
                      'x', 'triangle-down', 'triangle-left', 'triangle-right', 'hexagon']
    cluster_keys = sorted(
        comp_pitches[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None)
    )
    cluster_key_index = {(label, cid): idx for idx, (label, cid) in enumerate(cluster_keys)}

    arm_angle_deg  = result['target_info']['arm_angle']
    arm_angle_rad  = np.radians(arm_angle_deg)

    fig = go.Figure()

    for i, (label, cid) in enumerate(cluster_keys):
        grp = comp_pitches[(comp_pitches['cluster_label'] == label) & (comp_pitches['cluster'] == cid)]
        fig.add_trace(go.Scatter(
            x=hb_in(grp['pfx_x']),
            y=vb_in(grp['pfx_z']),
            mode='markers',
            name=f'Comparison Pitch ({_full_name(label)})',
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
                    # almost the entire bottom half of the right column; the
                    # legend keeps the space above it
                    len=0.48,
                    y=0,
                    yanchor='bottom',
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
        cx, cy = hb_in(row['pfx_x']), vb_in(row['pfx_z'])
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            mode='markers+text',
            name='Suggested Pitch',
            showlegend=(idx == 0),
            legendgroup='centroid',
            text=[f'<i>{_wrap_label(_full_name(label))}</i>'],
            textposition=[_label_position(cx, cy)],
            textfont=dict(size=16, color='#555'),
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
                f'<b>Suggested: {_full_name(label)}</b><br>'
                'HBreak: %{x:.1f} in<br>'
                'IVBreak: %{y:.1f} in'
                '<extra></extra>'
            ),
        ))

    # The first comp trace normally carries the colorbar; the existing-pitch
    # trace takes it over when no comp traces are drawn.
    existing_carries_scale = show_suggested is False or comp_pitches.empty

    if show_existing and target_pitches is not None and not target_pitches.empty:
        fig.add_trace(go.Scatter(
            x=hb_in(target_pitches['pfx_x']),
            y=vb_in(target_pitches['pfx_z']),
            mode='markers+text',
            name='Existing Pitch',
            marker=dict(
                symbol='diamond',
                size=18,
                color=target_pitches['release_speed'],
                colorscale='plasma',
                cmin=vmin,
                cmax=vmax,
                line=dict(color='black', width=3),
                showscale=existing_carries_scale,
                colorbar=dict(
                    title=dict(text='Velocity (mph)', side='right'),
                    x=1.02,
                    thickness=15,
                    # almost the entire bottom half of the right column; the
                    # legend keeps the space above it
                    len=0.48,
                    y=0,
                    yanchor='bottom',
                ) if existing_carries_scale else None,
            ),
            text=[f'<b>{_wrap_label(_full_name(pt))}</b>' for pt in target_pitches['pitch_type']],
            textposition=[_label_position(x, z) for x, z in zip(hb_in(target_pitches['pfx_x']), vb_in(target_pitches['pfx_z']))],
            textfont=dict(size=16, color='black'),
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
        title=dict(text=f'Potential Arsenal — {pitcher_name}<br><sup>Pitcher View</sup>', x=0.5, xanchor='center', font=dict(size=24)),
        xaxis_title='Horizontal Break (in)',
        yaxis_title='Induced Vertical Break (in)',
        # constraintoward='right' packs the square plot against the legend
        # instead of centering it with dead space on both sides.
        xaxis=dict(**grid_style, constraintoward='right'),
        yaxis=dict(**grid_style, scaleanchor='x', scaleratio=1),
        dragmode='select',
        legend=dict(x=1.02, y=1, xanchor='left', font=dict(size=16)),
        height=640,
        margin=dict(r=200),
    )
    return fig


def plot_pitch_clusters(result):
    """The app's Pitcher-View break plot for a suggest_pitches result.

    Thin wrapper over make_cluster_fig: handedness comes from the result and the
    velocity color band spans this result's pitches (the app instead passes a
    league-wide band). Returns the plotly figure.
    """
    is_righty = result['target_info'].get('p_throws', 'R') == 'R'
    return make_cluster_fig(result, is_righty)
