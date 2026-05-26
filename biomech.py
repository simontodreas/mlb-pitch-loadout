from distances import compute_euclidean_distances, compute_mahalanobis_distances
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Constants
BIOMECH_FEATURES = ['release_extension', 'arm_angle', 'max_velo', 'active_spin_fastball']


def evaluate_biomech_features(pitcher_summ, arsenal_comp, feature_sets, min_pitches=100, distance_fn=compute_euclidean_distances):
    """
    For each candidate feature set, compute biomechanical distances and correlate
    with arsenal distances. Returns a summary DataFrame ranked by Spearman correlation.
    
    Parameters:
        pitcher_summ  : pitcher-level summary DataFrame
        arsenal_comp  : arsenal distance DataFrame from compare_all_arsenals()
        feature_sets  : dict of {label: [feature columns]}
        min_pitches   : minimum pitches filter
    
    Returns:
        DataFrame with feature set label, Spearman rho, p-value, and n pairs
    """
    arsenal_both = pd.concat([
        arsenal_comp.rename(columns={'player_name1': 'p1', 'player_name2': 'p2'}),
        arsenal_comp.rename(columns={'player_name2': 'p1', 'player_name1': 'p2'})
    ])
    arsenal_lookup = arsenal_both.set_index(['p1', 'p2'])['arsenal_distance']

    results = []
    for label, features in feature_sets.items():
        biomech = distance_fn(
            pitcher_summ,
            features=features,
            label_cols=['player_name'],
            min_pitches=min_pitches
        )

        biomech['arsenal_distance'] = biomech.apply(
            lambda r: arsenal_lookup.get((r['player_name1'], r['player_name2']), np.nan), axis=1
        )
        biomech = biomech.dropna(subset=['arsenal_distance'])

        rho, pval = spearmanr(biomech['distance'], biomech['arsenal_distance'])
        results.append({
            'features': label,
            'spearman_rho': round(rho, 4),
            'p_value': pval,
            'n_pairs': len(biomech)
        })

    return pd.DataFrame(results).sort_values('spearman_rho').reset_index(drop=True)

def biomech_threshold_coverage(
    pitcher_summ,
    thresholds=(1.0, 1.5, 2.0),
    min_pitches=100,
    biomech_features=BIOMECH_FEATURES,
):
    """
    For each threshold, compute the distribution of comp counts across all
    pitchers. Helps pick a biomech_distance_threshold that gives reasonable
    coverage without pulling in dissimilar pitchers.

    Only the most recent year per pitcher is counted as a candidate target,
    but all pitcher-years are eligible as comps.
    """

    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )
    
    # One row per pitcher: most recent year only, as the target population
    targets = (
        pitcher_summ[pitcher_summ['n'] >= min_pitches]
        .sort_values('game_year', ascending=False)
        .drop_duplicates(subset='player_name')
        [['player_name', 'game_year']]
    )

    rows = []
    for threshold in thresholds:
        comp_counts = []
        for _, row in targets.iterrows():
            name, year = row['player_name'], row['game_year']
            mask = (
                (biomech_dist['player_name1'] == name) & (biomech_dist['game_year1'] == year)
            ) | (
                (biomech_dist['player_name2'] == name) & (biomech_dist['game_year2'] == year)
            )
            n_comps = (biomech_dist[mask]['distance'] <= threshold).sum()
            comp_counts.append(n_comps)

        comp_counts = np.array(comp_counts)

        rows.append({
            'threshold':    threshold,
            'mean_comps':   round(comp_counts.mean(), 1),
            'p10_comps':    int(np.percentile(comp_counts, 10)),
            'p25_comps':    int(np.percentile(comp_counts, 25)),
            'p50_comps':    int(np.percentile(comp_counts, 50)),
            'p75_comps':    int(np.percentile(comp_counts, 75)),
            'pct_zero':     round((comp_counts == 0).mean() * 100, 1),
            'pct_lt5':      round((comp_counts < 5).mean() * 100, 1),
        })

    df = pd.DataFrame(rows)
    print("── Biomech threshold coverage ──")
    print(df.to_string(index=False))
    return df