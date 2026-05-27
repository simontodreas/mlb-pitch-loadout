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
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )

    targets = (
        pitcher_summ[pitcher_summ['n'] >= min_pitches]
        .sort_values('game_year', ascending=False)
        .drop_duplicates(subset='player_name')
        [['player_name', 'game_year']]
    )

    left = biomech_dist.merge(
        targets, left_on=['player_name1', 'game_year1'], right_on=['player_name', 'game_year']
    )[['player_name', 'game_year', 'player_name2', 'game_year2', 'distance']].rename(
        columns={'player_name2': 'comp_pitcher', 'game_year2': 'comp_year'}
    )

    right = biomech_dist.merge(
        targets, left_on=['player_name2', 'game_year2'], right_on=['player_name', 'game_year']
    )[['player_name', 'game_year', 'player_name1', 'game_year1', 'distance']].rename(
        columns={'player_name1': 'comp_pitcher', 'game_year1': 'comp_year'}
    )

    target_pairs = pd.concat([left, right], ignore_index=True)

    # Remove self-comparisons
    target_pairs = target_pairs[target_pairs['player_name'] != target_pairs['comp_pitcher']]

    # Deduplicate comp pitcher-years: keep only the closest year per comp,
    # mirroring the drop_duplicates logic in suggest_pitches
    target_pairs = (
        target_pairs
        .sort_values('distance')
        .drop_duplicates(subset=['player_name', 'game_year', 'comp_pitcher'])
        .reset_index(drop=True)
    )

    rows = []
    for threshold in thresholds:
        comp_counts = (
            target_pairs[target_pairs['distance'] <= threshold]
            .groupby(['player_name', 'game_year'])
            .size()
            .reindex(pd.MultiIndex.from_frame(targets), fill_value=0)
            .values
        )

        rows.append({
            'threshold':  threshold,
            'mean_comps': round(comp_counts.mean(), 1),
            'p10_comps':  int(np.percentile(comp_counts, 10)),
            'p25_comps':  int(np.percentile(comp_counts, 25)),
            'p50_comps':  int(np.percentile(comp_counts, 50)),
            'p75_comps':  int(np.percentile(comp_counts, 75)),
            'pct_zero':   round((comp_counts == 0).mean() * 100, 1),
            'pct_lt5':    round((comp_counts < 5).mean() * 100, 1),
        })

    df = pd.DataFrame(rows)
    print("── Biomech threshold coverage ──")
    print(df.to_string(index=False))
    return df