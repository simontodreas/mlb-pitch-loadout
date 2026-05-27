from distances import compute_euclidean_distances, compute_mahalanobis_distances
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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


def biomech_threshold_calibration(
    pitcher_summ,
    arsenal_comp,
    biomech_features=BIOMECH_FEATURES,
    min_pitches=100,
    n_bins=20,
    max_biomech_dist=None,
):
    """
    Bin pitcher pairs by biomechanical distance and compute mean/median arsenal
    distance within each bin. Helps calibrate a biomech threshold by showing
    where the biomech→arsenal signal holds vs. degrades.

    Parameters:
        pitcher_summ      : pitcher-level summary DataFrame
        arsenal_comp      : arsenal distance DataFrame from compare_all_arsenals()
        biomech_features  : list of biomechanical feature columns
        min_pitches       : minimum pitches filter passed to distance function
        n_bins            : number of equal-width bins for biomech distance
        max_biomech_dist  : if set, drop pairs with biomech distance above this value
                            before binning (trims the long right tail)

    Returns:
        bin_df : DataFrame with columns:
                   biomech_bin_mid  – bin midpoint
                   mean_arsenal     – mean arsenal distance in that bin
                   median_arsenal   – median arsenal distance in that bin
                   n_pairs          – number of pairs in the bin
    """
    biomech_dist = compute_euclidean_distances(
        pitcher_summ,
        features=biomech_features,
        label_cols=['player_name', 'game_year'],
        min_pitches=min_pitches,
    )

    arsenal_both = pd.concat([
        arsenal_comp[['player_name1', 'game_year1', 'player_name2', 'game_year2', 'arsenal_distance']],
        arsenal_comp.rename(columns={
            'player_name1': 'player_name2', 'game_year1': 'game_year2',
            'player_name2': 'player_name1', 'game_year2': 'game_year1',
        })[['player_name1', 'game_year1', 'player_name2', 'game_year2', 'arsenal_distance']],
    ])
    arsenal_lookup = arsenal_both.set_index(
        ['player_name1', 'game_year1', 'player_name2', 'game_year2']
    )['arsenal_distance']

    merged = biomech_dist.copy()
    merged['arsenal_distance'] = merged.apply(
        lambda r: arsenal_lookup.get(
            (r['player_name1'], r['game_year1'], r['player_name2'], r['game_year2']), np.nan
        ),
        axis=1,
    )
    merged = merged.dropna(subset=['arsenal_distance'])

    if max_biomech_dist is not None:
        merged = merged[merged['distance'] <= max_biomech_dist]

    merged['biomech_bin'] = pd.cut(merged['distance'], bins=n_bins)

    rows = []
    for bin_interval, group in merged.groupby('biomech_bin', observed=True):
        rows.append({
            'biomech_bin_mid': round(bin_interval.mid, 3),
            'mean_arsenal':    round(group['arsenal_distance'].mean(), 4),
            'median_arsenal':  round(group['arsenal_distance'].median(), 4),
            'n_pairs':         len(group),
        })

    bin_df = pd.DataFrame(rows)
    print("── Biomech threshold calibration ──")
    print(bin_df.to_string(index=False))
    return bin_df


def plot_threshold_calibration(bin_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(bin_df['biomech_bin_mid'], bin_df['mean_arsenal'],
             color='steelblue', lw=2, marker='o', ms=4, label='Mean')
    ax1.plot(bin_df['biomech_bin_mid'], bin_df['median_arsenal'],
             color='steelblue', lw=1.5, ls='--', marker='o', ms=3, alpha=0.6,
             label='Median')
    ax1.set_ylabel('Arsenal distance')
    ax1.legend(fontsize=9)
    ax1.set_title('Arsenal distance vs. biomechanical distance bin')

    ax2.bar(bin_df['biomech_bin_mid'], bin_df['n_pairs'],
            width=(bin_df['biomech_bin_mid'].iloc[1] - bin_df['biomech_bin_mid'].iloc[0]) * 0.85,
            color='steelblue', alpha=0.5)
    ax2.set_ylabel('Number of pairs')
    ax2.set_xlabel('Biomechanical distance (bin midpoint)')

    plt.tight_layout()
    plt.show()