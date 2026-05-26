import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stability_analysis(
    statcast_df,
    features,
    sample_sizes=None,
    n_replicates=50,
    n_pitchers=50,
    group_by_pitch_type=False,
    random_state=42,
):
    """
    For each feature, compute stability across sample sizes by bootstrapping
    pitch-level data and measuring the sampling SD of estimates across replicates.

    Parameters
    ----------
    statcast_df          : pitch-level DataFrame (statcast_25_clean)
    features             : list of feature names to test
    sample_sizes         : list of ints; defaults to [10, 25, 50, 75, 100, 150, 200]
    n_replicates         : bootstrap replicates per sample size
    n_pitchers           : how many groups to sample for the analysis;
                           only groups with >= max(sample_sizes) pitches are eligible
    group_by_pitch_type  : if True, group by (player_name, pitch_type) instead of
                           player_name alone; use for pitch-characteristic features
    random_state         : for reproducibility

    Returns
    -------
    stability_df : aggregated long-format DataFrame with columns
                   [sample_size, feature, mean_se, p25_se, p50_se, p75_se]
    raw_df       : one row per (group, sample_size, feature, replicate-summary)
    """
    if sample_sizes is None:
        sample_sizes = [10, 25, 50, 75, 100, 150, 200]

    min_pitches = max(sample_sizes)
    rng         = np.random.default_rng(random_state)

    group_cols = ['player_name', 'pitch_type'] if group_by_pitch_type else ['player_name']

    pitch_counts = statcast_df.groupby(group_cols).size()
    eligible     = pitch_counts[pitch_counts >= min_pitches].index.tolist()
    n_sample     = min(n_pitchers, len(eligible))
    sampled      = [eligible[i] for i in rng.choice(len(eligible), size=n_sample, replace=False)]

    rows = []
    for key in sampled:
        if group_by_pitch_type:
            name, pitch_type = key
            group_data = statcast_df[
                (statcast_df['player_name'] == name) &
                (statcast_df['pitch_type'] == pitch_type)
            ].reset_index(drop=True)
            label = f"{name} / {pitch_type}"
        else:
            name       = key
            group_data = statcast_df[statcast_df['player_name'] == name].reset_index(drop=True)
            label      = name

        for n in sample_sizes:
            if n > len(group_data):
                continue

            replicate_vals = {f: [] for f in features}
            for _ in range(n_replicates):
                subset = group_data.sample(n=n, replace=False, random_state=None)
                for feature in features:
                    vals = subset[feature].dropna()
                    if len(vals) > 0:
                        replicate_vals[feature].append(vals.mean())

            for feature in features:
                vals = replicate_vals[feature]
                if len(vals) < 2:
                    continue
                vals = np.array(vals)
                rows.append({
                    'group':       label,
                    'sample_size': n,
                    'feature':     feature,
                    'sampling_sd': vals.std(),
                    'mean_est':    round(vals.mean(), 2),
                })

    raw_df = pd.DataFrame(rows)

    stability_df = (
        raw_df.groupby(['sample_size', 'feature'])['sampling_sd']
        .agg(
            mean_se='mean',
            p25_se=lambda x: np.percentile(x, 25),
            p50_se=lambda x: np.percentile(x, 50),
            p75_se=lambda x: np.percentile(x, 75),
        )
        .round(4)
        .reset_index()
    )

    return stability_df, raw_df


def plot_stability(stability_df, population_sds, threshold_pct=0.10):
    """
    Plot sampling SD vs sample size for each feature, with p25-p75 band
    and a feature-specific stability threshold at threshold_pct of population SD.

    Parameters
    ----------
    stability_df    : output of stability_analysis
    population_sds  : dict mapping feature name to its population SD,
                      e.g. {'arm_angle': 8.2, 'release_extension': 0.4, ...}
                      computed within handedness group from pitcher_summ / pitch_type_summ
    threshold_pct   : fraction of population SD to use as stability threshold (default 0.10)
    """
    features = stability_df['feature'].unique()
    ncols    = 2
    nrows    = int(np.ceil(len(features) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharey=False)
    axes      = axes.flatten()

    for i, feature in enumerate(features):
        ax  = axes[i]
        sub = stability_df[stability_df['feature'] == feature]

        ax.plot(sub['sample_size'], sub['mean_se'], color='steelblue', lw=2, label='Mean SE')
        ax.fill_between(
            sub['sample_size'], sub['p25_se'], sub['p75_se'],
            alpha=0.25, color='steelblue', label='P25–P75'
        )

        if feature in population_sds:
            threshold = population_sds[feature] * threshold_pct
            ax.axhline(
                threshold, color='firebrick', lw=1, linestyle='--',
                label=f'{int(threshold_pct * 100)}% of pop SD ({threshold:.2f})'
            )

        ax.set_title(feature)
        ax.set_xlabel('Sample size (pitches)')
        ax.set_ylabel('Sampling SD')
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature stability by sample size', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()