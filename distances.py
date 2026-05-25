def compute_mahalanobis_distances(df, features, label_cols, min_pitches=None):
    """
    Compute pairwise Mahalanobis distances between pitch types.
    Parameters:
        df          : DataFrame with label columns and numeric feature columns
        features    : list of feature columns to use
        label_cols  : list of columns to use as labels in the output
        min_pitches : minimum number of pitches to be considered
    Returns:
        Long-form DataFrame with label columns suffixed with '1' and '2', plus a distance column
    """
    if isinstance(min_pitches, int):
        df = df[df['n'] >= min_pitches]
    elif min_pitches is not None:
        raise TypeError("min_pitches must be an integer value")

    df = df[label_cols + features].dropna().reset_index(drop=True)
    X = df[features].values

    VI = linalg.inv(np.cov(X, rowvar=False))
    dist_matrix = cdist(X, X, metric="mahalanobis", VI=VI)
    i_idx, j_idx = np.triu_indices(len(df), k=1)
    result_dict = {}
    for col in label_cols:
        result_dict[f"{col}1"] = df[col].values[i_idx]
        result_dict[f"{col}2"] = df[col].values[j_idx]
    result_dict["distance"] = dist_matrix[i_idx, j_idx]
    
    return pd.DataFrame(result_dict).sort_values("distance").reset_index(drop=True)


def compute_euclidean_distances(df, features, label_cols, min_pitches=None, include_features=False):
    """
    Compute pairwise standardized Euclidean distances.
    Features are z-scored internally before computing distances.
    
    Parameters:
        df          : DataFrame with label columns and numeric feature columns
        features    : list of feature columns to use (standardized internally)
        label_cols  : list of columns to use as labels in the output
        min_pitches : minimum number of pitches to be considered
        include_features : if True, append scaled feature values for each pitcher
                           in the pair as additional columns (e.g., extension1, extension2)
    Returns:
        Long-form DataFrame with label columns suffixed with '1' and '2', plus a distance column,
        and optionally scaled feature columns suffixed with '1' and '2'
    """
    if isinstance(min_pitches, int):
        df = df[df['n'] >= min_pitches]
    elif min_pitches is not None:
        raise TypeError("min_pitches must be an integer value")

    df = df[label_cols + features].dropna().reset_index(drop=True)
    X = StandardScaler().fit_transform(df[features].values)

    dist_matrix = cdist(X, X, metric="euclidean")
    i_idx, j_idx = np.triu_indices(len(df), k=1)
    result_dict = {}
    for col in label_cols:
        result_dict[f"{col}1"] = df[col].values[i_idx]
        result_dict[f"{col}2"] = df[col].values[j_idx]
    result_dict["distance"] = dist_matrix[i_idx, j_idx]
    
    if include_features:
        for k, feat in enumerate(features):
            result_dict[f"{feat}1"] = X[i_idx, k]
            result_dict[f"{feat}2"] = X[j_idx, k]
            
    return pd.DataFrame(result_dict).sort_values("distance").reset_index(drop=True)