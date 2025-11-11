import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def get_available_numeric_cols(df: pd.DataFrame, candidate_cols: list) -> list:
    """Return subset of candidate_cols present in df"""
    return [c for c in candidate_cols if c in df.columns]


def run_kmeans(df: pd.DataFrame, feature_cols: list, n_clusters: int, random_state: int = 42):
    """Standardize features and run KMeans. Returns (kmeans_model, clusters_array, X_scaled)

    df is not modified by this function.
    """
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    return kmeans, clusters, X_scaled
