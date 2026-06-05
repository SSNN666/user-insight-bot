import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler


def perform_clustering(rfm_df, n_clusters=4):
    """
    使用 K-Means 对 RFM 标准化后聚类，返回聚类标签
    """
    features = ['recency', 'frequency', 'monetary']
    X = rfm_df[features].copy()
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df['segment'] = kmeans.fit_predict(X_scaled)

    # 按 monetary 均值对簇排序，便于解释（0: 最低价值，n-1: 最高价值）
    cluster_avg = rfm_df.groupby('segment')['monetary'].mean().sort_values()
    mapping = {old: new for new, old in enumerate(cluster_avg.index)}
    rfm_df['segment'] = rfm_df['segment'].map(mapping)

    return rfm_df, kmeans, scaler


def extract_rules(rfm_df):
    """
    用决策树学习 K-Means 的分群规则，提取可解释的文本规则
    """
    X = rfm_df[['recency', 'frequency', 'monetary']]
    y = rfm_df['segment']

    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X, y)

    feature_names = ['recency', 'frequency', 'monetary']
    rules = export_text(dt, feature_names=feature_names)
    return rules, dt


def segment_summary(rfm_df):
    """
    返回每个分群的统计概览
    """
    summary = rfm_df.groupby('segment').agg(
        用户数=('user_id', 'nunique'),
        平均近度=('recency', 'mean'),
        平均频次=('frequency', 'mean'),
        平均消费=('monetary', 'mean')
    ).reset_index()
    return summary