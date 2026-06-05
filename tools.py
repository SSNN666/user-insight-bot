from langchain.tools import tool
import pandas as pd
from data_cleaning import clean_data, compute_rfm
from user_segmentation import perform_clustering, extract_rules, segment_summary
from data_loader import load_orders_with_join, load_rfm_from_db, generate_mock_orders

# 缓存数据与模型，避免重复计算
_cached_rfm = None
_cached_segments = None
_cached_rules = None


def _load_and_process():
    global _cached_rfm, _cached_segments, _cached_rules
    if _cached_rfm is not None:
        return _cached_rfm, _cached_segments, _cached_rules

    # 尝试从数据库加载，失败则使用模拟数据
    try:
        df = load_orders_with_join()
        df = clean_data(df)
        rfm = compute_rfm(df)
    except Exception:
        print("数据库连接失败，使用模拟数据...")
        mock = generate_mock_orders()
        mock = clean_data(mock)
        rfm = compute_rfm(mock)

    rfm, kmeans, scaler = perform_clustering(rfm, n_clusters=4)
    rules, dt = extract_rules(rfm)

    _cached_rfm = rfm
    _cached_segments = rfm  # 包含 segment 列
    _cached_rules = rules
    return _cached_rfm, _cached_segments, _cached_rules


@tool
def get_user_segment_stats():
    """
    获取各用户分群的人数、平均近度、平均频次、平均消费金额。
    返回一个表格形式的文本。
    """
    _, segments, _ = _load_and_process()
    summary = segment_summary(segments)
    return summary.to_markdown(index=False)


@tool
def get_segment_rules():
    """
    获取决策树导出的用户分群规则，可帮助理解不同群体的特征。
    返回文本规则。
    """
    _, _, rules = _load_and_process()
    return rules


@tool
def get_high_value_users():
    """
    获取高价值用户列表（分群编号最大的群体），返回用户ID和他们的RFM指标。
    """
    rfm, segments, _ = _load_and_process()
    max_seg = segments['segment'].max()
    high_value = segments[segments['segment'] == max_seg][['user_id', 'recency', 'frequency', 'monetary']]
    return high_value.to_markdown(index=False)


# 工具列表
tools = [get_user_segment_stats, get_segment_rules, get_high_value_users]
