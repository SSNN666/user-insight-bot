import pandas as pd
import numpy as np


def clean_data(df):
    """
    数据清洗：缺失值处理、去重、过滤无效数据
    输入：原始订单宽表（至少包含 user_id, order_date, total_amount）
    """
    # 去重
    df = df.drop_duplicates()

    # 缺失值处理：total_amount 为空则用中位数填充
    df['total_amount'] = df['total_amount'].fillna(df['total_amount'].median())

    # 过滤异常：金额 <=0 或 缺失用户ID 的行
    df = df[(df['total_amount'] > 0) & df['user_id'].notna()]

    return df


def compute_rfm(df, reference_date=None):
    """
    计算 RFM 特征
    recency: 距离参考日期的天数
    frequency: 订单数
    monetary: 总消费金额
    """
    if reference_date is None:
        reference_date = df['order_date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('user_id').agg(
        last_order_date=('order_date', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('total_amount', 'sum')
    ).reset_index()

    rfm['recency'] = (reference_date - rfm['last_order_date']).dt.days
    return rfm[['user_id', 'recency', 'frequency', 'monetary']]