"""Data cleaning, RFM feature engineering, and outlier detection.

Phase 3 upgrades:
- ``detect_outliers()`` — IQR / Z-score extreme value detection
- ``clean_data()`` — now filters outliers (configurable)
- ``compute_rfm_incremental()`` — delta-only RFM for new orders
"""

import numpy as np
import pandas as pd

from config.settings import get_settings
from errors.exceptions import ComputationError
from log.logger import get_logger

logger = get_logger(__name__)


# ── Outlier Detection ───────────────────────────────────────────


def detect_outliers(
    df: pd.DataFrame,
    method: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Flag rows where any numeric column is an outlier.

    Args:
        df: Input DataFrame.
        method: ``"iqr"`` (default) or ``"zscore"``.
        columns: Columns to check. Defaults to ``['total_amount', 'quantity']``
                 if present, else all numeric.

    Returns:
        DataFrame with an added ``is_outlier`` boolean column.
    """
    settings = get_settings()
    method = method or settings.OUTLIER_METHOD
    if columns is None:
        defaults = ['total_amount', 'quantity']
        columns = [c for c in defaults if c in df.columns]
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.copy()
    df['is_outlier'] = False

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 4:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            col_outliers = (df[col] < lo) | (df[col] > hi)
        else:  # zscore
            z = (series - series.mean()) / series.std()
            col_mask = abs(z) > 3
            col_outliers = df[col].isin(series[col_mask].index)

        outlier_count = int(col_outliers.sum())
        df.loc[col_outliers, 'is_outlier'] = True
        if outlier_count > 0:
            sample_vals = df.loc[col_outliers, col].head(5).tolist()
            logger.info("outliers_detected", extra={
                "column": col, "method": method,
                "outlier_count": outlier_count,
                "sample_values": sample_vals,
            })

    total_outliers = df['is_outlier'].sum()
    if total_outliers > 0:
        logger.info("total_outliers", extra={"count": int(total_outliers)})

    return df


# ── Data Cleaning ───────────────────────────────────────────────


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, impute, filter invalid rows, and remove outliers.

    Args:
        df: Raw orders wide-table (must contain at least
            ``user_id``, ``order_date``, ``total_amount``).

    Returns:
        Cleaned DataFrame (outliers removed if ``FILTER_OUTLIERS`` is True).
    """
    try:
        df = df.drop_duplicates()
        df['total_amount'] = df['total_amount'].fillna(
            df['total_amount'].median()
        )
        df = df[(df['total_amount'] > 0) & df['user_id'].notna()]

        # Outlier detection & filtering
        settings = get_settings()
        df = detect_outliers(df)
        if settings.FILTER_OUTLIERS:
            before = len(df)
            df = df[~df['is_outlier']]
            removed = before - len(df)
            if removed > 0:
                logger.info("outliers_filtered", extra={
                    "removed": removed, "remaining": len(df),
                })

        logger.info("data_cleaned", extra={"rows": len(df)})
        return df
    except Exception as e:
        raise ComputationError(
            f"Data cleaning failed: {e}", context={"rows": len(df)}
        ) from e


# ── Full RFM Computation ────────────────────────────────────────


def compute_rfm(
    df: pd.DataFrame, reference_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per user (full recompute).

    Args:
        df: Cleaned orders DataFrame.
        reference_date: The "current" date for recency calculation.
            Defaults to ``max(order_date) + 1 day``.

    Returns:
        DataFrame with columns ``[user_id, recency, frequency, monetary]``.
    """
    try:
        if reference_date is None:
            reference_date = df['order_date'].max() + pd.Timedelta(days=1)

        rfm = df.groupby('user_id').agg(
            last_order_date=('order_date', 'max'),
            frequency=('order_id', 'nunique'),
            monetary=('total_amount', 'sum'),
        ).reset_index()

        rfm['recency'] = (
            reference_date - rfm['last_order_date']
        ).dt.days
        logger.info("rfm_computed", extra={"users": len(rfm)})
        return rfm[['user_id', 'recency', 'frequency', 'monetary']]
    except Exception as e:
        raise ComputationError(
            f"RFM computation failed: {e}", context={"rows": len(df)}
        ) from e


# ── Incremental RFM Computation ─────────────────────────────────


def compute_rfm_incremental(
    existing_rfm: pd.DataFrame | None,
    new_orders_df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Update RFM for only those users with new orders.

    Args:
        existing_rfm: Previously computed RFM DataFrame
            ``[user_id, recency, frequency, monetary]``.
            If ``None``, falls back to full ``compute_rfm()``.
        new_orders_df: New orders since last computation.
        reference_date: Reference date for recency.

    Returns:
        Updated RFM DataFrame with the same columns.
    """
    if existing_rfm is None or existing_rfm.empty:
        logger.info("incremental_rfm_fallback_to_full")
        return compute_rfm(new_orders_df, reference_date)

    try:
        if reference_date is None:
            reference_date = (
                new_orders_df['order_date'].max() + pd.Timedelta(days=1)
            )

        # Compute delta RFM for new orders only
        delta_rfm = new_orders_df.groupby('user_id').agg(
            new_last_order=('order_date', 'max'),
            new_frequency=('order_id', 'nunique'),
            new_monetary=('total_amount', 'sum'),
        ).reset_index()
        delta_rfm['new_recency'] = (
            reference_date - delta_rfm['new_last_order']
        ).dt.days

        # Merge with existing
        updated = existing_rfm.merge(
            delta_rfm[['user_id', 'new_recency', 'new_frequency', 'new_monetary']],
            on='user_id', how='left',
        )

        # For users with new orders: recency = min(existing, new)
        has_new = updated['new_recency'].notna()
        updated.loc[has_new, 'recency'] = updated.loc[
            has_new, ['recency', 'new_recency']
        ].min(axis=1).astype(int)
        # frequency & monetary are additive
        updated.loc[has_new, 'frequency'] = (
            updated.loc[has_new, 'frequency']
            + updated.loc[has_new, 'new_frequency'].fillna(0)
        ).astype(int)
        updated.loc[has_new, 'monetary'] = (
            updated.loc[has_new, 'monetary']
            + updated.loc[has_new, 'new_monetary'].fillna(0)
        )

        # New users (not in existing_rfm)
        new_users = delta_rfm[
            ~delta_rfm['user_id'].isin(existing_rfm['user_id'])
        ]
        if len(new_users) > 0:
            new_rows = pd.DataFrame({
                'user_id': new_users['user_id'],
                'recency': new_users['new_recency'].astype(int),
                'frequency': new_users['new_frequency'].astype(int),
                'monetary': new_users['new_monetary'],
                'last_order_date': new_users['new_last_order'],
            })
            updated = pd.concat(
                [updated, new_rows[['user_id', 'recency', 'frequency', 'monetary']]],
                ignore_index=True,
            )

        # Drop temporary columns
        result = updated[['user_id', 'recency', 'frequency', 'monetary']].copy()

        logger.info("incremental_rfm_done", extra={
            "existing_users": len(existing_rfm),
            "users_with_new": int(has_new.sum()),
            "new_users": len(new_users),
            "total": len(result),
        })
        return result

    except Exception as e:
        raise ComputationError(
            f"Incremental RFM failed: {e}",
            context={"existing_users": len(existing_rfm) if existing_rfm is not None else 0},
        ) from e
