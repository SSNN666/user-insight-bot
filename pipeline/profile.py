"""Extended user profile dimensions beyond RFM.

Phase 3 additions:
- repurchase_interval: avg days between consecutive orders
- price_sensitivity: 0-1 score (higher = more price sensitive)
- category_preference: dominant product category
- category_diversity: unique categories / total orders
"""

import numpy as np
import pandas as pd

from errors.exceptions import ComputationError
from log.logger import get_logger

logger = get_logger(__name__)


def compute_extended_profile(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended profile dimensions per user.

    Requires ``orders_df`` to contain at least:
    ``user_id``, ``order_date``, ``total_amount``, ``unit_price``, ``category``.

    Returns DataFrame with columns:
    ``[user_id, repurchase_interval, price_sensitivity,
       category_preference, category_diversity]``
    """
    try:
        df = orders_df.sort_values(['user_id', 'order_date']).copy()

        profiles: list[dict] = []
        for user_id, grp in df.groupby('user_id'):
            profile: dict = {'user_id': user_id}

            # ── Repurchase interval ──
            dates = grp['order_date'].sort_values()
            if len(dates) >= 2:
                gaps = dates.diff().dropna().dt.days
                profile['repurchase_interval'] = round(gaps.mean(), 1)
            else:
                profile['repurchase_interval'] = np.nan

            # ── Price sensitivity ──
            if 'unit_price' in grp.columns and grp['unit_price'].notna().any():
                prices = grp['unit_price'].dropna()
                if prices.mean() > 0:
                    raw_ps = prices.std() / prices.mean()
                    profile['price_sensitivity'] = round(
                        min(raw_ps, 1.0), 3
                    )
                else:
                    profile['price_sensitivity'] = 0.0
            else:
                profile['price_sensitivity'] = 0.5  # neutral default

            # ── Category preference ──
            if 'category' in grp.columns and grp['category'].notna().any():
                cats = grp['category'].dropna()
                profile['category_preference'] = cats.mode().iloc[0] if len(cats.mode()) > 0 else "unknown"
                profile['category_diversity'] = round(
                    cats.nunique() / len(cats), 3
                )
            else:
                profile['category_preference'] = "unknown"
                profile['category_diversity'] = 0.0

            profiles.append(profile)

        result = pd.DataFrame(profiles)

        # Fill NaN repurchase_interval with global median
        median_ri = result['repurchase_interval'].median()
        if pd.notna(median_ri):
            result['repurchase_interval'] = result['repurchase_interval'].fillna(median_ri)
        else:
            result['repurchase_interval'] = result['repurchase_interval'].fillna(0)

        logger.info("extended_profile_computed", extra={
            "users": len(result),
            "median_repurchase_interval": round(float(median_ri), 1) if pd.notna(median_ri) else 0,
        })
        return result

    except Exception as e:
        raise ComputationError(
            f"Extended profile computation failed: {e}",
            context={"rows": len(orders_df)},
        ) from e


# ── Segment Growth Metrics (interview enhancement) ────────────


def compute_segment_growth(
    prev_stats: dict,
    curr_stats: dict,
) -> list[dict]:
    """Compute period-over-period growth metrics per segment.

    Calculates three interview-ready metrics:
      - sales_growth_pct: total revenue growth per segment (%)
      - conversion_change_pct: frequency change as proxy for conversion (%)
      - gmv_lift_pct: Gross Merchandise Volume lift (%)

    Args:
        prev_stats: Previous snapshot's segment_stats (seg_id → dict).
        curr_stats: Current snapshot's segment_stats.

    Returns:
        List of dicts: [{segment, sales_growth_pct, conversion_change_pct,
                          gmv_lift_pct, user_count, avg_monetary}, ...]
    """
    results = []
    all_segs = sorted(set(list(prev_stats.keys()) + list(curr_stats.keys())))

    for seg in all_segs:
        prev = prev_stats.get(seg, {})
        curr = curr_stats.get(seg, {})

        prev_users = prev.get("user_count", 0)
        curr_users = curr.get("user_count", 0)
        prev_mon = prev.get("avg_monetary", 0)
        curr_mon = curr.get("avg_monetary", 0)
        prev_freq = prev.get("avg_frequency", 0)
        curr_freq = curr.get("avg_frequency", 0)

        # Total revenue per segment
        prev_revenue = prev_users * prev_mon
        curr_revenue = curr_users * curr_mon

        # Sales growth %
        sales_growth = (
            round((curr_revenue - prev_revenue) / prev_revenue * 100, 1)
            if prev_revenue > 0 else 0.0
        )

        # Conversion rate change % (frequency as proxy)
        conversion_change = (
            round((curr_freq - prev_freq) / prev_freq * 100, 1)
            if prev_freq > 0 else 0.0
        )

        # GMV lift %
        gmv_lift = (
            round((curr_revenue - prev_revenue) / prev_revenue * 100, 1)
            if prev_revenue > 0 else 0.0
        )

        results.append({
            "segment": int(seg),
            "sales_growth_pct": sales_growth,
            "conversion_change_pct": conversion_change,
            "gmv_lift_pct": gmv_lift,
            "user_count": curr_users,
            "avg_monetary": round(curr_mon, 2),
            "prev_user_count": prev_users,
            "curr_user_count": curr_users,
        })

    # Sort by GMV lift descending
    results.sort(key=lambda x: x["gmv_lift_pct"], reverse=True)
    return results
