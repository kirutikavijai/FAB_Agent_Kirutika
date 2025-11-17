from core.utils import query_metric
import numpy as np

def yoy_change_metric(metric, year1, year2, df, df_type="income"):
    s = query_metric(df, metric, [year1])
    e = query_metric(df, metric, [year2])

    if s.empty or e.empty:
        raise ValueError(
            f"Metric '{metric}' not found for years {year1} or {year2} in {df_type} data."
        )

    v1 = s["value"].iloc[0]
    v2 = e["value"].iloc[0]
    abs_change = v2 - v1
    pct_change = (abs_change / v1 * 100) if v1 else np.nan

    return {
        "metric": metric,
        "start_year": year1,
        "end_year": year2,
        "start_value": v1,
        "end_value": v2,
        "absolute_change": abs_change,
        "pct_change": pct_change,
    }
