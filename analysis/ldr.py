import pandas as pd
from core.utils import query_metric

def ldr_trend(start, end, balance_df):
    years = list(range(start, end + 1))
    loans = query_metric(balance_df, "Customer loans", years)
    deps  = query_metric(balance_df, "Customer deposits", years)

    rows = []
    for y in years:
        l = loans.query("year == @y")["value"].iloc[0]
        d = deps.query("year == @y")["value"].iloc[0]
        rows.append({
            "year": y,
            "loans": l,
            "deposits": d,
            "ldr": l / d,
        })
    return pd.DataFrame(rows)
