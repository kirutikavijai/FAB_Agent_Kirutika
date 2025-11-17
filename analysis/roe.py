import pandas as pd
from core.utils import compute_roe, compute_cagr, query_metric

def roe_trend(start, end, income_df, balance_df):
    years = list(range(start, end + 1))
    profit = query_metric(income_df, "Net profit", years)
    equity = query_metric(balance_df, "Total equity", years)

    rows = []
    for y in years:
        npv = profit.query("year == @y")["value"].iloc[0]
        eqv = equity.query("year == @y")["value"].iloc[0]
        rows.append({
            "year": y,
            "net_profit": npv,
            "equity": eqv,
            "roe": compute_roe(npv, eqv),
        })

    df = pd.DataFrame(rows)
    cagr = compute_cagr(df["net_profit"].iloc[0], df["net_profit"].iloc[-1], len(years) - 1)
    return df, cagr
