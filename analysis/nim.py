import pandas as pd
from core.utils import compute_nim, query_metric

def nim_trend(start, end, income_df, balance_df):
    years = list(range(start, end + 1))
    nii = query_metric(income_df, "Net interest income", years)
    ea  = query_metric(balance_df, "Earning assets", years)

    rows = []
    for y in years:
        ni  = nii.query("year == @y")["value"].iloc[0]
        eav = ea.query("year == @y")["value"].iloc[0]
        rows.append({
            "year": y,
            "net_interest_income": ni,
            "earning_assets": eav,
            "nim": compute_nim(ni, eav),
        })
    return pd.DataFrame(rows)
