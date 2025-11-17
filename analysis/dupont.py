from core.utils import query_metric

def dupont(year, income_df, balance_df):
    npv    = query_metric(income_df, "Net profit",       [year])["value"].iloc[0]
    inc    = query_metric(income_df, "Operating income", [year])["value"].iloc[0]
    assets = query_metric(balance_df, "Total assets",    [year])["value"].iloc[0]
    eq     = query_metric(balance_df, "Total equity",    [year])["value"].iloc[0]

    pm = npv / inc
    at = inc / assets
    em = assets / eq

    return {
        "year": year,
        "profit_margin": pm,
        "asset_turnover": at,
        "equity_multiplier": em,
        "roe": pm * at * em,
    }
