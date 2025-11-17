import pandas as pd
from core.utils import compute_cir, query_metric

def cir_trend(start, end, income_df):
    years = list(range(start, end + 1))
    op_inc = query_metric(income_df, "Operating income", years)
    opex   = query_metric(income_df, "Operating expenses", years)

    rows = []
    for y in years:
        inc = op_inc.query("year == @y")["value"].iloc[0]
        ex  = opex.query("year == @y")["value"].iloc[0]
        rows.append({
            "year": y,
            "operating_income": inc,
            "operating_expenses": ex,
            "cir": compute_cir(ex, inc),
        })
    return pd.DataFrame(rows)
