import pandas as pd
import numpy as np
from analysis.roe import roe_trend

def forecast_roe(start, end, income_df, balance_df, horizon=1):
    hist_df, _ = roe_trend(start, end, income_df, balance_df)
    x = hist_df["year"].values
    y = hist_df["roe"].values

    m, b = np.polyfit(x, y, 1)
    future_rows = []
    for i in range(1, horizon + 1):
        year = end + i
        future_rows.append({"year": year, "forecast_roe": m * year + b})

    future_df = pd.DataFrame(future_rows)
    return hist_df, future_df
