def capital_trend(start, end, capital_df):
    df = capital_df[(capital_df["year"] >= start) & (capital_df["year"] <= end)].copy()
    df["car"] = df["total_capital"] / df["rwa"]
    df["cet1_ratio"] = df["cet1"] / df["rwa"]
    return df
