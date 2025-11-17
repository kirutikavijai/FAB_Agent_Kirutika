def liquidity_trend(start, end, liquidity_df):
    df = liquidity_df[(liquidity_df["year"] >= start) & (liquidity_df["year"] <= end)].copy()
    df["lcr"] = df["hqla"] / df["net_outflows"]
    df["nsfr"] = df["asf"] / df["rsf"]
    return df
