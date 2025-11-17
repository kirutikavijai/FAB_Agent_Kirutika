def npl_trend(start, end, credit_df):
    df = credit_df[(credit_df["year"] >= start) & (credit_df["year"] <= end)].copy()
    df["npl_ratio"] = df["npl"] / df["total_loans"]
    df["ecl_ratio"] = df["ecl"] / df["total_loans"]
    return df
