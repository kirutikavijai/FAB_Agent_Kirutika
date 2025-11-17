from core.utils import compute_roe, query_metric

def scenario_opex(year, pct, income_df, balance_df):
    inc  = query_metric(income_df, "Operating income",   [year])["value"].iloc[0]
    opex = query_metric(income_df, "Operating expenses", [year])["value"].iloc[0]
    imp  = query_metric(income_df, "Impairment charges", [year])["value"].iloc[0]
    eq   = query_metric(balance_df, "Total equity",      [year])["value"].iloc[0]

    base_np = inc - opex - imp
    new_np  = inc - (opex * (1 + pct / 100)) - imp
    base_roe = compute_roe(base_np, eq) * 100
    new_roe  = compute_roe(new_np, eq) * 100

    return base_np, new_np, base_roe, new_roe

def scenario_ecl(year, pct, credit_df, income_df, balance_df):
    row = credit_df[credit_df["year"] == year].iloc[0]
    base_ecl = row["ecl"]
    shocked_ecl = base_ecl * (1 + pct / 100)

    np_base = query_metric(income_df, "Net profit", [year])["value"].iloc[0]
    eq      = query_metric(balance_df, "Total equity", [year])["value"].iloc[0]

    new_np = np_base - (shocked_ecl - base_ecl)

    from core.utils import compute_roe
    base_roe = compute_roe(np_base, eq) * 100
    new_roe  = compute_roe(new_np, eq) * 100

    return np_base, new_np, base_ecl, shocked_ecl, base_roe, new_roe
