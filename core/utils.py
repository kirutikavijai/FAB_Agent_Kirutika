import numpy as np

def compute_roe(np_val, eq): 
    return np_val / eq if eq else np.nan

def compute_roa(np_val, assets): 
    return np_val / assets if assets else np.nan

def compute_cir(opex, opinc): 
    return opex / opinc if opinc else np.nan

def compute_nim(nii, ea): 
    return nii / ea if ea else np.nan

def compute_cagr(start, end, n):
    if start <= 0 or n <= 0:
        return np.nan
    return (end / start) ** (1 / n) - 1

def pretty(df, cols, title=None):
    if title:
        print(f"\n=== {title} ===")
    print(df[cols].to_string(index=False))

def query_metric(df, metric, years):
    return (
        df[(df["metric"] == metric) & (df["year"].isin(years))]
        .sort_values("year")
        .reset_index(drop=True)
    )
