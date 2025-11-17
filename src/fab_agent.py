#!/usr/bin/env python3
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------
# GLOBAL SETTINGS
# --------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")

# Toggle this to False to suppress charts (e.g. evaluation mode)
SHOW_PLOTS = True

# ============================================
# 1. LOAD ALL CSV FILES
# ============================================

DATA_DIR = "data"

def load_csv(name: str) -> pd.DataFrame:
    """
    Load CSV from either:
    - data/<name>
    - or if 'name' is an absolute path, just load that path.
    """
    # If 'name' is absolute, os.path.join will return name on Windows
    path = os.path.join(DATA_DIR, name)
    df = pd.read_csv(path)

    # Auto-strip whitespace & quotes
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )
    if "metric" in df.columns:
        df["metric"] = (
            df["metric"]
            .astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
        )
    if "peer" in df.columns:
        df["peer"] = (
            df["peer"]
            .astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
        )
    if "year" in df.columns:
        # Make sure year is numeric
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df


income_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\income_statement.csv")
balance_df = load_csv("E:\\workspace\\FAB Agents\\FAB_Agent_Kirutika\\data\\balance_sheet.csv")
credit_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\credit_quality.csv")
capital_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\capital.csv")
liquidity_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\liquidity.csv")

peer_income_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\peer_income_statement.csv")
peer_balance_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\peer_balance_sheet.csv")

ALL_YEARS = sorted([int(y) for y in balance_df["year"].dropna().unique()])

# ============================================
# 2. BASIC UTILITIES
# ============================================

def query_metric(df: pd.DataFrame, metric: str, years):
    return (
        df[(df["metric"] == metric) & (df["year"].isin(years))]
        .sort_values("year")
        .reset_index(drop=True)
    )

def compute_roe(np_val, eq): return np_val / eq if eq else np.nan
def compute_roa(np_val, assets): return np_val / assets if assets else np.nan
def compute_cir(opex, opinc): return opex / opinc if opinc else np.nan
def compute_nim(nii, ea): return nii / ea if ea else np.nan

def compute_cagr(start, end, n):
    if start <= 0 or n <= 0: 
        return np.nan
    return (end / start) ** (1 / n) - 1

def pretty(df: pd.DataFrame, cols, title=None):
    if title: 
        print(f"\n=== {title} ===")
    print(df[cols].to_string(index=False))

# ============================================
# 3. ANALYSIS ENGINES (ALL FEATURES)
# ============================================

def roe_trend(start, end, use_peer=False):
    inc = peer_income_df if use_peer else income_df
    bal = peer_balance_df if use_peer else balance_df
    yrs = list(range(start, end + 1))

    profit = query_metric(inc, "Net profit", yrs)
    eq = query_metric(bal, "Total equity", yrs)

    rows = []
    for y in yrs:
        p = profit.query("year==@y")["value"].iloc[0]
        e = eq.query("year==@y")["value"].iloc[0]
        rows.append({
            "year": y,
            "net_profit": p,
            "equity": e,
            "roe": compute_roe(p, e)
        })

    df = pd.DataFrame(rows)
    cagr = compute_cagr(df["net_profit"].iloc[0], df["net_profit"].iloc[-1], len(yrs) - 1)
    return df, cagr

def cir_trend(start, end):
    yrs = list(range(start, end + 1))
    op_inc = query_metric(income_df, "Operating income", yrs)
    opex = query_metric(income_df, "Operating expenses", yrs)

    rows = []
    for y in yrs:
        inc = op_inc.query("year==@y")["value"].iloc[0]
        ex = opex.query("year==@y")["value"].iloc[0]
        rows.append({
            "year": y,
            "operating_income": inc,
            "operating_expenses": ex,
            "cir": compute_cir(ex, inc)
        })
    return pd.DataFrame(rows)

def nim_trend(start, end, use_peer=False):
    inc = peer_income_df if use_peer else income_df
    bal = peer_balance_df if use_peer else balance_df
    yrs = list(range(start, end + 1))

    nii = query_metric(inc, "Net interest income", yrs)
    ea = query_metric(bal, "Earning assets", yrs)

    rows = []
    for y in yrs:
        ni  = nii.query("year==@y")["value"].iloc[0]
        eav = ea.query("year==@y")["value"].iloc[0]
        rows.append({
            "year": y,
            "net_interest_income": ni,
            "earning_assets": eav,
            "nim": compute_nim(ni, eav)
        })
    return pd.DataFrame(rows)

def npl_trend(start, end):
    df = credit_df[(credit_df["year"] >= start) & (credit_df["year"] <= end)].copy()
    df["npl_ratio"] = df["npl"] / df["total_loans"]
    df["ecl_ratio"] = df["ecl"] / df["total_loans"]
    return df

def ldr_trend(start, end, use_peer=False):
    bal = peer_balance_df if use_peer else balance_df
    yrs = list(range(start, end + 1))
    loans = query_metric(bal, "Customer loans", yrs)
    deps  = query_metric(bal, "Customer deposits", yrs)

    rows = []
    for y in yrs:
        l = loans.query("year==@y")["value"].iloc[0]
        d = deps.query("year==@y")["value"].iloc[0]
        rows.append({
            "year": y,
            "loans": l,
            "deposits": d,
            "ldr": l / d
        })
    return pd.DataFrame(rows)

def capital_trend(start, end):
    df = capital_df[(capital_df["year"] >= start) & (capital_df["year"] <= end)].copy()
    df["car"] = df["total_capital"] / df["rwa"]
    df["cet1_ratio"] = df["cet1"] / df["rwa"]
    return df

def liquidity_trend(start, end):
    df = liquidity_df[(liquidity_df["year"] >= start) & (liquidity_df["year"] <= end)].copy()
    df["lcr"] = df["hqla"] / df["net_outflows"]
    df["nsfr"] = df["asf"] / df["rsf"]
    return df

def scenario_opex(year, pct):
    inc  = query_metric(income_df, "Operating income",   [year])["value"].iloc[0]
    opex = query_metric(income_df, "Operating expenses", [year])["value"].iloc[0]
    imp  = query_metric(income_df, "Impairment charges", [year])["value"].iloc[0]
    eq   = query_metric(balance_df, "Total equity",      [year])["value"].iloc[0]

    base_np = inc - opex - imp
    new_np  = inc - (opex * (1 + pct / 100)) - imp
    return base_np, new_np, compute_roe(base_np, eq) * 100, compute_roe(new_np, eq) * 100

def scenario_ecl(year, pct):
    row = credit_df[credit_df["year"] == year].iloc[0]
    base = row["ecl"]
    shocked = base * (1 + pct / 100)
    np_base = query_metric(income_df, "Net profit", [year])["value"].iloc[0]
    eq      = query_metric(balance_df, "Total equity", [year])["value"].iloc[0]
    new_np  = np_base - (shocked - base)
    return (
        np_base,
        new_np,
        base,
        shocked,
        compute_roe(np_base, eq) * 100,
        compute_roe(new_np, eq) * 100,
    )

def dupont(year):
    npv    = query_metric(income_df, "Net profit",       [year])["value"].iloc[0]
    inc    = query_metric(income_df, "Operating income", [year])["value"].iloc[0]
    assets = query_metric(balance_df, "Total assets",    [year])["value"].iloc[0]
    eq     = query_metric(balance_df, "Total equity",    [year])["value"].iloc[0]

    pm = npv / inc              # Profit margin
    at = inc / assets           # Asset turnover
    em = assets / eq            # Equity multiplier

    return {
        "year": year,
        "profit_margin": pm,
        "asset_turnover": at,
        "equity_multiplier": em,
        "roe": pm * at * em,
    }

def forecast_roe(start, end, h=1):
    df, _ = roe_trend(start, end)
    x = df["year"].values
    y = df["roe"].values
    m, b = np.polyfit(x, y, 1)
    future = [{"year": end + i, "forecast_roe": m * (end + i) + b} for i in range(1, h + 1)]
    return df, pd.DataFrame(future)

# ---------- NEW: GENERIC YoY METRIC CHANGE ----------

def yoy_change_metric(metric, year1, year2, df_type="income", use_peer=False):
    """
    Calculate year-over-year absolute and percentage change for a given metric.
    df_type: 'income' | 'balance' | 'capital' | 'liquidity'
    """
    if df_type == "income":
        df = peer_income_df if use_peer else income_df
    elif df_type == "balance":
        df = peer_balance_df if use_peer else balance_df
    elif df_type == "capital":
        df = capital_df
    elif df_type == "liquidity":
        df = liquidity_df
    else:
        df = income_df

    s = query_metric(df, metric, [year1])
    e = query_metric(df, metric, [year2])
    if s.empty or e.empty:
        raise ValueError(f"Metric '{metric}' not found for years {year1} or {year2} in {df_type} data.")

    v1 = s["value"].iloc[0]
    v2 = e["value"].iloc[0]
    abs_change = v2 - v1
    pct_change = (abs_change / v1 * 100) if v1 else np.nan

    result = {
        "metric": metric,
        "start_year": year1,
        "end_year": year2,
        "start_value": v1,
        "end_value": v2,
        "absolute_change": abs_change,
        "pct_change": pct_change,
    }
    return result

# ============================================
# 4. "LLM-STYLE" PARSER
# ============================================

def llm_parse(q: str):
    t = q.lower()

    intent = "unknown"
    metric = None
    df_type = None

    # --- Detect metric from text (can be expanded as needed) ---
    if "net profit" in t:
        metric = "Net profit"
        df_type = "income"
    elif "total assets" in t:
        metric = "Total assets"
        df_type = "balance"
    elif "total equity" in t:
        metric = "Total equity"
        df_type = "balance"
    elif "customer loans" in t:
        metric = "Customer loans"
        df_type = "balance"
    elif "customer deposits" in t:
        metric = "Customer deposits"
        df_type = "balance"

    # --- Detect special intents (order matters) ---
    if any(kw in t for kw in ["year-over-year", "year over year", "yoy", "percentage change"]):
        intent = "yoy_metric"
    elif "dashboard" in t:
        intent = "dashboard"
    elif "compare" in t and ("ldr" in t or "loan-to-deposit" in t or "loan to deposit" in t):
        intent = "ldr_compare"
    elif "compare" in t and "roe" in t:
        intent = "peer"
    elif "forecast" in t:
        intent = "forecast"
    elif "dupont" in t or "du pont" in t:
        intent = "dupont"
    elif "cir" in t or "cost to income" in t:
        intent = "cir"
    elif "nim" in t:
        intent = "nim"
    elif "npl" in t:
        intent = "npl"
    elif "ldr" in t or "loan-to-deposit" in t or "loan to deposit" in t:
        intent = "ldr"
    elif "car" in t or "cet1" in t:
        intent = "capital"
    elif "liquidity" in t or "lcr" in t or "nsfr" in t:
        intent = "liq"
    elif "scenario" in t and "ecl" in t:
        intent = "scenario_ecl"
    elif "scenario" in t or "opex" in t or "operating expense" in t:
        intent = "scenario_opex"
    elif "roe" in t:
        intent = "roe"

    # Extract years (any 20xx pattern)
    yrs = re.findall(r"(20\d{2})", t)
    yrs = [int(y) for y in yrs] if yrs else ALL_YEARS

    # Extract percentage (e.g. "10%")
    pct = re.findall(r"(\d+)%", t)
    pct = int(pct[0]) if pct else None

    return {
        "intent": intent,
        "years": yrs,
        "pct": pct,
        "metric": metric,
        "df_type": df_type,
    }

# ============================================
# 5. AGENT
# ============================================

def agent(question: str):
    task   = llm_parse(question)
    intent = task["intent"]
    yrs    = task["years"]
    pct    = task["pct"]
    metric = task["metric"]
    df_type= task["df_type"]

    start, end = min(yrs), max(yrs)

    print("\nParsed:", task)

    # ---------------- Core existing intents ----------------
    if intent == "roe":
        df, c = roe_trend(start, end)
        pretty(df, ["year", "net_profit", "equity", "roe"], "ROE Trend")
        if SHOW_PLOTS:
            plt.plot(df["year"], df["roe"], marker="o")
            plt.title("ROE Trend")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.show()
        print(f"CAGR of Net Profit between {start} and {end}: {c:.2%}")

    elif intent == "cir":
        df = cir_trend(start, end)
        pretty(df, ["year", "operating_income", "operating_expenses", "cir"], "CIR Trend")
        if SHOW_PLOTS:
            plt.plot(df["year"], df["cir"])
            plt.title("Cost-to-Income Ratio")
            plt.xlabel("Year")
            plt.ylabel("CIR")
            plt.show()

    elif intent == "nim":
        df = nim_trend(start, end)
        pretty(df, ["year", "net_interest_income", "earning_assets", "nim"], "NIM Trend")
        if SHOW_PLOTS:
            plt.plot(df["year"], df["nim"])
            plt.title("Net Interest Margin")
            plt.xlabel("Year")
            plt.ylabel("NIM")
            plt.show()

    elif intent == "npl":
        df = npl_trend(start, end)
        print("\n=== NPL & ECL Ratios ===")
        print(df.to_string(index=False))
        if SHOW_PLOTS:
            plt.plot(df["year"], df["npl_ratio"])
            plt.title("NPL Ratio")
            plt.xlabel("Year")
            plt.ylabel("NPL Ratio")
            plt.show()

    elif intent == "ldr":
        df = ldr_trend(start, end)
        pretty(df, ["year", "loans", "deposits", "ldr"], "Loan-to-Deposit Ratio")
        if SHOW_PLOTS:
            plt.plot(df["year"], df["ldr"])
            plt.title("LDR")
            plt.xlabel("Year")
            plt.ylabel("LDR")
            plt.show()

    elif intent == "capital":
        df = capital_trend(start, end)
        print("\n=== Capital Ratios ===")
        print(df.to_string(index=False))
        if SHOW_PLOTS:
            plt.plot(df["year"], df["car"])
            plt.title("CAR")
            plt.xlabel("Year")
            plt.ylabel("Capital Adequacy Ratio")
            plt.show()

    elif intent == "liq":
        df = liquidity_trend(start, end)
        print("\n=== Liquidity Ratios ===")
        print(df.to_string(index=False))
        if SHOW_PLOTS:
            plt.plot(df["year"], df["lcr"])
            plt.title("LCR")
            plt.xlabel("Year")
            plt.ylabel("Liquidity Coverage Ratio")
            plt.show()

    elif intent == "scenario_opex":
        year = end
        pct  = pct or 10
        base, new, br, nr = scenario_opex(year, pct)
        print(f"\n=== OPEX Scenario for {year} (+{pct}%) ===")
        print(f"Base Net Profit: {base:,.2f}")
        print(f"New  Net Profit: {new:,.2f}")
        print(f"Base ROE: {br:.2f}%")
        print(f"New  ROE: {nr:.2f}%")
        if SHOW_PLOTS:
            plt.bar(["Base", "Scenario"], [base, new])
            plt.title(f"Net Profit Scenario {year} (+{pct}%)")
            plt.show()

    elif intent == "scenario_ecl":
        year = end
        pct  = pct or 20
        np_base, new_np, base_ecl, shocked_ecl, br, nr = scenario_ecl(year, pct)
        print(f"\n=== ECL Shock Scenario for {year} (+{pct}%) ===")
        print(f"Base ECL:     {base_ecl:,.2f}")
        print(f"Shocked ECL:  {shocked_ecl:,.2f}")
        print(f"Base Net Profit: {np_base:,.2f}")
        print(f"New  Net Profit: {new_np:,.2f}")
        print(f"Base ROE: {br:.2f}%")
        print(f"New  ROE:  {nr:.2f}%")

    elif intent == "dupont":
        d = dupont(end)
        print("\n=== DuPont Decomposition ===")
        for k, v in d.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    elif intent == "forecast":
        hist, fc = forecast_roe(start, end)
        print("\n=== Historical ROE ===")
        print(hist.to_string(index=False))
        print("\n=== Forecast ROE ===")
        print(fc.to_string(index=False))
        if SHOW_PLOTS:
            plt.plot(hist["year"], hist["roe"], label="Historical")
            plt.plot(fc["year"], fc["forecast_roe"], label="Forecast", linestyle="--")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.legend()
            plt.title("ROE Forecast")
            plt.show()

    elif intent == "peer":
        fab_df, _  = roe_trend(start, end)
        peer_df, _ = roe_trend(start, end, use_peer=True)
        print("\n=== FAB vs Peer ROE Comparison ===")
        print("FAB:")
        print(fab_df.to_string(index=False))
        print("\nPeer:")
        print(peer_df.to_string(index=False))
        if SHOW_PLOTS:
            plt.plot(fab_df["year"], fab_df["roe"], label="FAB")
            plt.plot(peer_df["year"], peer_df["roe"], label="Peer")
            plt.legend()
            plt.title("ROE Comparison")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.show()

    elif intent == "dashboard":
        roe_df, _ = roe_trend(start, end)
        cir_df    = cir_trend(start, end)
        nim_df    = nim_trend(start, end)
        npl_df    = npl_trend(start, end)

        if SHOW_PLOTS:
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.plot(roe_df["year"], roe_df["roe"])
            plt.title("ROE")

            plt.subplot(2, 2, 2)
            plt.plot(cir_df["year"], cir_df["cir"])
            plt.title("CIR")

            plt.subplot(2, 2, 3)
            plt.plot(nim_df["year"], nim_df["nim"])
            plt.title("NIM")

            plt.subplot(2, 2, 4)
            plt.plot(npl_df["year"], npl_df["npl_ratio"])
            plt.title("NPL Ratio")

            plt.tight_layout()
            plt.show()
        else:
            print("\nDashboard metrics (no plots because SHOW_PLOTS=False):")
            print("ROE:")
            print(roe_df.to_string(index=False))
            print("\nCIR:")
            print(cir_df.to_string(index=False))
            print("\nNIM:")
            print(nim_df.to_string(index=False))
            print("\nNPL:")
            print(npl_df.to_string(index=False))

    # ---------------- NEW INTENTS ----------------

    elif intent == "yoy_metric":
        # Default metric if not explicitly detected
        if metric is None:
            metric  = "Net profit"
            df_type = "income"
        if len(yrs) < 2:
            # If only one year in text, compare last 2 years available
            year2 = max(ALL_YEARS)
            year1 = year2 - 1
        else:
            year1, year2 = min(yrs), max(yrs)

        res = yoy_change_metric(metric, year1, year2, df_type=df_type or "income")
        print(f"\n=== Year-over-Year Change: {res['metric']} ({year1} → {year2}) ===")
        print(f"Start value ({year1}): {res['start_value']:,.2f}")
        print(f"End   value ({year2}): {res['end_value']:,.2f}")
        print(f"Absolute change: {res['absolute_change']:,.2f}")
        print(f"Percentage change: {res['pct_change']:.2f}%")

    elif intent == "ldr_compare":
        # Compare LDR between two specific years
        df = ldr_trend(start, end)
        print("\n=== LDR Comparison ===")
        print(df.to_string(index=False))
        if len(yrs) >= 2:
            y1, y2 = min(yrs), max(yrs)
            row1 = df[df["year"] == y1].iloc[0]
            row2 = df[df["year"] == y2].iloc[0]
            print(f"\nBetween {y1} and {y2}:")
            print(f"LDR {y1}: {row1['ldr']:.4f}")
            print(f"LDR {y2}: {row2['ldr']:.4f}")
            delta = row2["ldr"] - row1["ldr"]
            direction = "increased" if delta > 0 else "decreased" if delta < 0 else "remained stable"
            print(f"Lending relative to deposits has {direction} by {delta:.4f} points.")

    else:
        print(
            "Unknown query.\n"
            "Try examples like:\n"
            " - 'Show ROE trend from 2021 to 2024'\n"
            " - 'What is the year-over-year percentage change in Net profit between 2021 and 2024?'\n"
            " - 'Compare loan-to-deposit ratio between 2021 and 2023'\n"
            " - 'Run a 10% OPEX increase scenario for 2023'\n"
            " - 'Show NIM trend 2020-2024'\n"
            " - 'Show dashboard for 2021-2024'"
        )

# Convenience wrapper for notebooks
def financial_agent(question: str):
    return agent(question)

# ============================================
# 6. EVALUATION / TESTING MODE
# ============================================

EVAL_QUESTIONS = [
    "Show ROE trend from 2021 to 2024",
    "What is the year-over-year percentage change in Net profit between 2021 and 2024?",
    "Show cost to income ratio trend between 2021 and 2024",
    "Show NIM trend from 2021 to 2024",
    "Show NPL ratio between 2021 and 2024",
    "Compare loan-to-deposit ratio between 2021 and 2023",
    "Run a 10% OPEX increase scenario for 2023",
    "Run a 20% ECL shock scenario for 2023",
    "Show DuPont decomposition for 2023",
    "Forecast ROE based on 2021-2024 and give 1-year forecast",
    "Show FAB vs Peer ROE comparison between 2021 and 2024",
    "Show capital ratios between 2021 and 2024",
    "Show liquidity ratios between 2021 and 2024",
    "Show dashboard for 2021 to 2024",
]

def run_evaluation():
    print("\n========== RUNNING EVALUATION / TEST QUERIES ==========\n")
    for i, q in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n\n----- Test {i}: {q} -----")
        agent(q)
    print("\n========== EVALUATION COMPLETE ==========\n")

# ============================================
# 7. CLI
# ============================================

if __name__ == "__main__":
    

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            '  python fab_agent.py "your question"\n'
            "Or to run built-in test suite:\n"
            "  python fab_agent.py --eval\n"
        )
        sys.exit(0)

    first_arg = sys.argv[1]

    if first_arg in ("--eval", "--demo"):
        SHOW_PLOTS = False  # disable charts during evaluation
        run_evaluation()
        sys.exit(0)

    # Normal single-question mode
    q = " ".join(sys.argv[1:])
    agent(q)
