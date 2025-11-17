#!/usr/bin/env python3
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-darkgrid")

# 1. Synthetic FAB Financial Data

income_rows = []
for year, op_inc, opex, imp in [
    (2021, 21000, 8000, 3200),
    (2022, 23000, 8300, 2800),
    (2023, 25500, 8700, 2600),
    (2024, 29000, 9100, 2400),
]:
    net = op_inc - opex - imp
    income_rows.extend([
        {"year": year, "metric": "Net profit", "value": net},
        {"year": year, "metric": "Operating income", "value": op_inc},
        {"year": year, "metric": "Operating expenses", "value": opex},
        {"year": year, "metric": "Impairment charges", "value": imp},
    ])

income_df = pd.DataFrame(income_rows)

balance_df = pd.DataFrame([
    {"year": 2021, "metric": "Total equity", "value": 65000},
    {"year": 2022, "metric": "Total equity", "value": 67000},
    {"year": 2023, "metric": "Total equity", "value": 69000},
    {"year": 2024, "metric": "Total equity", "value": 71000},
])


# 2. Data Query Utilities

def query_metric(df, metric, years):
    return (
        df[(df["metric"] == metric) & (df["year"].isin(years))]
        .sort_values("year")
        .reset_index(drop=True)
    )

def compute_roe(net_profit, equity):
    return net_profit / equity if equity else np.nan

def compute_cagr(start_val, end_val, periods):
    if start_val <= 0 or periods <= 0:
        return np.nan
    return (end_val / start_val) ** (1 / periods) - 1

# 3. ROE Trend Analysis Engine

def analyze_roe_change(start_year, end_year):
    years = list(range(start_year, end_year + 1))

    profit = query_metric(income_df, "Net profit", years)
    equity = query_metric(balance_df, "Total equity", years)

    rows = []
    for y in years:
        p = profit.loc[profit["year"] == y, "value"].iloc[0]
        e = equity.loc[equity["year"] == y, "value"].iloc[0]
        rows.append({"year": y, "net_profit": p, "equity": e, "roe": compute_roe(p, e)})

    df = pd.DataFrame(rows)
    cagr = compute_cagr(df["net_profit"].iloc[0], df["net_profit"].iloc[-1], len(years) - 1)
    return df, cagr


# 4. Scenario Analysis Engine


def simulate_opex_change(year, pct):
    inc = query_metric(income_df, "Operating income", [year])["value"].iloc[0]
    opex = query_metric(income_df, "Operating expenses", [year])["value"].iloc[0]
    imp = query_metric(income_df, "Impairment charges", [year])["value"].iloc[0]
    eq = query_metric(balance_df, "Total equity", [year])["value"].iloc[0]

    base_np = inc - opex - imp
    base_roe = compute_roe(base_np, eq)

    new_opex = opex * (1 + pct / 100)
    new_np = inc - new_opex - imp
    new_roe = compute_roe(new_np, eq)

    return base_np, new_np, base_roe * 100, new_roe * 100


# 5. Charting Utilities

def plot_trend(df, y_col, title, ylabel):
    plt.figure(figsize=(8, 4))
    plt.plot(df["year"], df[y_col], marker="o", linewidth=2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def plot_scenario(base_np, new_np, base_roe, new_roe, year):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(["Base", "Scenario"], [base_np, new_np])
    plt.title(f"Net Profit Impact {year}")
    plt.ylabel("AED mn")

    plt.subplot(1, 2, 2)
    plt.bar(["Base", "Scenario"], [base_roe, new_roe])
    plt.title(f"ROE Impact {year}")
    plt.ylabel("ROE %")

    plt.tight_layout()
    plt.show()


# 6. LLM-style Natural Language Parser

def llm_parse(q):
    q_low = q.lower()

    # Intent
    if "roe" in q_low and ("change" in q_low or "trend" in q_low or "improve" in q_low):
        intent = "roe_trend"
    elif "scenario" in q_low or "what if" in q_low or "opex" in q_low:
        intent = "scenario"
    else:
        intent = "unknown"

    # Years
    years = re.findall(r"(20\d{2})", q_low)
    years = [int(y) for y in years] if years else [2021, 2022, 2023, 2024]

    # Scenario %
    pct = re.findall(r"(\d+)%", q_low)
    pct = int(pct[0]) if pct else None

    return {"intent": intent, "years": years, "pct": pct, "raw": q}



# 7. Supervisor Agent


def financial_agent(question):
    task = llm_parse(question)
    intent = task["intent"]
    years = task["years"]
    pct = task["pct"]

    print("\n Parsed:", task, "\n")

    # ROE TREND 
    if intent == "roe_trend":
        start, end = min(years), max(years)

        df, cagr = analyze_roe_change(start, end)
        print(df)

        plot_trend(df, "roe", "ROE Trend", "ROE Ratio")

        print(
            f"\nROE improved from {df['roe'].iloc[0]*100:.2f}% "
            f"to {df['roe'].iloc[-1]*100:.2f}% between {start} and {end}."
        )
        print(f"Net Profit CAGR: {cagr*100:.2f}%")
        return

    # SCENARIO
    if intent == "scenario":
        year = years[-1]
        pct = pct or 10

        base_np, new_np, base_roe, new_roe = simulate_opex_change(year, pct)

        print(f"\nBase NP: {base_np:.2f} | Scenario NP: {new_np:.2f}")
        print(f"Base ROE: {base_roe:.2f}% | New ROE: {new_roe:.2f}%")

        plot_scenario(base_np, new_np, base_roe, new_roe, year)
        return

    print("I could not understand the request.")
    print("Try asking about ROE trend or a scenario analysis.")


# 8. CLI ENTRY POINT


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python agent.py \"your financial question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    financial_agent(question)
