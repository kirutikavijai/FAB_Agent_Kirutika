import matplotlib.pyplot as plt

from core.utils import pretty
from core.parser import llm_parse

from analysis.roe import roe_trend
from analysis.cir import cir_trend
from analysis.nim import nim_trend
from analysis.npl import npl_trend
from analysis.ldr import ldr_trend
from analysis.capital import capital_trend
from analysis.liquidity import liquidity_trend
from analysis.dupont import dupont
from analysis.scenario import scenario_opex, scenario_ecl
from analysis.forecast import forecast_roe
from analysis.yoy import yoy_change_metric

# Default matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")

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

def run_agent(question: str, data: dict, all_years, show_plots: bool = True):
    task = llm_parse(question, all_years)
    intent = task["intent"]
    years  = task["years"]
    pct    = task["pct"]
    metric = task["metric"]
    df_type = task["df_type"]

    start, end = min(years), max(years)

    income_df      = data["income"]
    balance_df     = data["balance"]
    credit_df      = data["credit"]
    capital_df     = data["capital"]
    liquidity_df   = data["liquidity"]
    peer_income_df = data["peer_income"]
    peer_balance_df= data["peer_balance"]

    print("\nParsed:", task)

    # ------------ CORE INTENTS ------------
    if intent == "roe":
        df, cagr = roe_trend(start, end, income_df, balance_df)
        pretty(df, ["year", "net_profit", "equity", "roe"], "ROE Trend")
        print(f"CAGR of Net Profit ({start}-{end}): {cagr:.2%}")
        if show_plots:
            plt.plot(df["year"], df["roe"], marker="o")
            plt.title("ROE Trend")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.show()

    elif intent == "cir":
        df = cir_trend(start, end, income_df)
        pretty(df, ["year", "operating_income", "operating_expenses", "cir"], "CIR Trend")
        if show_plots:
            plt.plot(df["year"], df["cir"])
            plt.title("Cost-to-Income Ratio")
            plt.xlabel("Year")
            plt.ylabel("CIR")
            plt.show()

    elif intent == "nim":
        df = nim_trend(start, end, income_df, balance_df)
        pretty(df, ["year", "net_interest_income", "earning_assets", "nim"], "NIM Trend")
        if show_plots:
            plt.plot(df["year"], df["nim"])
            plt.title("Net Interest Margin")
            plt.xlabel("Year")
            plt.ylabel("NIM")
            plt.show()

    elif intent == "npl":
        df = npl_trend(start, end, credit_df)
        print("\n=== NPL & ECL Ratios ===")
        print(df.to_string(index=False))
        if show_plots:
            plt.plot(df["year"], df["npl_ratio"])
            plt.title("NPL Ratio")
            plt.xlabel("Year")
            plt.ylabel("NPL Ratio")
            plt.show()

    elif intent == "ldr":
        df = ldr_trend(start, end, balance_df)
        pretty(df, ["year", "loans", "deposits", "ldr"], "Loan-to-Deposit Ratio")
        if show_plots:
            plt.plot(df["year"], df["ldr"])
            plt.title("LDR")
            plt.xlabel("Year")
            plt.ylabel("LDR")
            plt.show()

    elif intent == "capital":
        df = capital_trend(start, end, capital_df)
        print("\n=== Capital Ratios ===")
        print(df.to_string(index=False))
        if show_plots:
            plt.plot(df["year"], df["car"])
            plt.title("CAR")
            plt.xlabel("Year")
            plt.ylabel("Capital Adequacy Ratio")
            plt.show()

    elif intent == "liq":
        df = liquidity_trend(start, end, liquidity_df)
        print("\n=== Liquidity Ratios ===")
        print(df.to_string(index=False))
        if show_plots:
            plt.plot(df["year"], df["lcr"])
            plt.title("LCR")
            plt.xlabel("Year")
            plt.ylabel("Liquidity Coverage Ratio")
            plt.show()

    elif intent == "scenario_opex":
        year = end
        pct  = pct or 10
        base_np, new_np, base_roe, new_roe = scenario_opex(year, pct, income_df, balance_df)
        print(f"\n=== OPEX Scenario for {year} (+{pct}%) ===")
        print(f"Base Net Profit: {base_np:,.2f}")
        print(f"New  Net Profit: {new_np:,.2f}")
        print(f"Base ROE: {base_roe:.2f}%")
        print(f"New  ROE: {new_roe:.2f}%")
        if show_plots:
            plt.bar(["Base", "Scenario"], [base_np, new_np])
            plt.title(f"Net Profit Scenario {year} (+{pct}%)")
            plt.show()

    elif intent == "scenario_ecl":
        year = end
        pct  = pct or 20
        np_base, new_np, base_ecl, shocked_ecl, base_roe, new_roe = scenario_ecl(
            year, pct, credit_df, income_df, balance_df
        )
        print(f"\n=== ECL Shock Scenario for {year} (+{pct}%) ===")
        print(f"Base ECL:     {base_ecl:,.2f}")
        print(f"Shocked ECL:  {shocked_ecl:,.2f}")
        print(f"Base Net Profit: {np_base:,.2f}")
        print(f"New  Net Profit: {new_np:,.2f}")
        print(f"Base ROE: {base_roe:.2f}%")
        print(f"New  ROE:  {new_roe:.2f}%")

    elif intent == "dupont":
        result = dupont(end, income_df, balance_df)
        print("\n=== DuPont Decomposition ===")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    elif intent == "forecast":
        hist_df, fut_df = forecast_roe(start, end, income_df, balance_df, horizon=1)
        print("\n=== Historical ROE ===")
        print(hist_df.to_string(index=False))
        print("\n=== Forecast ROE ===")
        print(fut_df.to_string(index=False))
        if show_plots:
            plt.plot(hist_df["year"], hist_df["roe"], label="Historical")
            plt.plot(fut_df["year"], fut_df["forecast_roe"], label="Forecast", linestyle="--")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.legend()
            plt.title("ROE Forecast")
            plt.show()

    elif intent == "peer":
        fab_df, _   = roe_trend(start, end, income_df, balance_df)
        peer_df, _  = roe_trend(start, end, peer_income_df, peer_balance_df)
        print("\n=== FAB vs Peer ROE Comparison ===")
        print("\nFAB:")
        print(fab_df.to_string(index=False))
        print("\nPeer:")
        print(peer_df.to_string(index=False))
        if show_plots:
            plt.plot(fab_df["year"], fab_df["roe"], label="FAB")
            plt.plot(peer_df["year"], peer_df["roe"], label="Peer")
            plt.legend()
            plt.title("ROE Comparison")
            plt.xlabel("Year")
            plt.ylabel("ROE")
            plt.show()

    elif intent == "dashboard":
        roe_df, _ = roe_trend(start, end, income_df, balance_df)
        cir_df    = cir_trend(start, end, income_df)
        nim_df    = nim_trend(start, end, income_df, balance_df)
        npl_df    = npl_trend(start, end, credit_df)

        if show_plots:
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
            print("\nDashboard (no plots):")
            print("\nROE:")
            print(roe_df.to_string(index=False))
            print("\nCIR:")
            print(cir_df.to_string(index=False))
            print("\nNIM:")
            print(nim_df.to_string(index=False))
            print("\nNPL:")
            print(npl_df.to_string(index=False))

    # ------------ NEW INTENTS ------------
    elif intent == "yoy_metric":
        if metric is None:
            metric  = "Net profit"
            df_type = "income"

        if len(years) < 2:
            year2 = max(all_years)
            year1 = year2 - 1
        else:
            year1, year2 = min(years), max(years)

        df_map = {
            "income": income_df,
            "balance": balance_df,
            "capital": capital_df,
            "liquidity": liquidity_df,
        }
        df_chosen = df_map.get(df_type or "income", income_df)

        res = yoy_change_metric(metric, year1, year2, df_chosen, df_type or "income")
        print(f"\n=== YoY Change: {res['metric']} ({year1} → {year2}) ===")
        print(f"Start value ({year1}): {res['start_value']:,.2f}")
        print(f"End   value ({year2}): {res['end_value']:,.2f}")
        print(f"Absolute change: {res['absolute_change']:,.2f}")
        print(f"Percentage change: {res['pct_change']:.2f}%")

    elif intent == "ldr_compare":
        df = ldr_trend(start, end, balance_df)
        print("\n=== LDR Comparison ===")
        print(df.to_string(index=False))
        if len(years) >= 2:
            y1, y2 = min(years), max(years)
            row1 = df[df["year"] == y1].iloc[0]
            row2 = df[df["year"] == y2].iloc[0]
            print(f"\nBetween {y1} and {y2}:")
            print(f"LDR {y1}: {row1['ldr']:.4f}")
            print(f"LDR {y2}: {row2['ldr']:.4f}")
            delta = row2["ldr"] - row1["ldr"]
            direction = "increased" if delta > 0 else "decreased" if delta < 0 else "remained stable"
            print(f"Lending relative to deposits has {direction} by {delta:.4f} points.")
        if show_plots:
            plt.plot(df["year"], df["ldr"])
            plt.title("LDR")
            plt.xlabel("Year")
            plt.ylabel("LDR")
            plt.show()

    else:
        print(
            "Unknown query.\n"
            "Try examples like:\n"
            " - 'Show ROE trend from 2021 to 2024'\n"
            " - 'What is the year-over-year percentage change in Net profit between 2021 and 2024?'\n"
            " - 'Compare loan-to-deposit ratio between 2021 and 2023'\n"
            " - 'Run a 10% OPEX increase scenario for 2023'\n"
            " - 'Show dashboard for 2021-2024'"
        )

def run_evaluation(data: dict, all_years, show_plots: bool = False):
    print("\n========== RUNNING EVALUATION / TEST QUERIES ==========\n")
    for i, q in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n\n----- Test {i}: {q} -----")
        run_agent(q, data, all_years, show_plots=show_plots)
    print("\n========== EVALUATION COMPLETE ==========\n")
