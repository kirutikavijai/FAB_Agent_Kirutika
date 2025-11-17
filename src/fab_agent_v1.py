#!/usr/bin/env python3
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use("seaborn-v0_8-darkgrid")

# ============================================
# 1. LOAD ALL CSV FILES
# ============================================

DATA_DIR = "data"

def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    df = pd.read_csv(path)
    
    # Auto-strip whitespace & quotes
    df.columns = df.columns.str.strip().str.replace("'", "").str.replace('"', "")
    if "metric" in df.columns:
        df["metric"] = df["metric"].astype(str).str.strip().str.replace("'", "").str.replace('"', "")
    
    return df

income_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\income_statement.csv")
balance_df = load_csv("E:\\workspace\\FAB Agents\\FAB_Agent_Kirutika\\data\\balance_sheet.csv")
credit_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\credit_quality.csv")
capital_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\capital.csv")
liquidity_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\liquidity.csv")

peer_income_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\peer_income_statement.csv")
peer_balance_df = load_csv("E:\workspace\FAB Agents\FAB_Agent_Kirutika\data\peer_balance_sheet.csv")

ALL_YEARS = sorted(balance_df["year"].unique())

# ============================================
# 2. BASIC UTILITIES
# ============================================

def query_metric(df, metric, years):
    return (
        df[(df["metric"] == metric) & (df["year"].isin(years))]
        .sort_values("year")
        .reset_index(drop=True)
    )

def compute_roe(np, eq): return np / eq if eq else np.nan
def compute_roa(np, assets): return np / assets if assets else np.nan
def compute_cir(opex, opinc): return opex / opinc if opinc else np.nan
def compute_nim(nii, ea): return nii / ea if ea else np.nan

def compute_cagr(start, end, n):
    if start <= 0 or n <= 0: return np.nan
    return (end / start) ** (1/n) - 1

def pretty(df, cols, title=None):
    if title: print(f"\n=== {title} ===")
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
        rows.append({"year": y, "net_profit": p, "equity": e, "roe": compute_roe(p, e)})

    df = pd.DataFrame(rows)
    cagr = compute_cagr(df["net_profit"].iloc[0], df["net_profit"].iloc[-1], len(yrs) - 1)
    return df, cagr

def cir_trend(start, end):
    yrs = list(range(start, end + 1))
    op_inc = query_metric(income_df, "Operating income", yrs)
    opex = query_metric(income_df, "Operating expenses", yrs)

    rows=[]
    for y in yrs:
        inc = op_inc.query("year==@y")["value"].iloc[0]
        ex = opex.query("year==@y")["value"].iloc[0]
        rows.append({"year": y, "operating_income": inc, "operating_expenses": ex, "cir": compute_cir(ex, inc)})
    return pd.DataFrame(rows)

def nim_trend(start, end, use_peer=False):
    inc = peer_income_df if use_peer else income_df
    bal = peer_balance_df if use_peer else balance_df
    yrs = list(range(start, end+1))

    nii = query_metric(inc, "Net interest income", yrs)
    ea = query_metric(bal, "Earning assets", yrs)

    rows = []
    for y in yrs:
        ni = nii.query("year==@y")["value"].iloc[0]
        eav = ea.query("year==@y")["value"].iloc[0]
        rows.append({"year": y, "net_interest_income": ni, "earning_assets": eav, "nim": compute_nim(ni, eav)})
    return pd.DataFrame(rows)

def npl_trend(start, end):
    df = credit_df[(credit_df["year"]>=start)&(credit_df["year"]<=end)].copy()
    df["npl_ratio"] = df["npl"]/df["total_loans"]
    df["ecl_ratio"] = df["ecl"]/df["total_loans"]
    return df

def ldr_trend(start,end,use_peer=False):
    bal = peer_balance_df if use_peer else balance_df
    yrs=list(range(start,end+1))
    loans=query_metric(bal,"Customer loans",yrs)
    deps=query_metric(bal,"Customer deposits",yrs)
    rows=[]
    for y in yrs:
        l=loans.query("year==@y")["value"].iloc[0]
        d=deps.query("year==@y")["value"].iloc[0]
        rows.append({"year":y,"loans":l,"deposits":d,"ldr":l/d})
    return pd.DataFrame(rows)

def capital_trend(start,end):
    df=capital_df[(capital_df["year"]>=start)&(capital_df["year"]<=end)].copy()
    df["car"]=df["total_capital"]/df["rwa"]
    df["cet1_ratio"]=df["cet1"]/df["rwa"]
    return df

def liquidity_trend(start,end):
    df=liquidity_df[(liquidity_df["year"]>=start)&(liquidity_df["year"]<=end)].copy()
    df["lcr"]=df["hqla"]/df["net_outflows"]
    df["nsfr"]=df["asf"]/df["rsf"]
    return df

def scenario_opex(year,pct):
    inc=query_metric(income_df,"Operating income",[year])["value"].iloc[0]
    opex=query_metric(income_df,"Operating expenses",[year])["value"].iloc[0]
    imp=query_metric(income_df,"Impairment charges",[year])["value"].iloc[0]
    eq=query_metric(balance_df,"Total equity",[year])["value"].iloc[0]

    base_np=inc-opex-imp
    new_np=inc-(opex*(1+pct/100))-imp
    return base_np,new_np,compute_roe(base_np,eq)*100,compute_roe(new_np,eq)*100

def scenario_ecl(year,pct):
    row=credit_df[credit_df["year"]==year].iloc[0]
    base=row["ecl"]
    shocked=base*(1+pct/100)
    np_base=query_metric(income_df,"Net profit",[year])["value"].iloc[0]
    eq=query_metric(balance_df,"Total equity",[year])["value"].iloc[0]
    new_np=np_base-(shocked-base)
    return np_base,new_np,base,shocked,compute_roe(np_base,eq)*100,compute_roe(new_np,eq)*100

def dupont(year):
    npv=query_metric(income_df,"Net profit",[year])["value"].iloc[0]
    inc=query_metric(income_df,"Operating income",[year])["value"].iloc[0]
    assets=query_metric(balance_df,"Total assets",[year])["value"].iloc[0]
    eq=query_metric(balance_df,"Total equity",[year])["value"].iloc[0]

    pm=npv/inc
    at=inc/assets
    em=assets/eq
    return {"year":year,"profit_margin":pm,"asset_turnover":at,"equity_multiplier":em,"roe":pm*at*em}

def forecast_roe(start,end,h=1):
    df,_=roe_trend(start,end)
    x=df["year"].values
    y=df["roe"].values
    m,b=np.polyfit(x,y,1)
    future=[{"year":end+i,"forecast_roe":m*(end+i)+b} for i in range(1,h+1)]
    return df,pd.DataFrame(future)

# ============================================
# 4. LLM PARSER
# ============================================

def llm_parse(q):
    t=q.lower()
    if "dashboard" in t: intent="dashboard"
    elif "compare" in t: intent="peer"
    elif "forecast" in t: intent="forecast"
    elif "dupont" in t or "du pont" in t: intent="dupont"
    elif "cir" in t or "cost to income" in t: intent="cir"
    elif "nim" in t: intent="nim"
    elif "npl" in t: intent="npl"
    elif "ldr" in t: intent="ldr"
    elif "car" in t or "cet1" in t: intent="capital"
    elif "liquidity" in t or "lcr" in t or "nsfr" in t: intent="liq"
    elif "scenario" in t and "ecl" in t: intent="scenario_ecl"
    elif "scenario" in t or "opex" in t or "operating expense" in t: intent="scenario_opex"
    elif "roe" in t: intent="roe"
    else: intent="unknown"

    yrs=re.findall(r"(20\d{2})",t)
    yrs=[int(y) for y in yrs] if yrs else ALL_YEARS

    pct=re.findall(r"(\d+)%",t)
    pct=int(pct[0]) if pct else None

    return {"intent":intent,"years":yrs,"pct":pct}

# ============================================
# 5. AGENT
# ============================================

def agent(question):
    task=llm_parse(question)
    intent=task["intent"]
    yrs=task["years"]
    pct=task["pct"]

    start,end=min(yrs),max(yrs)

    print("\nParsed:",task)

    if intent=="roe":
        df,c=roe_trend(start,end)
        pretty(df,["year","net_profit","equity","roe"],"ROE Trend")
        plt.plot(df["year"],df["roe"],marker="o");plt.title("ROE Trend");plt.show()
        print("CAGR:",c)

    elif intent=="cir":
        df=cir_trend(start,end)
        pretty(df,["year","operating_income","operating_expenses","cir"],"CIR Trend")
        plt.plot(df["year"],df["cir"]);plt.title("CIR");plt.show()

    elif intent=="nim":
        df=nim_trend(start,end)
        pretty(df,["year","net_interest_income","earning_assets","nim"],"NIM Trend")
        plt.plot(df["year"],df["nim"]);plt.title("NIM");plt.show()

    elif intent=="npl":
        df=npl_trend(start,end)
        print(df)
        plt.plot(df["year"],df["npl_ratio"]);plt.title("NPL Ratio");plt.show()

    elif intent=="ldr":
        df=ldr_trend(start,end)
        pretty(df,["year","loans","deposits","ldr"])
        plt.plot(df["year"],df["ldr"]);plt.title("LDR");plt.show()

    elif intent=="capital":
        df=capital_trend(start,end)
        print(df)
        plt.plot(df["year"],df["car"]);plt.title("CAR");plt.show()

    elif intent=="liq":
        df=liquidity_trend(start,end)
        print(df)
        plt.plot(df["year"],df["lcr"]);plt.title("LCR");plt.show()

    elif intent=="scenario_opex":
        year=end
        pct=pct or 10
        base,new,br,nr=scenario_opex(year,pct)
        print("Base NP:",base,"New NP:",new)
        print("Base ROE:",br,"New ROE:",nr)
        plt.bar(["Base","Scenario"],[base,new]);plt.title("NP Scenario");plt.show()

    elif intent=="scenario_ecl":
        year=end
        pct=pct or 20
        print(scenario_ecl(year,pct))

    elif intent=="dupont":
        print(dupont(end))

    elif intent=="forecast":
        hist,fc=forecast_roe(start,end)
        print(hist);print(fc)
        plt.plot(hist["year"],hist["roe"],label="hist")
        plt.plot(fc["year"],fc["forecast_roe"],label="forecast")
        plt.legend();plt.show()

    elif intent=="peer":
        fab,_=roe_trend(start,end)
        peer,_=roe_trend(start,end,use_peer=True)
        plt.plot(fab["year"],fab["roe"],label="FAB")
        plt.plot(peer["year"],peer["roe"],label="Peer")
        plt.legend();plt.title("ROE Comparison");plt.show()

    elif intent=="dashboard":
        plt.figure(figsize=(10,8))
        roe,_=roe_trend(start,end)
        cir=cir_trend(start,end)
        nim=nim_trend(start,end)
        npl=npl_trend(start,end)

        plt.subplot(2,2,1);plt.plot(roe["year"],roe["roe"]);plt.title("ROE")
        plt.subplot(2,2,2);plt.plot(cir["year"],cir["cir"]);plt.title("CIR")
        plt.subplot(2,2,3);plt.plot(nim["year"],nim["nim"]);plt.title("NIM")
        plt.subplot(2,2,4);plt.plot(npl["year"],npl["npl_ratio"]);plt.title("NPL Ratio")
        plt.tight_layout();plt.show()

    else:
        print("Unknown query. Try: ROE, CIR, NIM, NPL, LDR, capital, liquidity, scenario, forecast, dupont, dashboard, compare")

# ============================================
# 6. CLI
# ============================================

if __name__=="__main__":
    if len(sys.argv)<2:
        print('Usage:\n python fab_agent.py "your question"')
        sys.exit(0)
    q=" ".join(sys.argv[1:])
    agent(q)
