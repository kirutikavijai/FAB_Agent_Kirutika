import re

def llm_parse(question: str, all_years):
    t = question.lower()

    # Defaults
    intent = "unknown"
    metric = None
    df_type = None

    # Metric detection
    metric_map = {
        "net profit": ("Net profit", "income"),
        "total assets": ("Total assets", "balance"),
        "total equity": ("Total equity", "balance"),
        "customer loans": ("Customer loans", "balance"),
        "customer deposits": ("Customer deposits", "balance"),
    }

    for key, (metric_name, df_t) in metric_map.items():
        if key in t:
            metric = metric_name
            df_type = df_t

    # Intent detection rules
    rules = [
        ("yoy_metric",   ["year-over-year", "yoy", "percentage change"]),
        ("dashboard",    ["dashboard"]),
        ("ldr_compare",  ["compare", "ldr", "loan-to-deposit", "loan to deposit"]),
        ("peer",         ["compare", "roe"]),
        ("forecast",     ["forecast"]),
        ("dupont",       ["dupont", "du pont"]),
        ("cir",          ["cir", "cost to income"]),
        ("nim",          ["nim"]),
        ("npl",          ["npl"]),
        ("ldr",          ["ldr"]),
        ("capital",      ["car", "cet1"]),
        ("liq",          ["lcr", "liquidity", "nsfr"]),
        ("scenario_ecl", ["scenario", "ecl"]),
        ("scenario_opex",["scenario", "opex", "operating expense"]),
        ("roe",          ["roe"]),
    ]

    for name, keywords in rules:
        if any(k in t for k in keywords):
            intent = name
            break

    years = re.findall(r"(20\d{2})", t)
    years = [int(y) for y in years] if years else all_years

    pct = re.findall(r"(\d+)%", t)
    pct = int(pct[0]) if pct else None

    return {
        "intent": intent,
        "years": years,
        "pct": pct,
        "metric": metric,
        "df_type": df_type,
    }
