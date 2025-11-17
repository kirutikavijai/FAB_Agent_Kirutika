import os
import pandas as pd

# By default, use ./data under project root.
# You can override with FAB_DATA_DIR env variable if needed.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_DIR = os.getenv("FAB_DATA_DIR", DEFAULT_DATA_DIR)

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
    )

    if "metric" in df.columns:
        df["metric"] = (
            df["metric"].astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
        )

    if "peer" in df.columns:
        df["peer"] = (
            df["peer"].astype(str)
            .str.strip()
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
        )

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from DATA_DIR or absolute path."""
    if os.path.isabs(filename):
        path = "E:\\workspace\\FAB Agents\\FAB_Agent_Kirutika\\data\\"+filename
    else:
        path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    return _clean_df(df)

def load_all_data():
    """
    Loads all required FAB CSVs into a dict and returns:
      data: dict of DataFrames
      all_years: sorted list of years from balance sheet
    """
    data = {
        "income":        load_csv("income_statement.csv"),
        "balance":       load_csv("balance_sheet.csv"),
        "credit":        load_csv("credit_quality.csv"),
        "capital":       load_csv("capital.csv"),
        "liquidity":     load_csv("liquidity.csv"),
        "peer_income":   load_csv("peer_income_statement.csv"),
        "peer_balance":  load_csv("peer_balance_sheet.csv"),
    }

    balance_years = data["balance"]["year"].dropna().unique().tolist()
    all_years = sorted(int(y) for y in balance_years)

    return data, all_years
