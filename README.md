📊 FAB Financial Analysis Agent

A command-line Python tool for analyzing synthetic FAB financial statements using an intelligent, LLM-style agent.

The system supports natural-language queries, multi-step reasoning, financial ratios, scenario simulations, peer comparisons, trend analysis, and automatic chart generation — all without using any external LLM API.

🚀 Features
1. Profitability Calculations

Net Profit

ROE (Return on Equity)
ROE = Net Profit / Total Equity

ROA (Return on Assets)
ROA = Net Profit / Total Assets

Profit Margin (DuPont)
Profit Margin = Net Profit / Operating Income

Asset Turnover (DuPont)
Asset Turnover = Operating Income / Total Assets

Equity Multiplier (Financial Leverage)
Equity Multiplier = Total Assets / Total Equity

ROE (DuPont Decomposition)
ROE = Profit Margin × Asset Turnover × Equity Multiplier

2. Efficiency & Income Calculations

CIR (Cost-to-Income Ratio)
CIR = Operating Expenses / Operating Income

NIM (Net Interest Margin)
NIM = Net Interest Income / Earning Assets

CAGR (Compound Annual Growth Rate)
CAGR = (Ending / Beginning)^(1/n) - 1

3. Credit Quality Calculations

NPL Ratio
NPL Ratio = NPL / Total Loans

ECL Ratio
ECL Ratio = ECL / Total Loans

ECL Shock Scenario:

Shocked ECL = ECL * (1 + pct/100)
New Net Profit = Old NP - (Shocked ECL - Base ECL)
Base ROE = Original NP / Equity
New ROE  = New NP / Equity

4. Capital Adequacy Calculations

CAR (Capital Adequacy Ratio)
CAR = Total Capital / RWA

CET1 Ratio
CET1 = CET1 / RWA

5. Liquidity Calculations

LCR (Liquidity Coverage Ratio)
LCR = HQLA / Net Outflows

NSFR (Net Stable Funding Ratio)
NSFR = ASF / RSF

6. Loan & Deposit Analytics

LDR (Loan-to-Deposit Ratio)
LDR = Customer Loans / Customer Deposits

7. Scenario Simulations

OPEX Stress Test

ECL Shock Simulation

8. Forecasting

Linear Regression ROE Forecast

9. Peer Bank Analytics

Compare FAB with peer bank metrics (ROE, NIM, CIR, etc.).

10. Dashboard Aggregation

Automatically generates a 4-panel dashboard showing:

ROE

CIR

NIM

NPL

🖥️ Usage

Run the agent from the terminal by passing a question directly as a string.

ROE Trend Analysis
python agent.py "Show how ROE changed between 2021 and 2024"

Scenario / What-if Analysis
python agent.py "What if operating expenses increase by 15% in 2024?"

Simple Trend Request
python agent.py "Give me ROE trend chart"
