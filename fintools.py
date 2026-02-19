from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
import os
import pandas as pd
import numpy as np
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import mongodb_connection as mongo
import yfinance as yf

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "pr-tart-sweatsuit-95"

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

# Ask user for API key and store in a variable
deepseek_api_key = "sk-daadb7c9968f4c07b802743b631164d1"

# Initialize DeepSeek model using the variable
model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=deepseek_api_key  # pass the key directly
)

"""
These functions are for the tool stock_health_check
Do not edit or change
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_financial_attribute(ticker_obj, attribute_name):
    try:
        data = getattr(ticker_obj, attribute_name)
        if data is None or data.empty:
            return pd.DataFrame()
        data.columns = pd.to_datetime(data.columns)
        return data
    except Exception:
        return pd.DataFrame()


def _fmt(val, decimals=2, suffix=""):
    """Format a float nicely, or return a dash if unavailable."""
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return str(val)


# ── Check functions ───────────────────────────────────────────────────────────

def check_revenue_growth(ticker_obj) -> dict:
    """Returns {'growth_pct': {year: pct, ...}, 'error': str|None}"""
    result = {"growth_pct": {}, "error": None}

    annual = _get_financial_attribute(ticker_obj, 'financials')
    if annual.empty or 'Total Revenue' not in annual.index:
        result["error"] = "Annual financial data (Total Revenue) not available."
        return result

    revenue = annual.loc["Total Revenue"].dropna()
    if len(revenue) < 2:
        result["error"] = "Insufficient revenue data for growth calculation."
        return result

    growth = revenue.pct_change(periods=-1) * 100
    growth = growth.iloc[::-1].dropna()

    result["growth_pct"] = {str(d.year): round(v, 2) for d, v in growth.head(5).items()}

    print("\n--- Revenue Growth ---")
    for yr, pct in result["growth_pct"].items():
        print(f"  {yr}: {pct:.2f}%")
    return result


def check_eps_history(ticker_obj) -> dict:
    """Returns {'trailing_eps': float|None, 'annual_eps': {year: eps}, 'quarterly_eps': {date: eps}}"""
    result = {"trailing_eps": None, "annual_eps": {}, "quarterly_eps": {}, "error": None}

    info = ticker_obj.info
    result["trailing_eps"] = info.get('trailingEps')

    annual = _get_financial_attribute(ticker_obj, 'financials')
    if not annual.empty and 'Basic EPS' in annual.index:
        eps_series = annual.loc["Basic EPS"].dropna().iloc[::-1]
        result["annual_eps"] = {str(d.year): round(v, 2) for d, v in eps_series.head(5).items()}

    qf = _get_financial_attribute(ticker_obj, 'quarterly_financials')
    if not qf.empty and 'Net Income' in qf.index and 'Diluted Average Shares' in qf.index:
        ni = qf.loc["Net Income"].dropna()
        shares = qf.loc["Diluted Average Shares"].dropna()
        common = ni.index.intersection(shares.index)
        if not common.empty:
            q_eps = (ni.reindex(common) / shares.reindex(common)).dropna().iloc[::-1]
            result["quarterly_eps"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in q_eps.head(5).items()}

    print("\n--- EPS History ---")
    print(f"  Trailing EPS: {result['trailing_eps']}")
    print(f"  Annual EPS: {result['annual_eps']}")
    print(f"  Quarterly EPS: {result['quarterly_eps']}")
    return result


def check_dividend_yield(ticker_obj) -> dict:
    """Returns {'dividend_yield_pct': float|None}
    
    yfinance may return dividendYield as a true decimal (0.0115 → 1.15%)
    OR, for some tickers, already as a percentage (1.15 → 1.15%).
    We normalise: if the raw value is > 1.0, assume it's already in % form.
    """
    result = {"dividend_yield_pct": None, "error": None}
    dy = ticker_obj.info.get('dividendYield')
    if dy is not None:
        # Values > 1.0 are already expressed as percentage points (e.g. 1.5 means 1.5%)
        # Values <= 1.0 are true decimals (e.g. 0.015 means 1.5%)
        pct = dy if dy > 1.0 else dy * 100
        result["dividend_yield_pct"] = round(pct, 2)

    print("\n--- Dividend Yield ---")
    print(f"  {_fmt(result['dividend_yield_pct'], suffix='%') if result['dividend_yield_pct'] else 'Not available'}")
    return result


def check_roe_history(ticker_obj) -> dict:
    """Returns {'annual_roe_pct': {year: pct}, 'quarterly_ttm_roe_pct': {date: pct}}"""
    result = {"annual_roe_pct": {}, "quarterly_ttm_roe_pct": {}, "error": None}

    af = _get_financial_attribute(ticker_obj, 'financials')
    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    if not af.empty and not ab.empty and \
       'Net Income' in af.index and 'Stockholders Equity' in ab.index:
        ni = af.loc["Net Income"].dropna()
        eq = ab.loc["Stockholders Equity"].dropna()
        common = ni.index.intersection(eq.index)
        if not common.empty:
            roe = (ni.reindex(common) / eq.reindex(common) * 100).dropna().iloc[::-1]
            result["annual_roe_pct"] = {str(d.year): round(v, 2) for d, v in roe.head(5).items()}

    qf = _get_financial_attribute(ticker_obj, 'quarterly_financials')
    qb = _get_financial_attribute(ticker_obj, 'quarterly_balance_sheet')
    if not qf.empty and not qb.empty and \
       'Net Income' in qf.index and 'Stockholders Equity' in qb.index:
        ni_q = qf.loc["Net Income"].dropna().sort_index()
        eq_q = qb.loc["Stockholders Equity"].dropna().sort_index()
        if len(ni_q) >= 4:
            ttm = ni_q.rolling(4, min_periods=4).sum()
            ttm_roe = {}
            for d, v in ttm.items():
                if d in eq_q.index and eq_q.loc[d] != 0:
                    ttm_roe[d.strftime('%Y-%m-%d')] = round(v / eq_q.loc[d] * 100, 2)
            result["quarterly_ttm_roe_pct"] = dict(
                list(sorted(ttm_roe.items(), reverse=True))[:5]
            )

    print("\n--- Return on Equity (ROE) ---")
    print(f"  Annual ROE: {result['annual_roe_pct']}")
    print(f"  Quarterly TTM ROE: {result['quarterly_ttm_roe_pct']}")
    return result


def check_free_cash_flow(ticker_obj) -> dict:
    """Returns {'annual_fcf': {year: val}, 'quarterly_fcf': {date: val}}"""
    result = {"annual_fcf": {}, "quarterly_fcf": {}, "error": None}

    # yfinance label varies by ticker/version — check both spellings
    CAPEX_LABELS = ["Capital Expenditures", "Capital Expenditure"]

    def _compute_fcf(cashflow_df) -> pd.Series | None:
        """Returns a Series on success, None on failure."""
        if cashflow_df.empty or 'Operating Cash Flow' not in cashflow_df.index:
            return None

        capex_label = next((l for l in CAPEX_LABELS if l in cashflow_df.index), None)
        if capex_label is None:
            return None

        ocf   = cashflow_df.loc["Operating Cash Flow"].dropna()
        capex = cashflow_df.loc[capex_label].dropna()
        common = ocf.index.intersection(capex.index)
        if common.empty:
            return None

        fcf = (ocf.reindex(common) + capex.reindex(common)).dropna().iloc[::-1]
        return fcf if not fcf.empty else None

    af_fcf = _compute_fcf(_get_financial_attribute(ticker_obj, 'cashflow'))
    if af_fcf is not None:
        result["annual_fcf"] = {str(d.year): round(v, 2) for d, v in af_fcf.head(5).items()}

    qf_fcf = _compute_fcf(_get_financial_attribute(ticker_obj, 'quarterly_cashflow'))
    if qf_fcf is not None:
        result["quarterly_fcf"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in qf_fcf.head(5).items()}

    print("\n--- Free Cash Flow ---")
    print(f"  Annual FCF:    {result['annual_fcf']}")
    print(f"  Quarterly FCF: {result['quarterly_fcf']}")
    return result


def check_operating_cash_flow_ratio(ticker_obj) -> dict:
    """Returns {'annual_ocf_ratio_pct': {year: pct}, 'quarterly_ocf_ratio_pct': {date: pct}}"""
    result = {"annual_ocf_ratio_pct": {}, "quarterly_ocf_ratio_pct": {}, "error": None}

    def _compute(fin_df, cf_df):
        if fin_df.empty or cf_df.empty or \
           'Total Revenue' not in fin_df.index or \
           'Operating Cash Flow' not in cf_df.index:
            return {}
        rev = fin_df.loc["Total Revenue"].dropna()
        ocf = cf_df.loc["Operating Cash Flow"].dropna()
        common = rev.index.intersection(ocf.index)
        if common.empty:
            return {}
        ratio = (ocf.reindex(common) / rev.reindex(common) * 100) \
                    .replace([float('inf'), -float('inf')], pd.NA).dropna().iloc[::-1]
        return ratio

    ar = _compute(_get_financial_attribute(ticker_obj, 'financials'),
                  _get_financial_attribute(ticker_obj, 'cashflow'))
    if not isinstance(ar, dict):
        result["annual_ocf_ratio_pct"] = {str(d.year): round(v, 2) for d, v in ar.head(5).items()}

    qr = _compute(_get_financial_attribute(ticker_obj, 'quarterly_financials'),
                  _get_financial_attribute(ticker_obj, 'quarterly_cashflow'))
    if not isinstance(qr, dict):
        result["quarterly_ocf_ratio_pct"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in qr.head(5).items()}

    print("\n--- Operating Cash Flow Ratio ---")
    print(f"  Annual OCF Ratio: {result['annual_ocf_ratio_pct']}")
    print(f"  Quarterly OCF Ratio: {result['quarterly_ocf_ratio_pct']}")
    return result


def check_ar_turnover_days(ticker_obj) -> dict:
    """Returns {'annual_dso_days': {year: days}}"""
    result = {"annual_dso_days": {}, "error": None}

    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    af = _get_financial_attribute(ticker_obj, 'financials')
    if ab.empty or af.empty or \
       'Accounts Receivable' not in ab.index or 'Total Revenue' not in af.index:
        result["error"] = "Data not available."
        return result

    ar = ab.loc["Accounts Receivable"].dropna().sort_index()
    rev = af.loc["Total Revenue"].dropna().sort_index()
    dso = {}
    for i in range(1, len(ar)):
        cur_d, prev_d = ar.index[i], ar.index[i-1]
        if cur_d in rev.index:
            cur_rev = rev.loc[cur_d]
            avg_ar = (ar.loc[prev_d] + ar.loc[cur_d]) / 2
            if cur_rev > 0 and avg_ar > 0:
                dso[str(cur_d.year)] = round(365 / (cur_rev / avg_ar), 2)

    result["annual_dso_days"] = dict(list(sorted(dso.items(), reverse=True))[:5])
    print("\n--- Days Sales Outstanding (DSO) ---")
    print(f"  {result['annual_dso_days']}")
    return result


def check_inventory_turnover_days(ticker_obj) -> dict:
    """Returns {'annual_dio_days': {year: days}}"""
    result = {"annual_dio_days": {}, "error": None}

    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    af = _get_financial_attribute(ticker_obj, 'financials')
    if ab.empty or af.empty or \
       'Inventory' not in ab.index or 'Cost Of Revenue' not in af.index:
        result["error"] = "Data not available."
        return result

    inv = ab.loc["Inventory"].dropna().sort_index()
    cor = af.loc["Cost Of Revenue"].dropna().sort_index()
    dio = {}
    for i in range(1, len(inv)):
        cur_d, prev_d = inv.index[i], inv.index[i-1]
        if cur_d in cor.index:
            cur_cor = cor.loc[cur_d]
            avg_inv = (inv.loc[prev_d] + inv.loc[cur_d]) / 2
            if cur_cor > 0 and avg_inv > 0:
                dio[str(cur_d.year)] = round(365 / (cur_cor / avg_inv), 2)

    result["annual_dio_days"] = dict(list(sorted(dio.items(), reverse=True))[:5])
    print("\n--- Days Inventory Outstanding (DIO) ---")
    print(f"  {result['annual_dio_days']}")
    return result


def check_supervisor_holdings(ticker_obj) -> dict:
    print("\n--- Supervisor Stock Holdings ---")
    print("  Not directly available via yfinance.")
    return {"note": "Senior supervisor stock holdings require specialized regulatory filings or alternative data providers."}


# ── KD & stock data ───────────────────────────────────────────────────────────

def stock_data_(stock_id):
    for suffix in [".TW", ".TWO"]:
        df = yf.download(f"{stock_id}{suffix}", period="90d", interval="1d")
        df.dropna(inplace=True)
        if not df.empty and len(df) >= 10:
            return df.sort_values(by="Date", ascending=True)
    return {"error": "資料不足或代號錯誤"}


def _calcu_KD_from_df(df: pd.DataFrame, period=9, init_k=50.0, init_d=50.0):
    """Core KD calculation shared by daily and weekly functions."""
    if isinstance(df, dict) or df.empty or len(df) < period:
        return None, None
    df = df.copy()
    for c in ["High", "Low", "Close"]:
        df[c] = df[c].astype(float)

    low_min  = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    rsv = ((df["Close"] - low_min) / (high_max - low_min) * 100).dropna()

    if rsv.empty:
        return None, None

    k, d = float(init_k), float(init_d)
    for r in rsv.values:
        k = (2/3) * k + (1/3) * float(r)
        d = (2/3) * d + (1/3) * k

    return round(k, 2), round(d, 2)


def calcu_KD(code, period=9, init_k=50.0, init_d=50.0):
    df = stock_data_(code)
    if isinstance(df, dict):
        return None, None
    df = df.copy()
    for c in ["High", "Low", "Close"]:
        df[c] = df[c].astype(float)

    low_min  = df["Low"].rolling(period).min()
    high_max = df["High"].rolling(period).max()
    rsv = ((df["Close"] - low_min) / (high_max - low_min) * 100).dropna()

    return _calcu_KD_from_df(df, period, init_k, init_d)


def calcu_KD_weekly(code, period=9, init_k=50.0, init_d=50.0):
    """KD indicator based on weekly candles (uses 2 years of data)."""
    for suffix in [".TW", ".TWO"]:
        df = yf.download(f"{code}{suffix}", period="2y", interval="1wk")
        df.dropna(inplace=True)
        if not df.empty and len(df) >= period:
            df = df.sort_index(ascending=True)
            return _calcu_KD_from_df(df, period, init_k, init_d)
    return None, None

#### Functions for stock_health_check end here ####


###########Funtions for tools#####################

def stock_data(stock_id):
  """
  This tool return a dataframe of a stock for the past 90 days
  """
  stock_type = "TW"
  ticker = f"{stock_id}.{stock_type}"
  df = yf.download(ticker, period="90d", interval="1d")
  df.dropna(inplace=True)
  if df.empty or len(df) < 10:
      stock_type = "TWO"
      ticker = f"{stock_id}.{stock_type}"
      df = yf.download(ticker, period="90d", interval="1d")
      df.dropna(inplace=True)

      if len(df) < 2:
          return {"error": "No information found"}

  df = df.sort_values(by="Date", ascending=True)
  return df, stock_type

def company_info(stock_id):
  ticker_symbol = f"{stock_id}.TW" # Example stock
  ticker = yf.Ticker(ticker_symbol)

   # Get the longBusinessSummary
  if ticker.info.get("longBusinessSummary"):
    long_business_summary = ticker.info.get("longBusinessSummary")
    sector = ticker.info.get("sector") if ticker.info.get("sector") else f"Couldn't find the sector of {ticker_symbol}"
    industry = ticker.info.get("industry") if ticker.info.get("industry") else f"Couldn't find industry of {ticker_symbol}"
  elif yf.Ticker(f"{stock_id}.TWO").info.get("longBusinessSummary"):
    long_business_summary = yf.Ticker(f"{stock_id}.TWO").info.get("longBusinessSummary")
    sector = yf.Ticker(f"{stock_id}.TWO").info.get("sector") if yf.Ticker(f"{stock_id}.TWO").info.get("sector") else f"Couldn't find the sector of {ticker_symbol}"
    industry = yf.Ticker(f"{stock_id}.TWO").info.get("industry") if yf.Ticker(f"{stock_id}.TWO").info.get("industry") else f"Couldn't find the industry of {ticker_symbol}"
     

  return {"stock":stock_id, "summary":long_business_summary, "sector": sector, "industry": industry}

def calcu_KD_d(code, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate K and D values for a stock with a day as a unit.
    """
    # Ensure correct dtype
    df, stock_type = stock_data(code)
    df = df.copy()
    for c in ["High", "Low", "Close"]:
        df[c] = df[c].astype(float)

    highs = df["High"]
    lows = df["Low"]
    closes = df["Close"]

    # --- Compute RSV ---
    low_min = lows.rolling(period).min()
    high_max = highs.rolling(period).max()

    rsv = (closes - low_min) / (high_max - low_min) * 100
    rsv = rsv.dropna()

    if rsv.empty:
        raise ValueError("RSV計算失敗，資料可能不足")

    # --- K/D smoothing ---
    k = float(init_k)
    d = float(init_d)

    for r in rsv.values:   # <-- use .values to ensure float
        r = float(r)
        k = (2/3) * k + (1/3) * r
        d = (2/3) * d + (1/3) * k

    return round(k, 2), round(d, 2)



def calcu_KD_w_multiple_(codes:list, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for multiple/many stocks.
    Extracts weekly High, Low, Close.
    Return the format {stock:{k:k_value, d:d_value}, stock_2...}
    """
    result = {}
    # Ensure correct dtype
    for code in codes:
        try:
            # --- Load data and ensure correct dtype ---
            df, stock_type = stock_data(code)
            # --- Flatten columns (optional, easier to work with) ---
            # Select only the first (or relevant) ticker
            df = df.xs(f"{code}.{stock_type}", axis=1, level=1)
        
            # --- Ensure correct dtype ---
            for c in ["High", "Low", "Close"]:
                df[c] = df[c].astype(float)
        
            # --- Resample to weekly (Friday) ---
            df = df.resample("W-FRI").agg({
              "High": "max",
              "Low": "min",
              "Close": "last"
            }).dropna()
        
            df = df.reset_index()  # optional, for easier handling
        
        
            highs = df["High"]
            lows = df["Low"]
            closes = df["Close"]
        
        
            # --- Compute RSV ---
            low_min = lows.rolling(period).min()
            high_max = highs.rolling(period).max()
            rsv = (closes - low_min) / (high_max - low_min) * 100
            rsv = rsv.dropna()
        
            if rsv.empty:
                raise ValueError("RSV calculation failed: insufficient weekly data")
        
            # --- Compute K/D ---
            k_values = []
            d_values = []
            k = float(init_k)
            d = float(init_d)
        
            for r in rsv.values:
                k = (2/3) * k + (1/3) * r
                d = (2/3) * d + (1/3) * k
                k_values.append(k)
                d_values.append(d)
        
            # --- Attach K/D to weekly dataframe ---
            df = df.iloc[-len(k_values):].copy()  # align lengths
            df["K"] = k_values
            df["D"] = d_values
        
            result[code] = {"k": k_values[-1], "d":d_values[-1]}

        except:
            result[code] = "No information was fetched"
    
    return str(result)

########### Put Tools Here #####################
#### Database ####
@tool
def check_database_connection():
    """
    Checks the MongoDB database connection.
    """
    return mongo.check_db_connection()

@tool
def fetch_all_stocks():
    """
    Fetch all stocks information stored in the watchlist database 
    The returned values is a string {collection_name: [ {stock: stock_code, ...}, {}]
    """
    return str(mongo.fetch_all())
    
@tool
def add_collection(collection_name):
    """
    Create a new collection if it does not already exist.
    """
    return mongo.add_collection(collection_name)

@tool
def list_collections():
    """
    List all collections in the database.
    """
    return mongo.list_collections()

@tool
def collection_information(collection_name):
    """
    Users input a collection name as an argument. This tool reads and returns the stocks in the collection from the MongoDB watchlist database. The returned information includes the stock codes and their corresponding company information (summary, sector, industry) that are stored in the specified collection. The output is formatted from a list to a string representation of the watchlist data.
    """
    watchlist = mongo.find_documents(collection_name)
    return str(watchlist)

@tool
def add_to_watchlist(stock_code, collection_name):
    """
    Adds a stock code to the watchlist according to the specified collection in the MongoDB database stock_watchlist.
    """
    company_info_result = company_info(stock_code)
    if "error" in company_info_result:
        print(f"Failed to retrieve company info for stock code {stock_code}: {company_info_result['error']}")
        return f"Failed to retrieve company info for stock code {stock_code}: {company_info_result['error']}"
    result = mongo.insert_document(collection_name, company_info_result)
    if result != None:
        print(f"Stock code {stock_code} added to watchlist in collection {collection_name}.")
        return f"Stock code {stock_code} added to watchlist in collection {collection_name}."
    else:
        print(f"Failed to add stock code {stock_code} to watchlist in collection {collection_name}.")
        return f"Failed to add stock code {stock_code} to watchlist in collection {collection_name}."


@tool
def add_m_to_watchlist(stock_codes, collection_name):
    """
    Add multiple stock codes to the watchlist according to the specified collection in the MongoDB database stock_watchlist.
    """
    results = []
    for stock_code in stock_codes:
        company_info_result = company_info(stock_code)
        if "error" in company_info_result:
            print(f"Failed to retrieve company info for stock code {stock_code}: {company_info_result['error']}")
            results.append(f"Failed to retrieve company info for stock code {stock_code}: {company_info_result['error']}")
            continue
        result = mongo.insert_document(collection_name, company_info_result)
        if result != None:
            print(f"Stock code {stock_code} added to watchlist in collection {collection_name}.")
            results.append(f"Stock code {stock_code} added to watchlist in collection {collection_name}.")
        else:
            print(f"Failed to add stock code {stock_code} to watchlist in collection {collection_name}.")
            results.append(f"Failed to add stock code {stock_code} to watchlist in collection {collection_name}.")
    return results

@tool
def delete_from_watchlist(collection_name, stock_code):
    """
    delete a stock code from the watchlist according to the specified collection and stock code in the MongoDB database stock_watchlist.
    """
    result = mongo.delete_by_stock(collection_name, stock_code)
    if "error" in result:
        print(f"Failed to delete company info for stock code {stock_code}: {result['error']}")
        return f"Failed to delete company info for stock code {stock_code}: {result['error']}"
    if result != None:
        print(f"Stock code {stock_code} deleted from watchlist in collection {collection_name}.")
        return f"Stock code {stock_code} deleted from watchlist in collection {collection_name}."
    else:
        print(f"Failed to delete stock code {stock_code} from watchlist in collection {collection_name}.")
        return f"Failed to delete stock code {stock_code} from watchlist in collection {collection_name}."

#### Calculation ####

@tool
def calcu_KD_w(code, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate K and D values for a stock with a week as a unit.
    """
    df, stock_type = stock_data(code)
    df = df.copy()
    try: 
        df = df.xs(f"{code}.{stock_type}", axis=1, level=1)
    
        # --- Ensure correct dtype ---
        for c in ["High", "Low", "Close"]:
            df[c] = df[c].astype(float)
    
        # --- Resample to weekly (Friday) ---
        df = df.resample("W-FRI").agg({
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna()
    
        df = df.reset_index()  # optional, for easier handling
    
        highs = df["High"]
        lows = df["Low"]
        closes = df["Close"]
    
    
        # --- Compute RSV ---
        low_min = lows.rolling(period).min()
        high_max = highs.rolling(period).max()
        rsv = (closes - low_min) / (high_max - low_min) * 100
        rsv = rsv.dropna()
    
        if rsv.empty:
            raise ValueError("RSV calculation failed: insufficient weekly data")
    
        # --- Compute K/D ---
        k_values = []
        d_values = []
        k = float(init_k)
        d = float(init_d)
    
        for r in rsv.values:
            k = (2/3) * k + (1/3) * r
            d = (2/3) * d + (1/3) * k
            k_values.append(k)
            d_values.append(d)
    
        result = {"k": k_values[-1], "d":d_values[-1]}
    
    except:
        result = "The data cannot be fetched"
        
    return str(result)


@tool
def calcu_KD_w_multiple(codes:list, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for multiple/many stocks.
    Extracts weekly High, Low, Close.
    Return the format {stock:{k:k_value, d:d_value}, stock_2...}
    """
    return calcu_KD_w_multiple_(codes)

@tool
def calcu_KD_w_series(
    code,
    period=9,
    init_k=50.0,
    init_d=50.0,
    near_threshold=1.0,   # how close (in K-D points) counts as "near cross"
    lookback=5,           # how many recent points to scan for cross/near-cross
):
    """
    Calculate consecutive weekly K/D values and detect golden/death cross signals.
    """
    
    df, stock_type = stock_data(code)
    df = df.copy()

    try: 
        df = df.xs(f"{code}.{stock_type}", axis=1, level=1)
    
        for c in ["High", "Low", "Close"]:
            df[c] = df[c].astype(float)
    
        # Weekly (Friday)
        df = df.resample("W-FRI").agg({
            "High": "max",
            "Low": "min",
            "Close": "last"
        }).dropna()
    
        if df.empty or len(df) < period:
            raise ValueError("Insufficient weekly data for KD calculation")
    
        highs = df["High"]
        lows = df["Low"]
        closes = df["Close"]
    
        # RSV
        low_min = lows.rolling(period).min()
        high_max = highs.rolling(period).max()
        denom = high_max - low_min
    
        rsv = (closes - low_min) / denom * 100
        rsv = rsv.replace([np.inf, -np.inf], np.nan).dropna()
    
        if rsv.empty:
            raise ValueError("RSV calculation failed: insufficient weekly data")
    
        # K/D series
        k = float(init_k)
        d = float(init_d)
        k_list = []
        d_list = []
        idx_list = []
    
        for idx, r in rsv.items():
            r = float(r)
            k = (2.0 / 3.0) * k + (1.0 / 3.0) * r
            d = (2.0 / 3.0) * d + (1.0 / 3.0) * k
            idx_list.append(idx)
            k_list.append(k)
            d_list.append(d)
    
        kd = pd.DataFrame(
            {"K": k_list, "D": d_list},
            index=pd.to_datetime(idx_list)
        )
    
        kd["diff"] = kd["K"] - kd["D"]
    
        # --- Cross detection ---
        kd["prev_diff"] = kd["diff"].shift(1)
        kd["golden_cross"] = (kd["prev_diff"] <= 0) & (kd["diff"] > 0)
        kd["death_cross"]  = (kd["prev_diff"] >= 0) & (kd["diff"] < 0)
    
        # --- Imminent cross detection ---
        kd["absdiff"] = kd["diff"].abs()
        kd["diff_change"] = kd["diff"].diff()
    
        kd["imminent_golden"] = (
            (kd["diff"] < 0) &
            (kd["absdiff"] <= near_threshold) &
            (kd["diff_change"] > 0)
        )
    
        kd["imminent_death"] = (
            (kd["diff"] > 0) &
            (kd["absdiff"] <= near_threshold) &
            (kd["diff_change"] < 0)
        )
    
        recent = kd.tail(max(lookback, 2)).copy()
    
        # Find last cross
        last_cross = None
        crosses = recent[(recent["golden_cross"]) | (recent["death_cross"])]
        if not crosses.empty:
            last_idx = crosses.index[-1]
            last_row = crosses.loc[last_idx]
            last_cross = {
                "date": last_idx.strftime("%Y-%m-%d"),
                "type": "golden_cross" if bool(last_row["golden_cross"]) else "death_cross",
                "k": float(last_row["K"]),
                "d": float(last_row["D"]),
            }
    
        latest = kd.iloc[-1]
        imminent = None
        if bool(latest["imminent_golden"]):
            imminent = "golden_cross_likely"
        elif bool(latest["imminent_death"]):
            imminent = "death_cross_likely"
    
        result = {
            "latest": {
                "k": float(latest["K"]),
                "d": float(latest["D"]),
                "diff": float(latest["diff"]),
            },
            "last_cross_recent": last_cross,
            "imminent": imminent,
            "recent_series": [
                {
                    "date": i.strftime("%Y-%m-%d"),
                    "k": float(row["K"]),
                    "d": float(row["D"]),
                    "diff": float(row["diff"]),
                    "golden_cross": bool(row["golden_cross"]),
                    "death_cross": bool(row["death_cross"]),
                }
                for i, row in recent.iterrows()
            ],
        }

    except:
        result = "Information not found"
    
    return str(result)

@tool
def stock_price_averages(stock_id):
    """
    Returns yearly and monthly average stock prices for the past 5 years.

    Args:
        stock_id: string, e.g., "00881"

    Returns:
        yearly_avg: DataFrame with yearly average prices (Open, High, Low, Close, Volume)
        monthly_avg: dict of DataFrames, each key is a year, value is monthly averages
    """
    ticker = f"{stock_id}.TW"

    # Fetch 5 years of daily data
    df = yf.download(ticker, period="5y", interval="1d")
    if df.empty or len(df) < 10:
          stock_type = "TWO"
          ticker = f"{stock_id}.{stock_type}"
          df = yf.download(ticker, period="90d", interval="1d")
          df.dropna(inplace=True)
    
          if len(df) < 2:
              return {"error": "No information found"}

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date ascending
    df = df.sort_index()

    # --- Yearly average ---
    yearly_avg = df.resample('YE').mean()

    # --- Monthly averages per year ---
    monthly_avg = {}
    for year in df.index.year.unique():
        year_df = df[df.index.year == year]
        monthly_avg[year] = year_df.resample('ME').mean()

    return yearly_avg, monthly_avg

@tool
def calcu_KD_w_watchlist(period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values of all the stocks in all the collections within the watchlist database. It calculates K and D values for each stock, and finally returns the results in a dictionary format.
    The output format is {stock_code: {k: k_value, d: d_value}, stock_code_2: {...}, ...}
    """
    print("Watch list search....")

    # Ensure correct dtype
    docs = mongo.fetch_all()
    result = {}
    for collection_name, documents in docs.items():
        stock_codes = []
        for document in documents:
            if "stock" not in document:
                print(f"Document {item} does not contain 'stock' field. Skipping.")
                continue
            if not isinstance(document["stock"], str):
                print(f"Document {item} has 'stock' field that is not a string. Skipping.")
                continue
            stock_codes.append(document["stock"])
            
        result[collection_name] = calcu_KD_w_multiple_(codes=stock_codes, period=period, init_k=init_k, init_d=init_d)

    return str(result)

@tool
def stock_per(code):
  """
  Fetch dividend and PER and PBR for the stock or stocks. Feed stock codes as string argument(s)
  Return the format {stock_code:{dividend: value, PER: value, PBR: value}, stock_code_2: {....} }
  """
  token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMS0yOCAwNTozMDoyNiIsInVzZXJfaWQiOiJ2aW5jZWppbTkxMTI2IiwiZW1haWwiOiJ2aW5jZWppbTkxMTI2QGdtYWlsLmNvbSIsImlwIjoiMTM3LjIyMC44MC4xMDYifQ.YtyRO5c9V9mI8s_SuwxFMZoNDT5q8R40JkxRlIcJdYQ"
  api = DataLoader()
  #api.login_by_token(api_token=token)
  result = {}

  today = datetime.today().strftime("%Y-%m-%d")
  end_date = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
  start_date = (datetime.today() - timedelta(days=8)).strftime("%Y-%m-%d")

  df = api.taiwan_stock_per_pbr(
      stock_id= code,
      start_date=start_date,
      end_date=end_date,
  )


  result = {"dividend_yield":float(df["dividend_yield"][0]), "PER":float(df["PER"][0]), "PBR":float(df["PBR"][0])} if df.shape[0] > 1 else "Not available"

  print (result)
  return result

#### Company information ####
@tool
def company_news(stock_id):
  """
  Fetch and get recent news about this company
  """
  news = []
  # Recent news (availability varies)
  ticker_symbol = f"{stock_id}.TW" # Example stock
  ticker = yf.Ticker(ticker_symbol)
  if len(ticker.news) == 0:
    # Recent news (availability varies)
    ticker_symbol = f"{stock_id}.TWO" # Example stock
    ticker = yf.Ticker(ticker_symbol)
    if len(ticker.news) == 0:
        return "No news found"
  
  for i in ticker.news:
    news.append(f"title: {i['content']['title']}, summary: {i['content']['summary']}, Date: {i['content']['pubDate']}")

  return str(news)

"""
The tool below is for stock health check
It compiles all the possible indicators
"""
def check_stock_health(stock_symbol: str) -> dict:
    """
    This tool provides comprehensive financial stock health check.
    Returns a dict with all metrics plus a Gemini AI summary.
    """
    print(f"\n{'='*60}")
    print(f"  Financial Health Report: {stock_symbol}")
    print(f"{'='*60}")

    report = {"symbol": stock_symbol}
    
    try:
        ticker = yf.Ticker(stock_symbol)
        info   = ticker.info

        # ── Company overview ─────────────────────────────────────────
        overview = {
            "company_name": info.get('longName', 'N/A'),
            "sector":       info.get('sector', 'N/A'),
            "industry":     info.get('industry', 'N/A'),
            "summary":      info.get('longBusinessSummary', 'N/A')[:500] + "...",
            "market_cap":   info.get('marketCap'),
            "pe_ratio":     info.get('trailingPE'),
            "pb_ratio":     info.get('priceToBook'),
        }
        report["overview"] = overview
        print(f"\nCompany   : {overview['company_name']}")
        print(f"Sector    : {overview['sector']}")
        print(f"Industry  : {overview['industry']}")
        print(f"Market Cap: {overview['market_cap']}")
        print(f"P/E Ratio : {overview['pe_ratio']}")
        print(f"P/B Ratio : {overview['pb_ratio']}")

        # ── Financial metrics ────────────────────────────────────────
        report["revenue_growth"]          = check_revenue_growth(ticker)
        report["eps_history"]             = check_eps_history(ticker)
        report["dividend_yield"]          = check_dividend_yield(ticker)
        report["roe_history"]             = check_roe_history(ticker)
        report["free_cash_flow"]          = check_free_cash_flow(ticker)
        report["operating_cf_ratio"]      = check_operating_cash_flow_ratio(ticker)
        report["ar_turnover_days"]        = check_ar_turnover_days(ticker)
        report["inventory_turnover_days"] = check_inventory_turnover_days(ticker)
        report["supervisor_holdings"]     = check_supervisor_holdings(ticker)

        # ── Technical – KD indicator ─────────────────────────────────
        # Extract stock_id from symbol (e.g. "2330.TW" → "2330")
        stock_id = stock_symbol.split(".")[0]
        k_val,  d_val  = calcu_KD(stock_id)
        kw_val, dw_val = calcu_KD_weekly(stock_id)
        report["kd_indicator"] = {
            "daily":  {"K": k_val,  "D": d_val},
            "weekly": {"K": kw_val, "D": dw_val},
        }
        print(f"\n--- KD Indicator (Daily)  ---")
        print(f"  K: {k_val}  |  D: {d_val}")
        print(f"--- KD Indicator (Weekly) ---")
        print(f"  K: {kw_val}  |  D: {dw_val}")

    except Exception as e:
        report["error"] = str(e)
        print(f"\n  Error: {e}")

    print(f"\n{'='*60}")
    print(f"  End of Report: {stock_symbol}")
    print(f"{'='*60}\n")

    return report


##################Tool Ends###################


# Create agent
agent = create_agent(
    model=model,
    tools=[fetch_all_stocks, add_collection, list_collections, calcu_KD_w, calcu_KD_w_multiple, stock_price_averages, calcu_KD_w_watchlist, calcu_KD_w_series, collection_information, add_to_watchlist, add_m_to_watchlist, delete_from_watchlist, check_database_connection, company_news, stock_per, check_stock_health],
    system_prompt="You are a very helpful assistant"
)

def get_response_from_agent(input):

  response = agent.invoke({
      "messages": [
          {"role": "user", "content": input}
      ]
  })

  messages = response["messages"]

  # iterate backwards to find last AIMessage
  for msg in reversed(messages):
      if isinstance(msg, AIMessage):
          return msg.content
  return None

