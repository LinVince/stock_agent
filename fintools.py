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

# ══════════════════════════════════════════════════════════════
#  Environment Setup
# ══════════════════════════════════════════════════════════════

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "pr-tart-sweatsuit-95"

LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

deepseek_api_key = "sk-daadb7c9968f4c07b802743b631164d1"

model = init_chat_model(
    "deepseek-chat",
    model_provider="deepseek",
    api_key=deepseek_api_key
)


# ══════════════════════════════════════════════════════════════
#  UNIFIED TICKER RESOLUTION  (all functions use these two)
#  US TW TWO 
# ══════════════════════════════════════════════════════════════

def resolve_ticker(stock_id: str):
    stock_id = str(stock_id).strip()

    # US stock — no suffix needed
    ticker = yf.Ticker(stock_id)
    try:
        info = ticker.info
        if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
            return ticker, "US"
    except Exception:
        pass

    # Taiwan stocks
    for suffix in ["TW", "TWO"]:
        ticker = yf.Ticker(f"{stock_id}.{suffix}")
        try:
            info = ticker.info
            if info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose"):
                return ticker, suffix
        except Exception:
            continue

    return None, None


def resolve_df(stock_id: str, period="90d", interval="1d"):
    stock_id = str(stock_id).strip()

    # US stock
    df = yf.download(stock_id, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)
    if not df.empty and len(df) >= 2:
        return df.sort_index(ascending=True), "US"

    # Taiwan stocks
    for suffix in ["TW", "TWO"]:
        df = yf.download(f"{stock_id}.{suffix}", period=period, interval=interval, progress=False)
        df.dropna(inplace=True)
        if not df.empty and len(df) >= 2:
            return df.sort_index(ascending=True), suffix

    return pd.DataFrame(), None

# ══════════════════════════════════════════════════════════════
#  SHARED INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

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
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return str(val)


def _get_weekly_df(stock_id: str):
    """
    Return a weekly-resampled (W-FRI) OHLC DataFrame for a plain stock code.
    Flattens multi-level yfinance columns automatically.
    Returns (DataFrame, suffix) or (empty DataFrame, None).
    """
    df, suffix = resolve_df(stock_id, period="90d", interval="1d")
    if df.empty:
        return pd.DataFrame(), None

    # Flatten multi-level columns produced by yf.download
    if isinstance(df.columns, pd.MultiIndex):
        key = stock_id if suffix == "US" else f"{stock_id}.{suffix}"
        df = df.xs(key, axis=1, level=1)
        #df = df.xs(f"{stock_id}.{suffix}", axis=1, level=1)

    for c in ["High", "Low", "Close"]:
        df[c] = df[c].astype(float)

    df = df.resample("W-FRI").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    return df.reset_index(), suffix


def _calcu_KD_from_df(df: pd.DataFrame, period=9, init_k=50.0, init_d=50.0):
    """Core KD calculation — works on any daily or weekly DataFrame."""
    if df is None or (isinstance(df, dict)) or df.empty or len(df) < period:
        return None, None
    df = df.copy()
    for c in ["High", "Low", "Close"]:
        df[c] = df[c].astype(float)
    rsv = ((df["Close"] - df["Low"].rolling(period).min()) /
           (df["High"].rolling(period).max() - df["Low"].rolling(period).min()) * 100).dropna()
    if rsv.empty:
        return None, None
    k, d = float(init_k), float(init_d)
    for r in rsv.values:
        k = (2/3) * k + (1/3) * float(r)
        d = (2/3) * d + (1/3) * k
    return round(k, 2), round(d, 2)




def get_hourly_prices(stock_id: str) -> dict:
    """
    Internal helper — fetches hourly OHLCV for the latest trading day,
    plus closing prices for the previous day, week, month, and year
    for percentage change comparisons.
    """
    # ── Hourly data for today ──────────────────────────────────
    df, suffix = resolve_df(stock_id, period="2d", interval="1h")
    if df.empty:
        return {"error": f"No hourly data found for {stock_id}"}

    if isinstance(df.columns, pd.MultiIndex):
        key = stock_id if suffix == "US" else f"{stock_id}.{suffix}"
        df = df.xs(key, axis=1, level=1)

    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip tz → naive
    latest_date = df.index.normalize().max()
    today_df = df[df.index.normalize() == latest_date].copy()

    if today_df.empty:
        return {"error": "No data found for the latest trading day"}

    current_price = round(float(today_df["Close"].iloc[-1]), 2)

    # Intraday trend: compare first open to last close
    day_open  = round(float(today_df["Open"].iloc[0]),  2)
    day_high  = round(float(today_df["High"].max()),    2)
    day_low   = round(float(today_df["Low"].min()),     2)
    intraday_chg_pct = round((current_price - day_open) / day_open * 100, 2)
    if intraday_chg_pct > 0:
        trend = "uptrend"
    elif intraday_chg_pct < 0:
        trend = "downtrend"
    else:
        trend = "flat"

    # ── Daily data for historical comparisons ─────────────────
    hist_df, _ = resolve_df(stock_id, period="13mo", interval="1d")
    if isinstance(hist_df.columns, pd.MultiIndex):
        key = stock_id if suffix == "US" else f"{stock_id}.{suffix}"
        hist_df = hist_df.xs(key, axis=1, level=1)

    hist_df.index = pd.to_datetime(hist_df.index).tz_localize(None)  # strip tz → naive
    hist_df = hist_df[hist_df.index.normalize() < latest_date].copy()

    def _pct(ref_price):
        if ref_price is None or ref_price == 0:
            return None
        return round((current_price - ref_price) / ref_price * 100, 2)

    def _closest_close(target_date):
        """Return the closing price of the nearest trading day on or before target_date."""
        subset = hist_df[hist_df.index.normalize() <= pd.Timestamp(target_date)]
        if subset.empty:
            return None
        return round(float(subset["Close"].iloc[-1]), 2)

    prev_day_close   = _closest_close(latest_date - timedelta(days=1))
    prev_week_close  = _closest_close(latest_date - timedelta(weeks=1))
    prev_month_close = _closest_close(latest_date - timedelta(days=30))
    prev_year_close  = _closest_close(latest_date - timedelta(days=365))

    comparisons = {
        "vs_prev_day":   {"ref_price": prev_day_close,   "chg_pct": _pct(prev_day_close)},
        "vs_prev_week":  {"ref_price": prev_week_close,  "chg_pct": _pct(prev_week_close)},
        "vs_prev_month": {"ref_price": prev_month_close, "chg_pct": _pct(prev_month_close)},
        "vs_prev_year":  {"ref_price": prev_year_close,  "chg_pct": _pct(prev_year_close)},
    }

    candles = [
        {
            "time":   row.name.strftime("%H:%M"),
            "open":   round(float(row["Open"]),  2),
            "high":   round(float(row["High"]),  2),
            "low":    round(float(row["Low"]),   2),
            "close":  round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        for _, row in today_df.iterrows()
    ]

    return {
        "stock":         stock_id,
        "market":        suffix,
        "date":          latest_date.strftime("%Y-%m-%d"),
        "current_price": current_price,
        "day_open":      day_open,
        "day_high":      day_high,
        "day_low":       day_low,
        "intraday_trend": trend,
        "intraday_chg_pct": intraday_chg_pct,
        "comparisons":   comparisons,
        "candles":       candles,
    }

def company_info(stock_id: str) -> dict:
    """Return company metadata for a plain stock code. Auto-detects TW vs TWO."""
    ticker, suffix = resolve_ticker(stock_id)
    if ticker is None:
        return {"error": f"Could not resolve ticker for {stock_id}"}
    info = ticker.info
    return {
        "stock":    stock_id,
        "summary":  info.get("longBusinessSummary", ""),
        "sector":   info.get("sector", f"Couldn't find the sector of {stock_id}"),
        "industry": info.get("industry", f"Couldn't find the industry of {stock_id}"),
    }


def calcu_KD(stock_id: str, period=9, init_k=50.0, init_d=50.0):
    """Daily KD for a plain stock code. Auto-detects TW vs TWO."""
    df, _ = resolve_df(stock_id, period="90d", interval="1d")
    return _calcu_KD_from_df(df, period, init_k, init_d)


def calcu_KD_weekly(stock_id: str, period=9, init_k=50.0, init_d=50.0):
    """Weekly KD for a plain stock code. Auto-detects TW vs TWO."""
    df, _ = resolve_df(stock_id, period="2y", interval="1wk")
    return _calcu_KD_from_df(df, period, init_k, init_d)


def calcu_KD_w_multiple_(codes: list, period=9, init_k=50.0, init_d=50.0):
    """Weekly KD for a list of plain stock codes. Returns {code: {k, d}, ...}"""
    result = {}
    for code in codes:
        try:
            df, _ = _get_weekly_df(code)
            if df.empty or len(df) < period:
                raise ValueError("Insufficient data")
            rsv = ((df["Close"] - df["Low"].rolling(period).min()) /
                   (df["High"].rolling(period).max() - df["Low"].rolling(period).min()) * 100).dropna()
            if rsv.empty:
                raise ValueError("RSV failed")
            k, d = float(init_k), float(init_d)
            k_vals, d_vals = [], []
            for r in rsv.values:
                k = (2/3) * k + (1/3) * float(r)
                d = (2/3) * d + (1/3) * k
                k_vals.append(k); d_vals.append(d)
            result[code] = {"k": round(k_vals[-1], 2), "d": round(d_vals[-1], 2)}
        except Exception:
            result[code] = "No information was fetched"
    return str(result)


# ══════════════════════════════════════════════════════════════
#  CHECK FUNCTIONS (only for check_stock_health)
# ══════════════════════════════════════════════════════════════

def check_revenue_growth(ticker_obj) -> dict:
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
    result["growth_pct"] = {str(d.year): round(v, 2) for d, v in growth.iloc[::-1].dropna().head(5).items()}
    print("\n--- Revenue Growth ---")
    for yr, pct in result["growth_pct"].items():
        print(f"  {yr}: {pct:.2f}%")
    return result


def check_eps_history(ticker_obj) -> dict:
    result = {"trailing_eps": None, "annual_eps": {}, "quarterly_eps": {}, "error": None}
    result["trailing_eps"] = ticker_obj.info.get('trailingEps')
    annual = _get_financial_attribute(ticker_obj, 'financials')
    if not annual.empty and 'Basic EPS' in annual.index:
        result["annual_eps"] = {str(d.year): round(v, 2)
                                for d, v in annual.loc["Basic EPS"].dropna().iloc[::-1].head(5).items()}
    qf = _get_financial_attribute(ticker_obj, 'quarterly_financials')
    if not qf.empty and 'Net Income' in qf.index and 'Diluted Average Shares' in qf.index:
        ni, shares = qf.loc["Net Income"].dropna(), qf.loc["Diluted Average Shares"].dropna()
        common = ni.index.intersection(shares.index)
        if not common.empty:
            q_eps = (ni.reindex(common) / shares.reindex(common)).dropna().iloc[::-1]
            result["quarterly_eps"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in q_eps.head(5).items()}
    print(f"\n--- EPS History ---\n  Trailing EPS: {result['trailing_eps']}")
    print(f"  Annual EPS: {result['annual_eps']}")
    print(f"  Quarterly EPS: {result['quarterly_eps']}")
    return result


def check_dividend_yield(ticker_obj) -> dict:
    result = {"dividend_yield_pct": None, "error": None}
    dy = ticker_obj.info.get('dividendYield')
    if dy is not None:
        result["dividend_yield_pct"] = round(dy if dy > 1.0 else dy * 100, 2)
    print(f"\n--- Dividend Yield ---\n  {_fmt(result['dividend_yield_pct'], suffix='%') if result['dividend_yield_pct'] else 'Not available'}")
    return result


def check_roe_history(ticker_obj) -> dict:
    result = {"annual_roe_pct": {}, "quarterly_ttm_roe_pct": {}, "error": None}
    af = _get_financial_attribute(ticker_obj, 'financials')
    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    if not af.empty and not ab.empty and 'Net Income' in af.index and 'Stockholders Equity' in ab.index:
        ni, eq = af.loc["Net Income"].dropna(), ab.loc["Stockholders Equity"].dropna()
        common = ni.index.intersection(eq.index)
        if not common.empty:
            roe = (ni.reindex(common) / eq.reindex(common) * 100).dropna().iloc[::-1]
            result["annual_roe_pct"] = {str(d.year): round(v, 2) for d, v in roe.head(5).items()}
    qf = _get_financial_attribute(ticker_obj, 'quarterly_financials')
    qb = _get_financial_attribute(ticker_obj, 'quarterly_balance_sheet')
    if not qf.empty and not qb.empty and 'Net Income' in qf.index and 'Stockholders Equity' in qb.index:
        ni_q = qf.loc["Net Income"].dropna().sort_index()
        eq_q = qb.loc["Stockholders Equity"].dropna().sort_index()
        if len(ni_q) >= 4:
            ttm = ni_q.rolling(4, min_periods=4).sum()
            ttm_roe = {d.strftime('%Y-%m-%d'): round(v / eq_q.loc[d] * 100, 2)
                       for d, v in ttm.items() if d in eq_q.index and eq_q.loc[d] != 0}
            result["quarterly_ttm_roe_pct"] = dict(list(sorted(ttm_roe.items(), reverse=True))[:5])
    print(f"\n--- ROE ---\n  Annual: {result['annual_roe_pct']}")
    print(f"  Quarterly TTM: {result['quarterly_ttm_roe_pct']}")
    return result


def check_free_cash_flow(ticker_obj) -> dict:
    result = {"annual_fcf": {}, "quarterly_fcf": {}, "error": None}
    CAPEX_LABELS = ["Capital Expenditures", "Capital Expenditure"]

    def _compute_fcf(cf_df):
        if cf_df.empty or 'Operating Cash Flow' not in cf_df.index:
            return None
        label = next((l for l in CAPEX_LABELS if l in cf_df.index), None)
        if not label:
            return None
        ocf, capex = cf_df.loc["Operating Cash Flow"].dropna(), cf_df.loc[label].dropna()
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
    print(f"\n--- Free Cash Flow ---\n  Annual: {result['annual_fcf']}")
    print(f"  Quarterly: {result['quarterly_fcf']}")
    return result


def check_operating_cash_flow_ratio(ticker_obj) -> dict:
    result = {"annual_ocf_ratio_pct": {}, "quarterly_ocf_ratio_pct": {}, "error": None}

    def _compute(fin_df, cf_df):
        if fin_df.empty or cf_df.empty or 'Total Revenue' not in fin_df.index or 'Operating Cash Flow' not in cf_df.index:
            return {}
        rev, ocf = fin_df.loc["Total Revenue"].dropna(), cf_df.loc["Operating Cash Flow"].dropna()
        common = rev.index.intersection(ocf.index)
        if common.empty:
            return {}
        return (ocf.reindex(common) / rev.reindex(common) * 100).replace([float('inf'), -float('inf')], pd.NA).dropna().iloc[::-1]

    ar = _compute(_get_financial_attribute(ticker_obj, 'financials'), _get_financial_attribute(ticker_obj, 'cashflow'))
    if not isinstance(ar, dict):
        result["annual_ocf_ratio_pct"] = {str(d.year): round(v, 2) for d, v in ar.head(5).items()}
    qr = _compute(_get_financial_attribute(ticker_obj, 'quarterly_financials'), _get_financial_attribute(ticker_obj, 'quarterly_cashflow'))
    if not isinstance(qr, dict):
        result["quarterly_ocf_ratio_pct"] = {d.strftime('%Y-%m-%d'): round(v, 2) for d, v in qr.head(5).items()}
    print(f"\n--- OCF Ratio ---\n  Annual: {result['annual_ocf_ratio_pct']}")
    print(f"  Quarterly: {result['quarterly_ocf_ratio_pct']}")
    return result


def check_ar_turnover_days(ticker_obj) -> dict:
    result = {"annual_dso_days": {}, "error": None}
    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    af = _get_financial_attribute(ticker_obj, 'financials')
    if ab.empty or af.empty or 'Accounts Receivable' not in ab.index or 'Total Revenue' not in af.index:
        result["error"] = "Data not available."
        return result
    ar = ab.loc["Accounts Receivable"].dropna().sort_index()
    rev = af.loc["Total Revenue"].dropna().sort_index()
    dso = {}
    for i in range(1, len(ar)):
        cur_d, prev_d = ar.index[i], ar.index[i-1]
        if cur_d in rev.index:
            avg_ar = (ar.loc[prev_d] + ar.loc[cur_d]) / 2
            if rev.loc[cur_d] > 0 and avg_ar > 0:
                dso[str(cur_d.year)] = round(365 / (rev.loc[cur_d] / avg_ar), 2)
    result["annual_dso_days"] = dict(list(sorted(dso.items(), reverse=True))[:5])
    print(f"\n--- DSO ---\n  {result['annual_dso_days']}")
    return result


def check_inventory_turnover_days(ticker_obj) -> dict:
    result = {"annual_dio_days": {}, "error": None}
    ab = _get_financial_attribute(ticker_obj, 'balance_sheet')
    af = _get_financial_attribute(ticker_obj, 'financials')
    if ab.empty or af.empty or 'Inventory' not in ab.index or 'Cost Of Revenue' not in af.index:
        result["error"] = "Data not available."
        return result
    inv = ab.loc["Inventory"].dropna().sort_index()
    cor = af.loc["Cost Of Revenue"].dropna().sort_index()
    dio = {}
    for i in range(1, len(inv)):
        cur_d, prev_d = inv.index[i], inv.index[i-1]
        if cur_d in cor.index:
            avg_inv = (inv.loc[prev_d] + inv.loc[cur_d]) / 2
            if cor.loc[cur_d] > 0 and avg_inv > 0:
                dio[str(cur_d.year)] = round(365 / (cor.loc[cur_d] / avg_inv), 2)
    result["annual_dio_days"] = dict(list(sorted(dio.items(), reverse=True))[:5])
    print(f"\n--- DIO ---\n  {result['annual_dio_days']}")
    return result


def check_supervisor_holdings(ticker_obj) -> dict:
    print("\n--- Supervisor Stock Holdings ---")
    print("  Not directly available via yfinance.")
    return {"note": "Senior supervisor stock holdings require specialized regulatory filings or alternative data providers."}


# ══════════════════════════════════════════════════════════════
#  TOOLS
# ══════════════════════════════════════════════════════════════

# ═════════════════════════════════
# Database Tools
# ═════════════════════════════════

@tool
def check_database_connection():
    """Checks the MongoDB database connection."""
    return mongo.check_db_connection()

@tool
def fetch_all_stocks():
    """Fetch all stocks stored in the watchlist database."""
    return str(mongo.fetch_all())

@tool
def add_collection(collection_name):
    """Create a new collection if it does not already exist."""
    return mongo.add_collection(collection_name)

@tool
def list_collections():
    """List all collections in the database."""
    return mongo.list_collections()

@tool
def collection_information(collection_name):
    """Returns the stocks and their info stored in the given collection."""
    return str(mongo.find_documents(collection_name))

@tool
def add_to_watchlist(stock_code, collection_name):
    """
    Add a stock to the watchlist. Accepts a plain stock code e.g. '2330'.
    Auto-detects whether it is a TW or TWO listed stock.
    """
    info = company_info(stock_code)
    if "error" in info:
        return f"Failed to retrieve info for {stock_code}: {info['error']}"
    result = mongo.insert_document(collection_name, info)
    if result is not None:
        return f"Stock {stock_code} added to {collection_name}."
    return f"Failed to add {stock_code} to {collection_name}."

@tool
def add_m_to_watchlist(stock_codes, collection_name):
    """
    Add multiple stocks to the watchlist. Accepts plain stock codes e.g. ['2330', '2454'].
    Auto-detects TW vs TWO for each stock.
    """
    results = []
    for code in stock_codes:
        info = company_info(code)
        if "error" in info:
            results.append(f"Failed for {code}: {info['error']}")
            continue
        result = mongo.insert_document(collection_name, info)
        results.append(f"{code} added to {collection_name}." if result is not None else f"Failed to add {code}.")
    return results

@tool
def delete_from_watchlist(collection_name, stock_code):
    """Delete a stock from the watchlist by plain stock code e.g. '2330'."""
    result = mongo.delete_by_stock(collection_name, stock_code)
    if "error" in result:
        return f"Failed to delete {stock_code}: {result['error']}"
    return f"Stock {stock_code} deleted from {collection_name}." if result else f"Failed to delete {stock_code}."


# ═════════════════════════════════
# Calculation Tools
# ═════════════════════════════════


@tool
def stock_hourly_prices(stock_id: str) -> str:
    """
    Fetch hourly OHLCV prices for the latest trading day of a stock,
    intraday trend (uptrend / downtrend / flat), and percentage change
    vs the previous day, week, month, and year.
    Useful for checking whether a stock like TSMC (2330) is up or down
    before deciding to buy a correlated TW stock.
    Accepts a US ticker e.g. 'AAPL' or a TW stock code e.g. '2330'.
    Auto-detects market (US / TW / TWO).
    """
    result = get_hourly_prices(stock_id)
    if "error" in result:
        return result["error"]

    c = result["comparisons"]

    def _fmt_chg(label, entry):
        ref = entry["ref_price"]
        pct = entry["chg_pct"]
        if pct is None:
            return f"  {label}: N/A"
        sign = "+" if pct >= 0 else ""
        return f"  {label}: ref {ref}  →  {sign}{pct}%"

    lines = [
        f"Stock : {result['stock']} ({result['market']})  |  Date: {result['date']}",
        f"Price : {result['current_price']}  |  Open: {result['day_open']}  "
        f"High: {result['day_high']}  Low: {result['day_low']}",
        f"Trend : {result['intraday_trend'].upper()}  ({'+' if result['intraday_chg_pct'] >= 0 else ''}"
        f"{result['intraday_chg_pct']}% from open)",
        "",
        "── Price Comparisons ──────────────────────────",
        _fmt_chg("vs Prev Day  ", c["vs_prev_day"]),
        _fmt_chg("vs Prev Week ", c["vs_prev_week"]),
        _fmt_chg("vs Prev Month", c["vs_prev_month"]),
        _fmt_chg("vs Prev Year ", c["vs_prev_year"]),
        "",
        "── Hourly Candles ─────────────────────────────",
    ]

    for candle in result["candles"]:
        lines.append(
            f"  {candle['time']}  O:{candle['open']}  H:{candle['high']}  "
            f"L:{candle['low']}  C:{candle['close']}  Vol:{candle['volume']:,}"
        )

    output = "\n".join(lines)
    print(output)
    return output
    
@tool
def calcu_KD_w(code, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for a stock.
    Accepts a plain stock code e.g. '2330'. Auto-detects TW vs TWO.
    """
    try:
        df, _ = _get_weekly_df(code)
        if df.empty or len(df) < period:
            return "The data cannot be fetched"
        rsv = ((df["Close"] - df["Low"].rolling(period).min()) /
               (df["High"].rolling(period).max() - df["Low"].rolling(period).min()) * 100).dropna()
        if rsv.empty:
            return "RSV calculation failed: insufficient weekly data"
        k, d = float(init_k), float(init_d)
        k_vals, d_vals = [], []
        for r in rsv.values:
            k = (2/3) * k + (1/3) * float(r)
            d = (2/3) * d + (1/3) * k
            k_vals.append(k); d_vals.append(d)
        return str({"k": round(k_vals[-1], 2), "d": round(d_vals[-1], 2)})
    except Exception:
        return "The data cannot be fetched"

@tool
def calcu_KD_w_multiple(codes: list, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for multiple stocks.
    Accepts plain stock codes e.g. ['2330', '2454']. Auto-detects TW vs TWO per stock.
    """
    return calcu_KD_w_multiple_(codes, period, init_k, init_d)

@tool
def calcu_KD_w_series(code, period=9, init_k=50.0, init_d=50.0, near_threshold=1.0, lookback=5):
    """
    Calculate consecutive weekly K/D values and detect golden/death cross signals.
    Accepts a plain stock code e.g. '2330'. Auto-detects TW vs TWO.
    """
    try:
        df, _ = _get_weekly_df(code)
        if df.empty or len(df) < period:
            return "Insufficient weekly data for KD calculation"

        for c in ["High", "Low", "Close"]:
            df[c] = df[c].astype(float)

        rsv = ((df["Close"] - df["Low"].rolling(period).min()) /
               (df["High"].rolling(period).max() - df["Low"].rolling(period).min()) * 100
               ).replace([np.inf, -np.inf], np.nan).dropna()

        if rsv.empty:
            return "RSV calculation failed: insufficient weekly data"

        k, d = float(init_k), float(init_d)
        k_list, d_list, idx_list = [], [], []
        for idx, r in rsv.items():
            k = (2/3) * k + (1/3) * float(r)
            d = (2/3) * d + (1/3) * k
            idx_list.append(idx); k_list.append(k); d_list.append(d)

        kd = pd.DataFrame({"K": k_list, "D": d_list}, index=pd.to_datetime(idx_list))
        kd["diff"] = kd["K"] - kd["D"]
        kd["prev_diff"] = kd["diff"].shift(1)
        kd["golden_cross"] = (kd["prev_diff"] <= 0) & (kd["diff"] > 0)
        kd["death_cross"]  = (kd["prev_diff"] >= 0) & (kd["diff"] < 0)
        kd["absdiff"] = kd["diff"].abs()
        kd["diff_change"] = kd["diff"].diff()
        kd["imminent_golden"] = (kd["diff"] < 0) & (kd["absdiff"] <= near_threshold) & (kd["diff_change"] > 0)
        kd["imminent_death"]  = (kd["diff"] > 0) & (kd["absdiff"] <= near_threshold) & (kd["diff_change"] < 0)

        recent = kd.tail(max(lookback, 2)).copy()
        last_cross = None
        crosses = recent[recent["golden_cross"] | recent["death_cross"]]
        if not crosses.empty:
            r = crosses.iloc[-1]
            last_cross = {"date": crosses.index[-1].strftime("%Y-%m-%d"),
                          "type": "golden_cross" if bool(r["golden_cross"]) else "death_cross",
                          "k": float(r["K"]), "d": float(r["D"])}

        latest = kd.iloc[-1]
        imminent = ("golden_cross_likely" if bool(latest["imminent_golden"])
                    else "death_cross_likely" if bool(latest["imminent_death"]) else None)

        return str({
            "latest": {"k": float(latest["K"]), "d": float(latest["D"]), "diff": float(latest["diff"])},
            "last_cross_recent": last_cross,
            "imminent": imminent,
            "recent_series": [
                {"date": i.strftime("%Y-%m-%d"), "k": float(row["K"]), "d": float(row["D"]),
                 "diff": float(row["diff"]), "golden_cross": bool(row["golden_cross"]),
                 "death_cross": bool(row["death_cross"])}
                for i, row in recent.iterrows()
            ],
        })
    except Exception:
        return "Information not found"

@tool
def stock_price_averages(stock_id):
    """
    Returns yearly and monthly average stock prices for the past 5 years.
    Accepts a plain stock code e.g. '2330'. Auto-detects TW vs TWO.
    """
    df, _ = resolve_df(stock_id, period="5y", interval="1d")
    if df.empty:
        return {"error": "No information found"}
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    yearly_avg = df.resample('YE').mean()
    monthly_avg = {year: df[df.index.year == year].resample('ME').mean()
                   for year in df.index.year.unique()}
    return yearly_avg, monthly_avg

@tool
def calcu_KD_w_watchlist(period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for all stocks across all watchlist collections.
    Returns {collection_name: {stock_code: {k, d}, ...}, ...}
    """
    print("Watch list search....")
    docs = mongo.fetch_all()
    result = {}
    for collection_name, documents in docs.items():
        codes = [d["stock"] for d in documents if isinstance(d.get("stock"), str)]
        result[collection_name] = calcu_KD_w_multiple_(codes=codes, period=period, init_k=init_k, init_d=init_d)
    return str(result)

@tool
def stock_per(code):
    """
    Fetch dividend yield, PER and PBR for a stock.
    Accepts a plain TW stock code e.g. '2330' or a US ticker e.g. 'AAPL'.
    Auto-detects market.
    """
    _, suffix = resolve_ticker(code)

    if suffix == "US":
        ticker = yf.Ticker(code)
        info = ticker.info
        dy  = info.get("dividendYield")
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")

        if dy is None and per is None and pbr is None:
            return "Not available"

        result = {
            "dividend_yield": round(dy * 100, 2) if dy is not None else None,
            "PER":            round(per, 2)       if per is not None else None,
            "PBR":            round(pbr, 2)       if pbr is not None else None,
        }

    else:
        # Taiwan stocks — use FinMind
        api        = DataLoader()
        end_date   = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=8)).strftime("%Y-%m-%d")
        df = api.taiwan_stock_per_pbr(stock_id=code, start_date=start_date, end_date=end_date)

        result = (
            {
                "dividend_yield": float(df["dividend_yield"][0]),
                "PER":            float(df["PER"][0]),
                "PBR":            float(df["PBR"][0]),
            }
            if df.shape[0] > 1 else "Not available"
        )

    print(result)
    return result

# ═════════════════════════════════
# News
# ═════════════════════════════════

@tool
def company_news(stock_id):
    """
    Fetch recent news for a company.
    Accepts a plain stock code e.g. '2330'. Auto-detects TW vs TWO.
    """
    ticker, _ = resolve_ticker(stock_id)
    if ticker is None:
        return "No news found"
    news = [f"title: {i['content']['title']}, summary: {i['content']['summary']}, Date: {i['content']['pubDate']}"
            for i in ticker.news]
    return str(news) if news else "No news found"

# ═════════════════════════════════
# Health Check
# ═════════════════════════════════


@tool
def check_stock_health(stock_id: str) -> str:
    """
    Comprehensive financial health check for a stock.
    Accepts a plain stock code e.g. '2330'. Auto-detects TW vs TWO.
    """
    ticker, suffix = resolve_ticker(stock_id)
    if ticker is None:
        return f"Could not resolve ticker for stock {stock_id}"

    stock_symbol = f"{stock_id}.{suffix}"
    print(f"\n{'='*60}\n  Financial Health Report: {stock_symbol}\n{'='*60}")

    report = {"symbol": stock_symbol}
    try:
        info = ticker.info
        overview = {
            "company_name": info.get('longName', 'N/A'),
            "sector":       info.get('sector', 'N/A'),
            "industry":     info.get('industry', 'N/A'),
            "summary":      (info.get('longBusinessSummary') or 'N/A')[:500] + "...",
            "market_cap":   info.get('marketCap'),
            "pe_ratio":     info.get('trailingPE'),
            "pb_ratio":     info.get('priceToBook'),
        }
        report["overview"] = overview
        print(f"\nCompany: {overview['company_name']}  |  Sector: {overview['sector']}")
        print(f"Market Cap: {overview['market_cap']}  |  P/E: {overview['pe_ratio']}  |  P/B: {overview['pb_ratio']}")

        report["revenue_growth"]          = check_revenue_growth(ticker)
        report["eps_history"]             = check_eps_history(ticker)
        report["dividend_yield"]          = check_dividend_yield(ticker)
        report["roe_history"]             = check_roe_history(ticker)
        report["free_cash_flow"]          = check_free_cash_flow(ticker)
        report["operating_cf_ratio"]      = check_operating_cash_flow_ratio(ticker)
        report["ar_turnover_days"]        = check_ar_turnover_days(ticker)
        report["inventory_turnover_days"] = check_inventory_turnover_days(ticker)
        report["supervisor_holdings"]     = check_supervisor_holdings(ticker)

        k_val,  d_val  = calcu_KD(stock_id)
        kw_val, dw_val = calcu_KD_weekly(stock_id)
        report["kd_indicator"] = {
            "daily":  {"K": k_val,  "D": d_val},
            "weekly": {"K": kw_val, "D": dw_val},
        }
        print(f"\n--- KD Daily  K:{k_val}  D:{d_val}")
        print(f"--- KD Weekly K:{kw_val}  D:{dw_val}")

    except Exception as e:
        report["error"] = str(e)
        print(f"\n  Error: {e}")

    print(f"\n{'='*60}\n  End of Report: {stock_symbol}\n{'='*60}\n")
    return str(report)


# ══════════════════════════════════════════════════════════════
#  AGENT
# ══════════════════════════════════════════════════════════════

agent = create_agent(
    model=model,
    tools=[
        fetch_all_stocks, add_collection, list_collections,
        calcu_KD_w, calcu_KD_w_multiple, stock_price_averages,
        calcu_KD_w_watchlist, calcu_KD_w_series, collection_information,
        add_to_watchlist, add_m_to_watchlist, delete_from_watchlist,
        check_database_connection, company_news, stock_per, check_stock_health, stock_hourly_prices
    ],
    system_prompt="You are a stock analyst. Return plaintext and do not apply any style like bold texts to the return values."
)

def get_response_from_agent(input):
    response = agent.invoke({"messages": [{"role": "user", "content": input}]})
    for msg in reversed(response["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content
    return None
