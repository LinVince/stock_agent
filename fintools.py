from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
import os
import numpy as np
import pandas as pd
from FinMind.data import DataLoader
from datetime import datetime, timedelta
import mongodb_connection as mongo

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


###########Funtions for tools#####################

def stock_data(stock_id):
  """
  This tool return a dataframe of a stock for the past 90 days
  """
  ticker = f"{stock_id}.TW"
  df = yf.download(ticker, period="90d", interval="1d")
  df.dropna(inplace=True)
  if df.empty or len(df) < 10:
      return {"error": "資料不足或代號錯誤"}
  print(len(df))

  if len(df) < 2:
      return {"error": "計算後資料不足"}

  df = df.sort_values(by="Date", ascending=True)
  return df

def company_info(stock_id):
  ticker_symbol = f"{stock_id}.TW" # Example stock
  ticker = yf.Ticker(ticker_symbol)

   # Get the longBusinessSummary
  long_business_summary = ticker.info.get("longBusinessSummary") if ticker.info.get("longBusinessSummary") else f"Could not retrieve longBusinessSummary for {ticker_symbol}."
  sector = ticker.info.get("sector") if ticker.info.get("sector") else f"Couldn't find the sector of {ticker_symbol}"
  industry = ticker.info.get("industry") if ticker.info.get("industry") else f"Couldn't find industry of {ticker_symbol}"


  return {"stock":stock_id, "summary":long_business_summary, "sector": sector, "industry": industry}

def calcu_KD_w_multiple_(codes:list, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values for multiple/many stocks.
    Extracts weekly High, Low, Close.
    Return the format {stock:{k:k_value, d:d_value}, stock_2...}
    """
    result = {}
    # Ensure correct dtype
    for code in codes:
            # --- Load data and ensure correct dtype ---
      df = stock_data(code)
      # --- Flatten columns (optional, easier to work with) ---
      # Select only the first (or relevant) ticker
      df = df.xs(f"{code}.TW", axis=1, level=1)

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

    return str(result)

###########Put Tools Here#####################
import yfinance as yf
import pandas as pd
import requests

@tool
def calcu_KD_d(code, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate K and D values for a stock with a day as a unit.
    """
    # Ensure correct dtype
    df = stock_data(code)
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


@tool
def calcu_KD_w(code, period=9, init_k=50.0, init_d=50.0):
    """
    Calculate K and D values for a stock with a week as a unit.
    """
    df = stock_data(code)
    df = df.copy()
    df = df.xs(f"{code}.TW", axis=1, level=1)

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

    df.dropna(inplace=True)
    if df.empty:
        return {"error": "資料不足或代號錯誤"}

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date ascending
    df = df.sort_index()

    # --- Yearly average ---
    yearly_avg = df.resample('Y').mean()

    # --- Monthly averages per year ---
    monthly_avg = {}
    for year in df.index.year.unique():
        year_df = df[df.index.year == year]
        monthly_avg[year] = year_df.resample('M').mean()

    return yearly_avg, monthly_avg

@tool
def check_database_connection():
    """
    Checks the MongoDB database connection.
    """
    return mongo.check_db_connection()

@tool
def watchlist_information():
    """
    Reads and returns the stock watchlist from the MongoDB database.
    """
    watchlist = mongo.get_db()
    return str(watchlist)

@tool
def company_news(stock_id):
  """
  Fetch and get recent news about this company
  """
  # Recent news (availability varies)
  ticker_symbol = f"{stock_id}.TW" # Example stock
  ticker = yf.Ticker(ticker_symbol)
  news = []
  for item in ticker.news[:10]:
    news.append(item)
  return news


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
def calcu_KD_w_watchlist(collection_name,period=9, init_k=50.0, init_d=50.0):
    """
    Calculate weekly K and D values of the stocks in a collection of the watchlist database.It extracts the stock codes from the specified collection from the parameter (argument), then calculates K and D values for each stock, and finally returns the results in a dictionary format.
    Return the format {stock:{k:k_value, d:d_value}, stock_2...}
    """
    print("Watch list search....")

    # Ensure correct dtype
    codes = mongo.find_documents(collection_name)
    return calcu_KD_w_multiple(codes=[item["stock"] for item in codes], period=period, init_k=init_k, init_d=init_d)

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

##################Tool Ends###################


# Create agent
agent = create_agent(
    model=model,
    tools=[calcu_KD_d, calcu_KD_w, calcu_KD_w_multiple, stock_price_averages, calcu_KD_w_watchlist, watchlist_information, add_to_watchlist, check_database_connection, company_news, stock_per],
    system_prompt="You are a helpful assistant"
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


