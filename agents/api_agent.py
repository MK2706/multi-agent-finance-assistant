from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

@app.post("/stock")
def get_stock_info(request: StockRequest):
    data = yf.Ticker(request.ticker)
    hist = data.history(period="2d")
    if hist.empty:
        return {"error": "No data found"}
    
    latest = hist.iloc[-1]
    prev = hist.iloc[-2]
    change = round((latest['Close'] - prev['Close']) / prev['Close'] * 100, 2)
    
    return {
        "ticker": request.ticker,
        "price": latest['Close'],
        "change_percent": change
    }