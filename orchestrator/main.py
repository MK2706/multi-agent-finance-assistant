from fastapi import FastAPI
import requests

app = FastAPI()

@app.get("/morning-brief")
def morning_brief():
    stock_data = requests.post("http://localhost:8001/stock", json={"ticker": "AAPL"}).json()
    voice = requests.get("http://localhost:8003/speak").json()
    rag = requests.get("http://localhost:8002/rag").json()

    return {
        "stock": stock_data,
        "rag_summary": rag["result"],
        "voice_status": voice["status"]
    }