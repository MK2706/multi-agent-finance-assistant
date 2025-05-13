# financial_assistant.py

import yfinance as yf
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# ==============================
# CONFIGURATION - Gemini API Key
# ==============================
GENAI_API_KEY = "AIzaSyAthCtTVRbEHD6qT-hyj5wY5nHH_bqXeFA"  # 🔐 Replace this with your actual Gemini API key
genai.configure(api_key=GENAI_API_KEY)

# ==============================
# Initialize session state
# ==============================
def initialize_session_state():
    if 'initialized' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            dummy_texts = ["init text"]
            st.session_state.vectorstore = FAISS.from_texts(dummy_texts, embeddings)
            st.session_state.qa_prompt = PromptTemplate(
                template="""Use the following context to answer the question.
                Context: {context}
                Question: {question}
                Answer:""",
                input_variables=["context", "question"]
            )
            st.session_state.embeddings = embeddings
            
            try:
                gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")
                st.session_state.gemini_model = gemini_model
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"⚠️ Failed to initialize Gemini model. Error: {str(e)}")
                st.session_state.gemini_model = None
                st.session_state.model_loaded = False
            
            st.session_state.initialized = True

# ==============================
# Helper Functions
# ==============================
def add_to_index(text, metadata):
    st.session_state.vectorstore.add_texts([text], metadatas=[metadata])

def search_index(query, k=3):
    return st.session_state.vectorstore.similarity_search_with_score(query, k=k)

def generate_answer_gemini(question, context_docs):
    if 'gemini_model' not in st.session_state or st.session_state.gemini_model is None:
        return "⚠️ Gemini model is not available. Please check your API key."
    
    try:
        context_text = "\n".join([doc[0].page_content for doc in context_docs])
        prompt = f"""You are a financial assistant. Use the following data to generate a market-style report like a Bloomberg analyst.
Context:
{context_text}

Question:
{question}

Give a professional, crisp answer with key figures and sentiment."""
        response = st.session_state.gemini_model.generate_content(prompt)
        return response.text if hasattr(response, 'text') else "Received unexpected response format from Gemini."
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

def get_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        return {
            "info": stock.info,
            "history": stock.history(period="5d"),
            "recommendations": stock.recommendations,
            "actions": stock.actions
        }
    except Exception as e:
        return None

def scrape_filings(symbol):
    try:
        stock = yf.Ticker(symbol)
        return {
            "financials": stock.financials.to_string(),
            "balance_sheet": stock.balance_sheet.to_string(),
            "cashflow": stock.cashflow.to_string()
        }
    except Exception as e:
        return None

def get_yahoo_news(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}?p={symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        headlines = []
        for h3 in soup.find_all("h3"):
            text = h3.get_text()
            if text and symbol.upper() in text.upper():
                headlines.append(text.strip())
        return headlines[:5]
    except Exception as e:
        return []

def extract_symbol(text):
    symbol_map = {
        "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
        "google": "GOOGL", "tesla": "TSLA", "nvidia": "NVDA",
        "meta": "META", "netflix": "NFLX", "tsmc": "TSM", "samsung": "005930.KQ"
    }
    if not text:
        return None
    text_lower = text.lower()
    for name, symbol in symbol_map.items():
        if name in text_lower:
            return symbol
    return None
