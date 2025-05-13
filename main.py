import streamlit as st
import yfinance as yf
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

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
        st.error(f"Failed to fetch stock data: {str(e)}")
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
        st.error(f"Failed to fetch filings: {str(e)}")
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
        st.error(f"Failed to fetch news: {str(e)}")
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

# ==============================
# Main App
# ==============================
def main():
    st.set_page_config(page_title="Financial Assistant", layout="wide")
    st.title("💹  Financial Assistant")
    st.write("Ask me about market sentiment, stock performance, or financial summaries.")

    initialize_session_state()

    with st.sidebar:
        st.header("Tracked Companies")
        st.markdown("- Apple\n- Microsoft\n- Amazon\n- Google\n- Tesla\n- Nvidia\n- Meta\n- Netflix\n- TSMC\n- Samsung")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Ask your financial question")
        user_question = st.text_input("Type a market-related query and press Enter", key="user_question_input")

        if user_question:
            with st.spinner("Analyzing your question..."):
                try:
                    symbol = extract_symbol(user_question)
                    if symbol:
                        st.success(f"🔎 Detected symbol: {symbol}")

                        stock_data = get_stock_data(symbol)
                        filings_data = scrape_filings(symbol)
                        news_headlines = get_yahoo_news(symbol)

                        if stock_data:
                            add_to_index(str(stock_data), {"source": "stock_data", "symbol": symbol})
                        if filings_data:
                            add_to_index(str(filings_data), {"source": "filings_data", "symbol": symbol})
                        if news_headlines:
                            news_text = "\n".join(news_headlines)
                            add_to_index(news_text, {"source": "news", "symbol": symbol})

                        search_results = search_index(user_question)
                        answer = generate_answer_gemini(user_question, search_results)

                        st.subheader("💡 Generated Response")
                        st.markdown(f"**{answer}**")

                    else:
                        st.warning("⚠️ Please mention a supported company like Tesla, Samsung, or Meta.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with col2:
        st.subheader("Example Prompts")
        st.subheader("How It Works")
        st.markdown("""
        1. Identifies mentioned company.  
        2. Fetches live Yahoo Finance data + headlines.  
        3. Embeds + indexes data using FAISS.  
        4. Uses Gemini to answer with context-aware responses.
        """)

if __name__ == "__main__":
    main()
