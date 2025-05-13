import streamlit as st
from financial_assistant import initialize_session_state, get_stock_data, scrape_filings, get_yahoo_news, extract_symbol, add_to_index, search_index, generate_answer_gemini

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
