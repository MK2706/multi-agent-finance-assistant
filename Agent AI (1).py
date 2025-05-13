#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# financial_agent.py
import streamlit as st
import yfinance as yf
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import whisper
from gtts import gTTS
import io
import os
import base64

# =============================================
# ADD YOUR OPENAI API KEY HERE (OPTION 1 - RECOMMENDED)
# Option 1: Create a secrets.toml file in .streamlit folder with:
# OPENAI_API_KEY = "your-api-key-here"
# =============================================

# Initialize all components in session state
def initialize_session_state():
    if 'initialized' not in st.session_state:
        # =============================================
        # ADD YOUR OPENAI API KEY HERE (OPTION 2)
        # Option 2: Uncomment and set your key here:
        # os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
        # =============================================
        
        try:
            # Initialize models
            with st.spinner("Loading AI models (this may take a few minutes)..."):
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.whisper_model = whisper.load_model("base")
                st.session_state.faiss_index = faiss.IndexFlatL2(384)
                st.session_state.document_store = {}
                st.session_state.doc_counter = 0
                
                # Initialize LLM chain
                template = """You are a financial analyst assistant. Use the following context to answer the question.
                If you don't know the answer, say you don't know. Always provide accurate financial information.

                Context: {context}

                Question: {question}
                Answer:"""
                st.session_state.qa_prompt = PromptTemplate(
                    template=template, input_variables=["context", "question"]
                )
                
                st.session_state.initialized = True
                st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"Error initializing models: {str(e)}")
            st.session_state.model_loaded = False

# [Rest of the functions remain the same as previous version...]
# (add_to_index, search_index, generate_answer, speech_to_text, text_to_speech, 
#  get_stock_data, scrape_filings, extract_symbol)

def display_audio_player(audio_bytes):
    """Display audio player and download button for TTS output"""
    st.audio(audio_bytes, format='audio/mp3')
    
    # Add download button
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="response.mp3">Download Audio Response</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Financial Analysis Agent", layout="wide")
    st.title("💰 Financial Analysis Agent")
    st.write("Ask questions about stocks and company financials")
    
    initialize_session_state()
    
    if not st.session_state.get('model_loaded', False):
        st.warning("Models failed to load. Please check your setup.")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        input_method = st.radio("Input method:", ["Text", "Voice"])
        
        # API key input (Option 3)
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Get your key from https://platform.openai.com/account/api-keys")
        if api_key:
            os.environ['OPENAI_API_KEY'] = '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'
        
        st.markdown("---")
        st.markdown("**Supported Companies:** Apple, Microsoft, Amazon, Google, Tesla, Nvidia, Meta, Netflix")
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        user_input = None
        
        if input_method == "Voice":
            st.subheader("Voice Input")
            audio_bytes = st.file_uploader("Upload voice recording (MP3/WAV)", 
                                         type=["wav", "mp3"],
                                         help="Record a question about a company's financials")
            if audio_bytes:
                with st.spinner("Processing voice input..."):
                    try:
                        user_input = speech_to_text(audio_bytes.read())
                        st.text_area("Transcribed Text", value=user_input, height=100)
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
        else:
            st.subheader("Text Input")
            user_input = st.text_area("Enter your financial question", 
                                     height=100,
                                     placeholder="E.g., What's Apple's current P/E ratio?")
        
        if user_input and st.button("Get Answer", use_container_width=True):
            with st.spinner("Analyzing your question..."):
                try:
                    # Step 1: Extract symbol
                    symbol = extract_symbol(user_input)
                    
                    if symbol:
                        st.success(f"Analyzing: {symbol}")
                        
                        # Step 2: Get data
                        with st.expander("View raw data", expanded=False):
                            tab1, tab2 = st.tabs(["Stock Data", "Filings Data"])
                            
                            with tab1:
                                stock_data = get_stock_data(symbol)
                                st.write("Latest Stock Data:")
                                st.dataframe(stock_data['history'])
                            
                            with tab2:
                                filings_data = scrape_filings(symbol)
                                st.write("Corporate Actions:")
                                st.dataframe(filings_data['actions'])
                        
                        # Step 3: Index data
                        add_to_index(str(stock_data), {"source": "stock_data", "symbol": symbol})
                        add_to_index(str(filings_data), {"source": "filings_data", "symbol": symbol})
                        
                        # Step 4: Search
                        search_results = search_index(user_input)
                        
                        # Step 5: Generate answer
                        context = [result["document"] for result in search_results]
                        answer = generate_answer(user_input, context)
                        
                        # Display results
                        st.subheader("Analysis Result")
                        st.markdown(f"**{answer}**")
                        
                        # Add audio output option
                        if input_method == "Voice":
                            try:
                                audio_output = text_to_speech(answer)
                                display_audio_player(audio_output)
                            except Exception as e:
                                st.error(f"Error generating speech: {str(e)}")
                        
                        # Show sources
                        with st.expander("See analysis sources"):
                            st.json(search_results)
                    else:
                        st.warning("Could not identify a company in your query. Try mentioning a company name like Apple or Microsoft.")
                except Exception as e:
                    st.error(f"Error processing your request: {str(e)}")
    
    with col2:
        st.subheader("Example Questions")
        st.markdown("""
        - What's Apple's current P/E ratio?
        - Show me Microsoft's recent stock performance
        - What were Amazon's last quarterly earnings?
        - Is Tesla's stock overvalued?
        - What's Nvidia's current market cap?
        """)
        
        st.markdown("---")
        st.subheader("How It Works")
        st.markdown("""
        1. Enter your question about a public company
        2. The system:
           - Identifies the company
           - Fetches real market data
           - Analyzes financial information
           - Generates a natural language response
        3. Get both text and audio answers
        """)

if __name__ == "__main__":
    main()

