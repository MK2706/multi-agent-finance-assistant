import streamlit as st
import requests

st.title("📈 Morning Market Brief")

if st.button("Get Morning Brief"):
    res = requests.get("http://localhost:8000/morning-brief").json()
    
    st.subheader("Stock Data")
    st.json(res["stock"])
    
    st.subheader("Summary")
    st.write(res["rag_summary"])
    
    st.audio("voice.mp3")