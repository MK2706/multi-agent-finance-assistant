# Multi-Agent Finance Assistant

A voice-based assistant using FastAPI, LangChain, Whisper, and Streamlit.## 🚀 Docker Setup

1. **Build the Docker image**:

```bash
docker build -t multi-agent-assistant .
```

2. **Run each agent in a separate terminal or container**:

```bash
# Agent 1: API Agent
docker run -p 8001:8001 multi-agent-assistant uvicorn agents.api_agent:app --host 0.0.0.0 --port 8001

# Agent 2: Retriever Agent
docker run -p 8002:8002 multi-agent-assistant uvicorn agents.retriever_agent:app --host 0.0.0.0 --port 8002

# Agent 3: Voice Agent
docker run -p 8003:8003 multi-agent-assistant uvicorn agents.voice_agent:app --host 0.0.0.0 --port 8003

# Orchestrator
docker run -p 8000:8000 multi-agent-assistant
```

3. **Run the Streamlit App**:

```bash
docker run -p 8501:8501 multi-agent-assistant streamlit run streamlit_app/app.py
```