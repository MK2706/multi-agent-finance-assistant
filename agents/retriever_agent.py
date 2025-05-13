from fastapi import FastAPI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

app = FastAPI()

@app.get("/rag")
def get_answer():
    loader = TextLoader("data/sample_finance.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    query = "What happened in the market yesterday?"
    docs = retriever.get_relevant_documents(query)

    return {"result": docs[0].page_content}