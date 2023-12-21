# app/config.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import os

class Config:
    model = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model=model, temperature=0)
    embeddings = OpenAIEmbeddings()
    chunk_size = 2000
    chroma_persist_directory = 'chroma_store'
