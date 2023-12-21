# app/qa_processor.py
import json
import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.config import Config  # Import Config class from config.py

class QAProcessor:
    def __init__(self, pdf_path: str, questions: List[str]):
        self.pdf_path = pdf_path
        self.questions = questions
        self.config = Config()  # Create an instance of Config

    def process_qa(self):
        # Check if the file exists
        if not os.path.isfile(self.pdf_path):
            raise ValueError(f"File not found at path: {self.pdf_path}")

        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()

        # Generate VectorDB using Chroma and OpenAIEmbeddings
        text_splitter = CharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        docsearch = Chroma.from_documents(texts, self.config.embeddings)

        # Create a RetrievalQA chain using the ChatOpenAI model and the document retriever
        qa = RetrievalQA.from_chain_type(
            llm=self.config.llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )

        # Process each question and get answers
        results = []
        for question in self.questions:
            answer = qa.run(question)
            results.append({"question": question, "answer": answer})
            yield json.dumps({"question": question, "answer": answer}).encode("utf-8")

        # Return the accumulated results as a list
        return results
