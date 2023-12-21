import json
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = FastAPI()


class QAProcessor:
    def __init__(self, pdf_path: str, questions: List[str]):
        self.pdf_path = pdf_path
        self.questions = questions

    def process_qa(self):
        self._validate_file()
        documents = self._load_documents()
        docsearch = self._create_doc_search(documents)
        qa = self._create_qa_chain(docsearch)
        return self._get_answers(qa)

    def _validate_file(self):
        if not os.path.isfile(self.pdf_path):
            raise ValueError(f"File not found at path: {self.pdf_path}")

    def _load_documents(self):
        loader = PyPDFLoader(self.pdf_path)
        return loader.load()

    def _create_doc_search(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        return Chroma.from_documents(texts, embeddings)

    def _create_qa_chain(self, docsearch):
        llm = ChatOpenAI(
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )

    def _get_answers(self, qa):
        results = []
        for question in self.questions:
            answer = qa.run(question)
            results.append({"question": question, "answer": answer})
            yield json.dumps({"question": question, "answer": answer}).encode("utf-8")

        return results


@app.post("/qa")
async def run_qa(pdf_file: UploadFile = File(...), questions: List[str] = None):
    if questions is None:
        questions = []

    processor = QAProcessor(pdf_file.filename, questions)
    
    # Save the uploaded PDF file to a temporary location
    with open(pdf_file.filename, "wb") as pdf:
        pdf.write(pdf_file.file.read())

    try:
        # Process PDF and questions
        return StreamingResponse(
            processor.process_qa(),
            media_type="application/json",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
