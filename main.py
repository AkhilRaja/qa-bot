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

# Set the OpenAI API key
openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

# Function to process PDF and questions
def process_qa(pdf_path: str, questions: List[str]):
    # Load your text data using the TextLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Generate VectorDB using Chroma and OpenAIEmbeddings
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    # Create a ChatOpenAI instance for interactive chat using the OpenAI model
    llm = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        openai_api_key=openai_api_key,
    )

    # Create a RetrievalQA chain using the ChatOpenAI model and the document retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Process each question and get answers
    results = []
    for question in questions:
        answer = qa.run(question)
        results.append({"question": question, "answer": answer})

    return results

@app.post("/qa")
async def run_qa(pdf_file: UploadFile = File(...), questions: List[str] = None):
    if questions is None:
        questions = []
    
    # Save the uploaded PDF file to a temporary location
    with open(pdf_file.filename, "wb") as pdf:
        pdf.write(pdf_file.file.read())

    try:
        # Process PDF and questions
        results = process_qa(pdf_file.filename, questions)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up: Remove the temporary PDF file
        os.remove(pdf_file.filename)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
