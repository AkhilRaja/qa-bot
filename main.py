import os
from fastapi import FastAPI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastapi.responses import StreamingResponse

app = FastAPI()

# Set the OpenAI API key
openai_api_key = "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"
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load your text data using the TextLoader
loader = PyPDFLoader("test-notebooks/sample.pdf")
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

@app.get("/qa")
async def run_qa():
    # Run the QA system by providing a question to the chain
    question = "What is this document ?"
    answer = qa.run(question)
    return StreamingResponse(iter([answer]), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
