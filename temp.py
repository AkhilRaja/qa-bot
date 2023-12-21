import os
from functools import lru_cache
from typing import AsyncGenerator, Literal

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseChatMessageHistory, Document, format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

from pydantic import BaseModel, BaseSettings


from contextlib import asynccontextmanager


class Settings(BaseSettings):
    openai_api_key = "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"

    class Config:  # type: ignore
        env_file = ".env"
        env_file_encoding = "utf-8"


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatSSEResponse(BaseModel):
    type: Literal["context", "start", "streaming", "end", "error"]
    value: str | list[Document]


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore


@lru_cache()
def get_vectorstore() -> Chroma:
    settings = get_settings()

    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)  # type: ignore
    print("Debug Statement here")
    vectorstore = Chroma(
        collection_name="chroma",
        embedding_function=embeddings,
        persist_directory="chroma",
    )
    print("There are", vectorstore._collection.count(), "in the collection")
    return vectorstore


def combine_documents(
    docs: list[Document],
    document_prompt: PromptTemplate = PromptTemplate.from_template("{page_content}"),
    document_separator: str = "\n\n",
) -> str:
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)




app = FastAPI(
    title="QA Chatbot Streaming using FastAPI, LangChain Expression Language , OpenAI, and Chroma",
    version="0.1.0",
)

# Deprecated
@app.on_event("startup")
async def startup_event() -> None:
    # vectorstore = get_vectorstore()

    # is_collection_empty: bool = vectorstore._collection.count() == 0  # type: ignore

    # if is_collection_empty:
    #     vectorstore.add_texts(  # type: ignore
    #         texts=[
    #             "Cats are playing in the garden.",
    #             "Dogs are playing in the river.",
    #             "Dogs and cats are mortal enemies, but they often play together.",
    #         ]
    #     )

    # if not os.path.exists("message_store"):
    #     os.mkdir("message_store")


    loader = PyPDFLoader("sample.pdf")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})



async def generate_standalone_question(
    chat_history: str, question: str, settings: Settings
) -> str:
    prompt = PromptTemplate.from_template(
        template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    )
    llm = ChatOpenAI(temperature=0, openai_api_key=settings.openai_api_key)

    chain = prompt | llm | StrOutputParser()  # type: ignore

    return await chain.ainvoke(  # type: ignore
        {
            "chat_history": chat_history,
            "question": question,
        }
    )


async def search_relevant_documents(query: str, k: int = 5) -> list[Document]:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    return await retriever.aget_relevant_documents(query=query, k=k)


async def generate_response(
    context: str, chat_memory: BaseChatMessageHistory, message: str, settings: Settings
) -> AsyncGenerator[str, None]:
    prompt = PromptTemplate.from_template(
        """Answer the question based only on the following context:
{context}
Question: {question}"""
    )

    llm = ChatOpenAI(temperature=0, openai_api_key=settings.openai_api_key)

    chain = prompt | llm  # type: ignore

    response = ""
    async for token in chain.astream({"context": context, "question": message}):  # type: ignore
        yield token.content
        response += token.content

    chat_memory.add_user_message(message=message)
    chat_memory.add_ai_message(message=response)


@app.post("/chat")
async def chat(
    request: ChatRequest, settings: Settings = Depends(get_settings)
) -> StreamingResponse:
    memory_key = f"./message_store/{request.session_id}.json"

    chat_memory = FileChatMessageHistory(file_path=memory_key)
    memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)

    # standalone_question = await generate_standalone_question(
    #     chat_history=memory.buffer, question=request.message, settings=settings
    # )

    # relevant_documents = await search_relevant_documents(query=standalone_question)

    # combined_documents = combine_documents(relevant_documents)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever)
    chat_history = []
    query = "Give me a summary of the document"
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])

    return StreamingResponse(
        generate_response(
            context=combined_documents,
            chat_memory=chat_memory,
            message=request.message,
            settings=settings,
        ),
        media_type="text/plain",
    )

@app.get("/") 
async def main_route():     
  return {"message": "Hey, It is me Goku"}

# @app.get("/hello")
# async def hello():
#     return "Hello World"

# @app.post("/chat/sse/")
# async def chat_sse(
#     request: ChatRequest, settings: Settings = Depends(get_settings)
# ) -> StreamingResponse:
#     memory_key = f"./message_store/{request.session_id}.json"

#     chat_memory = FileChatMessageHistory(file_path=memory_key)
#     memory = ConversationBufferMemory(chat_memory=chat_memory, return_messages=False)

#     standalone_question = await generate_standalone_question(
#         chat_history=memory.buffer, question=request.message, settings=settings
#     )

#     relevant_documents = await search_relevant_documents(query=standalone_question, k=2)

#     return StreamingResponse(
#         generate_sse_response(
#             context=relevant_documents,
#             chat_memory=chat_memory,
#             message=request.message,
#             settings=settings,
#         ),
#         media_type="text/event-stream",
#     )