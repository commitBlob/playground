import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

vectorstore = PineconeVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
    index_name="langgraph-agentic-rag",
)

retriever = PineconeVectorStore(
    index_name="langgraph-agentic-rag",
    embedding=OpenAIEmbeddings(),
    text_key="text"
).as_retriever()