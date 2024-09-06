import json
from flask import Flask, jsonify, request, make_response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
load_dotenv()

agent_executor = None
def get_document_from_txt(url):
    loader = TextLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    spliDocs = splitter.split_documents(docs)
    return spliDocs
def create_db(docs):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    return vector_store

def build_retriever_tool(vector_store, language):
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
    retriever,
    "seha_information",
    f"Searches and returns information about Seha Organization in the UAE. Onle use {language} language to reply",
    )
    return retriever_tool

def create_agent(language):
    global agent_executor
    docs = get_document_from_txt("seha.md")
    vector_store = create_db(docs)
    retriever_tool = build_retriever_tool(vector_store, language)
    search_tool = TavilySearchResults(max_results=2)
    tools = [retriever_tool, search_tool]
    model = ChatOpenAI(model="gpt-4")
    agent_executor = create_react_agent(model, tools)
    return agent_executor
data = json.loads(request.data)
input = data['input']   
response = agent_executor.invoke({"messages": input})
print(response['messages'][-1].content)