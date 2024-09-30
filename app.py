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
from flask_cors import CORS
load_dotenv()
app = Flask(__name__)
CORS(app)
agent_executor = None
def get_document_from_txt(url):
    loader = TextLoader(url,encoding='utf8')
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

def build_retriever_tool(vector_store, language, title, info):
    retriever = vector_store.as_retriever()
    retriever_tool = create_retriever_tool(
    retriever,
    title,
    info,
    )
    return retriever_tool

def create_agent(language, file, title, info):
    global agent_executor
    docs = get_document_from_txt(file)
    vector_store = create_db(docs)
    retriever_tool = build_retriever_tool(vector_store, language, title, info)
    search_tool = TavilySearchResults(max_results=2)
    tools = [retriever_tool, search_tool]
    model = ChatOpenAI(model="gpt-4")
    agent_executor = create_react_agent(model, tools)
    return agent_executor


@app.route('/seha/', methods=['POST'])
def get_Seha():
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent("", 'seha.md', "seha_information", f"Searches and returns information about Seha Organization in the UAE.")

    data = json.loads(request.data)
    input = data['input']   
    response = agent_executor.invoke({"messages": input})
    print(response['messages'][-1].content)
    return response['messages'][-1].content

#@app.route('/seha/create', methods=['POST'])
#def create_seha():
#    global agent_executor
#    data = json.loads(request.data)
#    language = data['language']  
#    agent_executor = create_agent(language)
#    return make_response('OK', 200)

@app.route('/oman/', methods=['POST'])
def get_oman():
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent("",'KnowledgeBase.txt', "oman_law_2034", f"Searches and returns information about Oman law number 2034.")

    data = json.loads(request.data)
    input = data['input']   
    response = agent_executor.invoke({"messages": input})
    print(response['messages'][-1].content)
    return response['messages'][-1].content

@app.route('/reba/', methods=['POST'])
def get_oman():
    global agent_executor
    if agent_executor is None:
        agent_executor = create_agent("",'KnowledgeBaseRefined.md', "Definition of Reba in Islam", f"Searches and returns information about Reba in Islam, and how Islamic banking solved this issue.")

    data = json.loads(request.data)
    input = data['input']   
    response = agent_executor.invoke({"messages": input})
    print(response['messages'][-1].content)
    return response['messages'][-1].content


if __name__ == '__main__':
    app.run(debug=True)
