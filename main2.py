
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrappers = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool =WikipediaQueryRun(api_wrapper=api_wrappers)

wiki_tool.name
from google.colab import userdata
token=userdata.get('HF_TOKEN')

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceInstructEmbeddings
from langchain.embeddings import VertexAIEmbeddings, SentenceTransformerEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

## Loading the documents form the website ##
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
text_spliter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =50).split_documents(docs)


## Creating the vector store of the documents
from langchain.embeddings import SentenceTransformerEmbeddings
embedding = SentenceTransformerEmbeddings(model_name='hkunlp/instructor-xl') # Use SentenceTransformerEmbeddings
vectordb = FAISS.from_documents(text_spliter ,embedding)
retriever = vectordb.as_retriever()
retriever

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(retriever, "LangSmith Search",
                                       'Search for any information about langsmith or you can search about langsmith here.')

retriever_tool.name

from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_api_wrapper = ArxivAPIWrapper(top_k_search=1,docs_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)

arxiv_tool.name

tools = [wiki_tool,retriever_tool, arxiv_tool]

# !pip install -q langchain_llms

import google.generativeai as genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel('gemini-pro')

from langchain.agents import create_openai_functions_agent

from langchain import hub

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    template="You are very helpful assistant. {tools} {agent_scratchpad} {tool_names} {input}",
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
)

from langchain.llms import GenerativeAI

from langchain.agents import create_react_agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

