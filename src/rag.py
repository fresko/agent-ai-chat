from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA #version deprecate until 2.08
from langchain.chains.question_answering import load_qa_chain

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
PATH_TO_DATA = "docs/url.txt"
PATH_TO_CHROMA = "docs/db"

class QAModel:
  def __init__(self):
    self.load_data()
    self.split_text()
    self.store_vector()
    #self.init_qa_retriever()
    self.init_qa_retriever_new()
    self.init_qa_chain()
  
  def call(self,request:str):
    return self.retriever({"query":request})  
    
  def call_new(self,request:str):
    return self.retriever.invoke({"input":request})

  def call_new2(self,retriever_doc,request:str):
    return self.qa_chain.run(input_documents=retriever_doc, question=request)  

  def load_data(self):
    with open(PATH_TO_DATA) as ps:
      urls = [source for source in ps.readlines()]
      self.urls = WebBaseLoader(urls).load()

      #for doc in self.urls:
       # print(f"Título: {doc.metadata.get('title', 'N/A')}")
        #print(f"URL: {doc.metadata['source']}")
        #print(f"Contenido: {doc.page_content}")
        #print("---")
  def split_text(self):
      text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=200,
      separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])
      self.chunks = text_splitter.split_documents(self.urls)
      
  def store_vector(self):
    self.vectorstores = Chroma.from_documents(
     documents=self.chunks ,
     embedding=GoogleGenerativeAIEmbeddings(model = "models/embedding-001"),
     #embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
     persist_directory=PATH_TO_CHROMA)
    
  #version deprecate QA
  def init_qa_retriever(self):
      self.retriever = RetrievalQA.from_chain_type(
      llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3),
      chain_type = "map_reduce",
      retriever = self.vectorstores.as_retriever(search_type="mmr"))
     

  #version Nueva  QA
  def init_qa_retriever_new(self):  
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") 
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    self.retriever = create_retrieval_chain(self.vectorstores.as_retriever(), combine_docs_chain)    
    #self.retriever.invoke({"input": "What are autonomous agents?"})   

  #Inicia la cadena de chat
  def init_qa_chain(self):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
    self.qa_chain = load_qa_chain(llm, chain_type="stuff")
    #answer = self.qa_chain.run(documents=retrieved_docs['documents'], question=user_input)