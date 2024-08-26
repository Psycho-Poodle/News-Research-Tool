import os 
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()
# print(os.getenv("OPENAI_API_KEY"))



st.title("News Research TOOL")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
main_placefolder = st.empty()
llm = OpenAI(temperature=0.9,max_tokens=500)

process_url_clicked = st.sidebar.button("Process URLs")
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data loading....Started...✅✅✅✅✅")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placefolder.text("Text Splitting....Started...✅✅✅✅✅")
    docs = text_splitter.split_documents(data)
    
    
    #embbedings
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs,embedding=embeddings)
    main_placefolder.text("Embbeding Vectors....Started...✅✅✅✅✅")
    time.sleep(2)
    vectorstore_openai.save_local("vectorindex_openai")
    
query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists('vectorindex_openai'):
        embeddings = OpenAIEmbeddings()
        vectorindex = FAISS.load_local('vectorindex_openai', embeddings=embeddings,allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorindex.as_retriever())
        result = chain({"question": query},return_only_outputs=True)

        st.header("Answer")
        st.write(result['answer'])
        
        #Display Source if available
        sources = result.get("sources","")
        if sources:
            st.subheader("Sources:")
            source_list = sources.split("\n")
            for source in source_list:
                st.write(source)
            
            
    