__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Streamlit 인터페이스 설정
st.title("PDF Service with ChatGPT")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
api_key = st.text_input("Enter your API Key")
question = st.text_input("Enter your question")

if st.button("Process"):
    if uploaded_file is not None and api_key and question:
        # API 키 설정
        os.environ["OPENAI_API_KEY"] = api_key

        # 파일 처리
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PDF 로더 및 텍스트 분리 설정
        loader = PyPDFLoader(uploaded_file.name)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        # PDF 문서 로드 및 분리
        pages = text_splitter.split_documents(loader.load())

        # 임베딩과 Chroma
        embeddings_model = OpenAIEmbeddings()
        db = Chroma.from_documents(pages, embeddings_model)

        # 질의 및 검색
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=4096)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        result = qa_chain({"query": question })

        # 결과 표시
        st.write(result)
    else:
        st.warning("Please upload a file, enter API key and a question.")
