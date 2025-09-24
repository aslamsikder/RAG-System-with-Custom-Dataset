import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -----------------------------
# 1. Load Environment Variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# -----------------------------
# 2. Load and Prepare Data
# -----------------------------
@st.cache_resource
def load_data(file_path: str):
    """Load, split, and embed CSV data into Chroma DB."""
    loader = CSVLoader(file_path="data/questions.csv", encoding="utf-8")
    data = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(data)

    # Create Chroma vector database
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db


# -----------------------------
# 3. Setup LLM & Prompt Template
# -----------------------------
def create_chain(db):
    """Create Retrieval Chain using Chroma retriever and OpenAI Chat model."""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer and explain the following question based only on the provided context.
    Think step by step before providing a detailed answer and explanation.
    You are the most talented question-answer generator and always helpful 
    to find the actual answer. 

    - If the question is asked in Bangla, answer and explain in Bangla.
    - If the question is asked in English, answer and explain in English.
    - Provide the actual answer, e.g., Option 1: ক) or Option 2: খ) or Option 3: গ) or Option 4: ঘ).

    <context>
    {context}
    </context>
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    return retriever_chain


# -----------------------------
# 4. Streamlit Frontend
# -----------------------------
def main():
    st.set_page_config(page_title="Bangla Q&A with LLM", page_icon="📖")
    st.title("RAG System for Answering the Question with related Explanation")

    # Load Database
    db = load_data("data/questions.csv")
    retriever_chain = create_chain(db)

    # User Input
    user_query = st.text_input("Enter your question (Bangla):")

    if st.button("Get Answer") and user_query:
        with st.spinner("Thinking..."):
            response = retriever_chain.invoke({"input": user_query})
            st.subheader("Answer:")
            st.write(response['answer'])


if __name__ == "__main__":
    main()
