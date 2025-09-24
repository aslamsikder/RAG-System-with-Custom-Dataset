from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def build_retriever_chain(db):
    """Create retrieval chain using Chroma retriever and OpenAI LLM."""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer and explain the following question based only on the provided context.
    Think step by step before providing a detailed answer and explanation.
    You are the most talented question-answer generator and always helpful.

    - If the question is in Bangla, answer in Bangla.
    - If in English, answer in English.
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
