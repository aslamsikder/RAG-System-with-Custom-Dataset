import streamlit as st
from config import load_env
from data_loader import load_and_split
from embeddings import create_vector_store
from chain_builder import build_retriever_chain
from metrics import evaluate_retrieval

# -----------------------------
# Load environment
# -----------------------------
load_env()

st.set_page_config(page_title="Bangla Q&A + Metrics Auto", page_icon="📖")
st.title("RAG System for Answering the Question with related Explanation")

# -----------------------------
# Initialize DB and Chain
# -----------------------------
@st.cache_resource
def init_db(file_path="data/questions.csv"):
    documents = load_and_split(file_path)
    db = create_vector_store(documents)
    retriever_chain = build_retriever_chain(db)
    return retriever_chain, db, documents

retriever_chain, db, documents = init_db()
retriever = db.as_retriever()

# -----------------------------
# User Input
# -----------------------------
user_query = st.text_input("Enter your question (Bangla):")

if st.button("Get Answer") and user_query:
    with st.spinner("Thinking..."):
        # Get LLM Answer
        response = retriever_chain.invoke({"input": user_query})
        st.subheader("Answer:")
        st.write(response['answer'])

        # Automatically find the most similar document as ground truth
        retrieved_docs = retriever.get_relevant_documents(user_query)
        if retrieved_docs:
            ground_truth_doc = retrieved_docs[0]  # top-1 document
            ground_truth_id = ground_truth_doc.metadata.get("id")
            st.subheader("Auto-detected Ground Truth Document ID:")
            st.write(ground_truth_id)

            # Compute retrieval metrics for top-k evaluation
            metrics = evaluate_retrieval(
                retriever,
                queries=[user_query],
                ground_truths=[ground_truth_id]
            )
            st.subheader("Retrieval Metrics:")
            st.json(metrics)
        else:
            st.warning("No documents retrieved for this query.")
