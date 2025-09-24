from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(documents):
    """Create Chroma vector store from document chunks."""
    db = Chroma.from_documents(documents, OpenAIEmbeddings())
    return db
