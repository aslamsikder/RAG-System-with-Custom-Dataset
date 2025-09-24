from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split(file_path: str, chunk_size=500, chunk_overlap=100):
    """Load CSV, split documents, and assign unique IDs."""
    loader = CSVLoader(file_path="data/questions.csv", encoding="utf-8")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = splitter.split_documents(data)

    # Assign unique ID to each chunk for evaluation
    for i, doc in enumerate(documents):
        if not hasattr(doc, "metadata"):
            doc.metadata = {}
        doc.metadata["id"] = f"doc_{i}"
    return documents
