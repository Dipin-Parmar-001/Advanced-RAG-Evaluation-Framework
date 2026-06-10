from langchain_chroma import Chroma
from embeddings import GeminiEmbeddings

vector_store = Chroma(
    collection_name="pdf_embeddings",
    embedding_function=GeminiEmbeddings(),
    persist_directory="db"
)

data = vector_store.get()
print("Total docs:", len(data["documents"]))