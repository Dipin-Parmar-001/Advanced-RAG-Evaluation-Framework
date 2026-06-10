from retrieval_engine import AdvancedRetrievalEngine
from embeddings import GeminiEmbeddings
from langchain_chroma import Chroma

DB_DIR = "db"
embeddings = GeminiEmbeddings()

vector_store = Chroma(
    collection_name="pdf_embeddings",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

engine = AdvancedRetrievalEngine()

query = input("Enter your query: ")

result = engine.retrieve(query=query, vector_store=vector_store, k=20)

print("\nQUERY:", query)

for i, chunk in enumerate(result.chunks):
    print(f"\nResult {i}:")
    print(chunk.document.metadata)
    print(chunk.document.page_content[0:500])

