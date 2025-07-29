import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb import PersistentClient

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: Documents) -> list:
        return self.model.encode(input).tolist()

class ChromaClient:
    def __init__(self, vector_name="default", path="./chroma_store"):
        self.vector_name = vector_name
        self.id = 100

        self.chroma_client = PersistentClient(path=path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=vector_name,
            embedding_function=SentenceTransformerEmbeddingFunction()
        )

    def add_document(self, content, metadata):
        self.collection.add(documents=[content], metadatas=[metadata], ids=[str(self.id)])
        self.id += 1

    def query(self, query_text, top_k=10, include=["documents", "metadatas"]):
        return self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=include
        )

    def query_score(self, query_text, top_k=10, include=["documents", "metadatas", "distances"]):
        return self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=include
        )