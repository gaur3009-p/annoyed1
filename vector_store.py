import chromadb
from sentence_transformers import SentenceTransformer
import uuid

client = chromadb.Client()
collection = client.get_or_create_collection("rookus_campaigns")

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def store_campaign(text, metadata):
    emb = embedder.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[emb],
        metadatas=[metadata],
        ids=[str(uuid.uuid4())]
    )


def search_similar(query, k=3):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    return results
