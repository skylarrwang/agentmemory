import numpy as np
from openai import OpenAI
from typing import List, Dict
from memory.topics import Topic
from dotenv import load_dotenv

# Ensure OPENAI_API_KEY is loaded from .env before creating the client
load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for text"""
    if not text.strip():
        return np.zeros(1536)  # OpenAI embedding dimension
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_relevant_topics(
    query: str,
    topics: List[Topic],
    embeddings: Dict[str, np.ndarray],
    max_k: int = 3,
    min_threshold: float = 0.5,
) -> List[Topic]:
    """Return top-k topics most relevant to the query using cosine similarity."""
    if not topics or not query.strip():
        return []
    
    query_emb = get_embedding(query)
    scores: List[tuple[float, Topic]] = []
    
    for topic in topics:
        topic_emb = embeddings.get(topic.id)
        if topic_emb is None:
            continue
        similarity = cosine_similarity(query_emb, topic_emb)
        scores.append((similarity, topic))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    return [topic for sim, topic in scores if sim > min_threshold][:max_k]