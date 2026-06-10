import os
from typing import List, Tuple, Any
import requests
from dotenv import load_dotenv

load_dotenv()

class SiliconFlowReranker:
    """
    Reranker using SiliconFlow's API model Qwen/Qwen3-Reranker-8B.
    """
    def __init__(self) -> None:
        self.api_url = "https://api.siliconflow.com/v1/rerank"
        self.api_key = os.getenv("SILICON_FLOW_API_KEY")
    
    def rerank(self, query: str, chunks: List[Any]) -> List[Tuple[Any, float]]:
        """
        Reranks a list of chunks based on query relevance using SiliconFlow.
        
        Args:
            query: The user search query.
            chunks: A list of RetrievedChunk objects.
            
        Returns:
            A list of tuples (RetrievedChunk, relevance_score) sorted descending by score.
        """
        documents = [
            chunk.document.page_content for chunk in chunks
        ]
        payload = {
            "model": "Qwen/Qwen3-Reranker-8B",
            "query": query,
            "documents": documents,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print("Status:", response.status_code)
            print("Response:", response.text)

        response.raise_for_status()

        data = response.json()
        ranked = []

        for item in data["results"]:
            idx = item["index"]
            score = item["relevance_score"]
            ranked.append((chunks[idx], score))
        
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked
