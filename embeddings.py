import os
import time
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

class OpenRouterEmbeddings(Embeddings):
    """
    LangChain compatible embeddings class using OpenRouter API.
    """
    def __init__(self, model_name: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

    def _get_embedding_for_chunk(self, text: str) -> List[float]:
        """
        Formats and sends a single chunk using the structured content layout
        required by the multimodal Nemotron model.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Advanced RAG Application"
        }
        
        # This mirrors the structured requestBody layout shown on OpenRouter
        payload = {
            "model": self.model_name,
            "input": [
                {
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ],
            "encoding_format": "float"
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"OpenRouter Error {response.status_code}: {response.text}")
            
        data = response.json()
        # Parse out the embedding vector from the inner structure
        return data["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Loops through your chunks and structures each one for ChromaDB"""
        if not texts:
            return []
            
        # Call them individually to match the model's specialized payload format
        return [self._get_embedding_for_chunk(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embeds the user search query"""
        return self._get_embedding_for_chunk(text)


def _post_with_retry(url: str, json: dict, max_retries: int = 5, initial_backoff: float = 2.0) -> requests.Response:
    """Helper function to perform HTTP POST request with exponential backoff on rate limits."""
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=json)
            if response.status_code == 429:
                print(f"[Gemini Embeddings] Rate limited (429). Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            return response
        except Exception as e:
            print(f"[Gemini Embeddings] Request failed: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            backoff *= 2
    return requests.post(url, json=json)


class GeminiEmbeddings(Embeddings):
    """
    LangChain compatible embeddings class using Google's Gemini API.
    """
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent?key={self.api_key}"
        self.batch_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents?key={self.api_key}"

    def _get_embedding_for_chunk(self, text: str) -> List[float]:
        payload = {
            "model": f"models/{self.model_name}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        response = _post_with_retry(self.api_url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Gemini API Error {response.status_code}: {response.text}")
        data = response.json()
        return data["embedding"]["values"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        payload = {
            "requests": [
                {
                    "model": f"models/{self.model_name}",
                    "content": {
                        "parts": [{"text": t}]
                    }
                }
                for t in texts
            ]
        }
        response = _post_with_retry(self.batch_api_url, json=payload)
        if response.status_code != 200:
            # Fallback to individual embeddings with retry
            return [self._get_embedding_for_chunk(text) for text in texts]
        
        data = response.json()
        return [item["values"] for item in data["embeddings"]]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding_for_chunk(text)