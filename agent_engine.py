import os
import requests
from typing import List, Dict, Any

class CloudAgentEngine:
    def __init__(self):
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.grading_model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
        self.synthesis_model = "z-ai/glm-4.5-air:free"

    def _call_llm(self, model: str, prompt: str, temperature: float = 0.1) -> str:
        """Helper to route chat requests securely to the cloud"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }

        try:
            response = requests.post(self.openrouter_url, json=payload, headers=headers)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API error: {e}")
        
        return ""
    
    def grade_document_relevance(self, query: str, document_text: str) -> str:
        """Node: Evaluates if a retrieved chunk is actually useful."""
        
        prompt = f"""You are a retrieval grader. Evaluate if the following retrieved document contains context or facts relevant to the user's query.
        
User Query: "{query}"
Retrieved Document: "{document_text}" 

Respond with exactly one word: 'YES' if the document is relevant or provides any useful context to help answer the query, or 'NO' if it is completely irrelevant."""
        
        return self._call_llm(self.grading_model, prompt).upper()
    
    def grade_hallucination(self, context: str, generated_answer: str) -> str:
        """Node: Guardrail that checks if the LLM made up any facts."""
        prompt = f"""You are a Fact-Checking Specialist. Compare the generated answer against the approved source context text below.
        
Source Context:
{context}

Generated Answer:
{generated_answer}

Is the generated answer strictly grounded in the provided source context? Has the model invented any facts, numbers, or assumptions not found in the text?
Respond with exactly one word: 'PASSED' if the answer is completely faithful to the context, or 'FAILED' if there is any hallucination or unverified claim."""
        return self._call_llm(self.grading_model, prompt).upper()

    # Backwards compatibility alias
    def grade_hallunication(self, context: str, generated_answer: str) -> str:
        """Deprecated: Use grade_hallucination instead."""
        return self.grade_hallucination(context, generated_answer)
    
