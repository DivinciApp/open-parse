import os
import time
import requests
import logging

from typing import List, Literal
from requests.exceptions import RequestException

# Create custom logger for Ollama
ollama_logger = logging.getLogger('ollama')
ollama_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - OLLAMA - %(message)s')
handler.setFormatter(formatter)
ollama_logger.addHandler(handler)

OllamaModel = Literal["bge-large", "nomic-embed-text"]


class OllamaEmbeddings:
    def __init__(
        self,
        model: OllamaModel = "bge-large",
        api_url: str = "http://local-ollama:11434",
        batch_size: int = 256,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        self.model = model
        self.api_url = (
            api_url or 
            os.environ.get("OLLAMA_API_URL")
        ).rstrip('/')
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._check_connection()

    def _check_connection(self) -> None:
        """Test connection to Ollama service"""
        for attempt in range(self.max_retries):
            try:
                response = requests.get(f"{self.api_url}/api/tags")
                response.raise_for_status()
                return
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    raise ConnectionError(
                        f"Failed to connect to Ollama API at {self.api_url}. "
                        f"Error: {str(e)}"
                    )
                time.sleep(self.retry_delay)

class OllamaEmbeddings:
    def _get_embedding(self, text: str) -> List[float]:
        try:
            text_preview = text[:50] + "..." if len(text) > 50 else text
            ollama_logger.info(f"Embedding text: {text_preview}")
            
            payload = {
                "model": self.model,
                "prompt": text
            }
            ollama_logger.info(f"Request to: {self.api_url}/api/embeddings")
            
            response = requests.post(
                f"{self.api_url}/api/embeddings",
                json=payload
            )
            
            if response.status_code != 200:
                ollama_logger.error(f"Error response: {response.text}")
            else:
                ollama_logger.info("Embedding successful")
                
            response.raise_for_status()
            result = response.json()
            
            if 'embedding' not in result:
                raise ValueError(f"❌ Unexpected response format: {result}")
                
            return result['embedding']
        except Exception as e:
            ollama_logger.error(f"Failed: {str(e)}")
            raise
    
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        res = []
        non_empty_texts = [text for text in texts if text]

        for i in range(0, len(non_empty_texts), self.batch_size):
            batch_texts = non_empty_texts[i : i + self.batch_size]
            batch_embeddings = [self._get_embedding(text) for text in batch_texts]
            res.extend(batch_embeddings)

        # Pad empty texts with zeros
        embedding_size = len(res[0]) if res else 1
        return res + [[0.0] * embedding_size] * (len(texts) - len(non_empty_texts))