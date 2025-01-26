from enum import Enum
from typing import Literal, Union, Protocol, List
from .openai import OpenAIEmbeddings, cosine_similarity, EmbeddingModel as OpenAIModel
from .ollama import OllamaEmbeddings, OllamaModel

class EmbeddingsProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class EmbeddingsClient(Protocol):
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        ...

EmbeddingModel = Union[OpenAIModel, OllamaModel]

__all__ = [
    'OpenAIEmbeddings',
    'OllamaEmbeddings', 
    'cosine_similarity',
    'EmbeddingModel',
    'EmbeddingsProvider',
    'EmbeddingsClient'
]