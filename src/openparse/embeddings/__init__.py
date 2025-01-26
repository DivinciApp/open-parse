from enum import Enum
from typing import Union, Protocol, List
from .openai import OpenAIEmbeddings, cosine_similarity, EmbeddingModel as OpenAIModel
from .ollama import OllamaEmbeddings, OllamaModel
from .cloudflare import CloudflareEmbeddings, CloudflareModel

class EmbeddingsProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    CLOUDFLARE = "cloudflare"

class EmbeddingsClient(Protocol):
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        ...

EmbeddingModel = Union[
    OpenAIModel, OllamaModel, CloudflareModel
]

__all__ = [
    'OpenAIEmbeddings',
    'OllamaEmbeddings', 
    'cosine_similarity',
    'EmbeddingModel',
    'EmbeddingsProvider',
    'EmbeddingsClient',
    'CloudflareEmbeddings',
    'CloudflareModel',
]