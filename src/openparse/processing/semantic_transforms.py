from enum import Enum
from typing import Protocol, List

from openparse.schemas import Node
from openparse.config import config

from .basic_transforms import ProcessingStep
from openparse.embeddings.openai import OpenAIEmbeddings, cosine_similarity
from openparse.embeddings.ollama import OllamaEmbeddings

class EmbeddingsProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class EmbeddingsClient(Protocol):
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        ...

def create_embeddings_client(
    provider: str,
    **kwargs
) -> EmbeddingsClient:
    if provider == EmbeddingsProvider.OPENAI:
        return OpenAIEmbeddings(**kwargs)
    elif provider == EmbeddingsProvider.OLLAMA:
        return OllamaEmbeddings(**kwargs)
    raise ValueError(f"❌ Unknown embeddings provider: {provider}")

class CombineNodesSemantically(ProcessingStep):
    """
    Combines nodes that are semantically related using configurable embeddings.
    """

    def __init__(
        self,
        min_similarity: float = 0.8,
        max_tokens: int = 1000,
        **embedding_kwargs
    ):
        self.embedding_client: EmbeddingsClient = create_embeddings_client(
            provider=config._embeddings_provider,
            **embedding_kwargs
        )
        self.min_similarity = min_similarity
        self.max_tokens = max_tokens

    def process(self, nodes: List[Node]) -> List[Node]:
        modified = True
        while modified:
            modified = False
            nodes = sorted(nodes)

            embeddings = self.embedding_client.embed_many([node.text for node in nodes])
            i = 0

            while i < len(nodes) - 1:
                current_embedding = embeddings[i]
                next_embedding = embeddings[i + 1]
                similarity = cosine_similarity(current_embedding, next_embedding)
                is_within_token_limit = (
                    nodes[i].tokens + nodes[i + 1].tokens <= self.max_tokens
                )

                if similarity >= self.min_similarity and is_within_token_limit:
                    nodes[i] = nodes[i] + nodes[i + 1]
                    del nodes[i + 1]
                    del embeddings[i + 1]

                    modified = True
                    continue
                i += 1

        return nodes

    def _get_node_similarities(self, nodes: List[Node]) -> List[float]:
        """
        Get the similarity of each node with the node that precedes it
        """
        embeddings = self.embedding_client.embed_many([node.text for node in nodes])

        similarities = []
        for i in range(1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i - 1], embeddings[i]))

        return [0] + similarities
