from typing import List, Optional

from openparse.schemas import Node
from openparse.config import config
from openparse.embeddings import (
    EmbeddingsProvider,
    EmbeddingsClient,
    OpenAIEmbeddings,
    OllamaEmbeddings,
    cosine_similarity
)

from .basic_transforms import ProcessingStep
from openparse.config import Config

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
        config: Optional[Config] = None,
        model: Optional[str] = None,
        min_similarity: float = 0.8,
        max_tokens: int = 1000,
        **kwargs
    ):
        self.config = config or Config()
        self.embedding_client: EmbeddingsClient = create_embeddings_client(
            provider=self.config._embeddings_provider,
            model=model,
            **kwargs
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
