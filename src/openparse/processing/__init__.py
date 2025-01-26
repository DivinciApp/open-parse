from typing import Literal, Union
from .basic_transforms import (
    CombineBullets,
    CombineHeadingsWithClosestText,
    CombineNodesSpatially,
    ProcessingStep,
    RemoveFullPageStubs,
    RemoveMetadataElements,
    RemoveNodesBelowNTokens,
    RemoveRepeatedElements,
    RemoveTextInsideTables,
)
from .ingest import (
    BasicIngestionPipeline,
    IngestionPipeline,
    NoOpIngestionPipeline,
    SemanticIngestionPipeline,
)
from .semantic_transforms import CombineNodesSemantically
from openparse.embeddings.openai import OpenAIEmbeddings, EmbeddingModel as OpenAIModel
from openparse.embeddings.ollama import OllamaEmbeddings, OllamaModel
from openparse.embeddings.cloudflare import CloudflareEmbeddings, CloudflareModel

EmbeddingModel = Union[OpenAIModel, OllamaModel]
EmbeddingsProvider = Literal["openai", "ollama", "cloudflare"]

__all__ = [
    "ProcessingStep",
    "RemoveTextInsideTables",
    "RemoveFullPageStubs",
    "RemoveMetadataElements",
    "RemoveRepeatedElements",
    "CombineHeadingsWithClosestText",
    "CombineBullets",
    "CombineNodesSpatially",
    "BasicIngestionPipeline",
    "IngestionPipeline",
    "SemanticIngestionPipeline",
    "NoOpIngestionPipeline",
    "RemoveNodesBelowNTokens",
    "CombineNodesSemantically",
    'OpenAIEmbeddings',
    'OllamaEmbeddings',
    'CloudflareEmbeddings',
    'CloudflareModel',
    'EmbeddingModel',
    'EmbeddingsProvider',
]
