from abc import ABC
from typing import List, Optional

from openparse import consts
from openparse.processing.basic_transforms import (
    CombineBullets,
    CombineHeadingsWithClosestText,
    CombineNodesSpatially,
    CombineSlicedImages,
    ProcessingStep,
    RemoveFullPageStubs,
    RemoveMetadataElements,
    RemoveNodesBelowNTokens,
    RemoveRepeatedElements,
    RemoveTextInsideTables,
)
from openparse.processing.semantic_transforms import (
    CombineNodesSemantically,
)
from openparse.schemas import Node
from openparse.config import Config


class IngestionPipeline(ABC):
    """
    A pipeline for ingesting and processing Nodes.

    Attributes:
        transformations (List[ProcessingStep]): A list of transforms to process the extracted elements.
        verbose (Optional[bool]): Whether to print out processing steps.
    """

    transformations: List[ProcessingStep]
    verbose: Optional[bool] = False

    def run(self, nodes: List[Node]) -> List[Node]:
        nodes = sorted(nodes)
        for transform_func in self.transformations:
            if self.verbose:
                print("Processing with", transform_func.__class__.__name__)
            nodes = transform_func.process(sorted(nodes))

        return nodes

    def append_transform(self, transform: ProcessingStep) -> None:
        """
        Add a transform to the pipeline.

        Args:
            transform (ProcessingStep): The transform to add.
        """
        self.transformations.append(transform)


class NoOpIngestionPipeline(IngestionPipeline):
    """
    A no-operation (no-op) pipeline for cases where no processing should be performed.
    """

    def __init__(self):
        self.transformations = []


class BasicIngestionPipeline(IngestionPipeline):
    """
    A basic pipeline for ingesting and processing Nodes.
    """

    def __init__(self):
        self.transformations = [
            RemoveTextInsideTables(),
            CombineSlicedImages(),
            RemoveFullPageStubs(max_area_pct=0.35),
            # mostly aimed at combining bullets and weird formatting
            CombineNodesSpatially(
                x_error_margin=10, y_error_margin=4, criteria="both_small"
            ),
            CombineHeadingsWithClosestText(),
            CombineBullets(),
            CombineNodesSpatially(
                x_error_margin=0, y_error_margin=10, criteria="both_small"
            ),
            RemoveMetadataElements(),
            CombineNodesSpatially(criteria="either_stub"),
            RemoveRepeatedElements(threshold=2),
            # # tried everything to combine, remove stubs that are still left
            RemoveNodesBelowNTokens(min_tokens=50),
            # # combines bullets split across pages
            # # (previously page metdata would have prevented this)
            CombineBullets(),
        ]


class SemanticIngestionPipeline(IngestionPipeline):
    """
    A semantic pipeline for ingesting and processing Nodes.
    """

    def __init__(
        self,
        min_tokens: int = consts.TOKENIZATION_LOWER_LIMIT,
        max_tokens: int = consts.TOKENIZATION_UPPER_LIMIT,
        embeddings_provider: str = "ollama",
        model: Optional[str] = None,
        **kwargs
    ) -> None:
        super().__init__()
        temp_config = Config()
        temp_config._embeddings_provider = embeddings_provider

        embedding_kwargs = {}
        if embeddings_provider == "openai":
            embedding_kwargs['api_key'] = kwargs.get('openai_api_key')

        self.transformations = [
            RemoveTextInsideTables(),
            CombineSlicedImages(),
            RemoveFullPageStubs(max_area_pct=0.35),
            # mostly aimed at combining bullets and weird formatting
            CombineNodesSpatially(
                x_error_margin=10,
                y_error_margin=2,
                criteria="both_small",
            ),
            CombineHeadingsWithClosestText(),
            CombineBullets(),
            RemoveMetadataElements(),
            RemoveRepeatedElements(threshold=2),
            RemoveNodesBelowNTokens(min_tokens=10),
            CombineBullets(),
            CombineNodesSemantically(
                config=temp_config,
                model=model,
                min_similarity=0.6,
                max_tokens=max_tokens // 2,
                **embedding_kwargs
            ),
            RemoveNodesBelowNTokens(min_tokens=min_tokens),
        ]
