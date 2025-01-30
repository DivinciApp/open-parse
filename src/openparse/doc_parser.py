from pathlib import Path
from typing import List, Literal, TypedDict, TypeVar, Union, Optional, Dict

from openparse import consts, tables, text
from openparse._types import NOT_GIVEN, NotGiven
from openparse.pdf import Pdf
from openparse.processing import (
    BasicIngestionPipeline,
    IngestionPipeline,
    NoOpIngestionPipeline,
)
from openparse.schemas import Node, ParsedDocument, TableElement, TextElement

from openparse.schemas import ImageElement
from openparse.processing.markitdown_doc_parser import DocumentParser as MarkItDownParser
from openparse.config import config, Config

IngestionPipelineType = TypeVar("IngestionPipelineType", bound=IngestionPipeline)


class UnitableArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["unitable"]
    min_table_confidence: float
    table_output_format: Literal["html"]


class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["markdown", "html"]


class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["markdown", "html"]


def _table_args_dict_to_model(
    args_dict: Union[TableTransformersArgsDict, PyMuPDFArgsDict],
) -> Union[tables.TableTransformersArgs, tables.PyMuPDFArgs]:
    if args_dict["parsing_algorithm"] == "table-transformers":
        return tables.TableTransformersArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "pymupdf":
        return tables.PyMuPDFArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "unitable":
        return tables.UnitableArgs(**args_dict)
    else:
        raise ValueError(
            f"Unsupported parsing_algorithm: {args_dict['parsing_algorithm']}"
        )


class DocumentParser:
    """
    A parser for extracting elements from PDF documents, including text and tables.

    Attributes:
        processing_pipeline (Optional[IngestionPipelineType]): A subclass of IngestionPipeline to process extracted elements.
        table_args (Optional[Union[TableTransformersArgsDict, PyMuPDFArgsDict]]): Arguments to customize table parsing.
    """

    _verbose: bool = False

    def __init__(
        self,
        *,
        processing_pipeline: Union[IngestionPipeline, NotGiven, None] = NOT_GIVEN,
        # table_args: Union[TableTransformersArgsDict, PyMuPDFArgsDict, NotGiven] = NOT_GIVEN,
        table_args=None,
        use_markitdown: bool = False,
        llm_client: Optional[object] = None,
        verbose: bool = False,
         **kwargs
    ):
        self._verbose = verbose
        
        # Initialize processing pipeline
        self.processing_pipeline: IngestionPipeline
        if processing_pipeline is NOT_GIVEN:
            self.processing_pipeline = BasicIngestionPipeline()
        elif processing_pipeline is None:
            self.processing_pipeline = NoOpIngestionPipeline()
        else:
            self.processing_pipeline = processing_pipeline

        # Set pipeline verbosity
        self.processing_pipeline.verbose = self._verbose
        
        # Initialize parsers and args
        self.table_args = table_args
        self.use_markitdown = use_markitdown
        if use_markitdown:
            self.markitdown_parser = MarkItDownParser(llm_client=llm_client)

    def _process_directory(
        self,
        files: List[Path],
        batch_size: int
    ) -> List[ParsedDocument]:
        """Process directory of files in batches."""
        results = self.markitdown_parser.parse_batch(files, batch_size)
        return [
            ParsedDocument(
                nodes=nodes,
                filename=file_path.name,  # Use file_path from enumerate
                num_pages=1,
                coordinate_system=consts.COORDINATE_SYSTEM,
                table_parsing_kwargs=None,
                **metadata
            )
            for file_path, (nodes, metadata) in zip(files, results)
        ]

    def _process_markitdown(
        self,
        file_path: Path,
        nodes: List[Node],
        metadata: Dict
    ) -> ParsedDocument:
        """Process single file with MarkItDown."""
        if self.processing_pipeline:
            nodes = self.processing_pipeline.run(nodes)

        # Use page_count directly from metadata since it's already set
        num_pages = metadata.get('page_count', 1)

        return ParsedDocument(
            nodes=nodes,
            filename=file_path.name,
            num_pages=num_pages,
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs=None,
            **metadata
        )

    def _process_pdfminer(
        self,
        file: Union[str, Path],
        parse_elements: Optional[Dict[str, bool]],
        embeddings_provider: Optional[str],
        ocr: bool
    ) -> ParsedDocument:
        """Process file with PDFMiner."""
        temp_config = self._update_config(parse_elements, embeddings_provider)
        doc = Pdf(file)
        nodes = self._extract_nodes(doc, ocr, temp_config)
        return ParsedDocument(
            nodes=nodes,
            filename=Path(file).name,
            num_pages=doc.num_pages,
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs=self._get_table_kwargs(),
            **doc.file_metadata
        )

    def _update_config(
        self,
        parse_elements: Optional[Dict[str, bool]],
        embeddings_provider: Optional[str]
    ) -> Config:
        """Update config with overrides."""
        temp_config = config
        if parse_elements:
            temp_config._parse_elements.update(parse_elements)
        if embeddings_provider:
            temp_config._embeddings_provider = embeddings_provider
        return temp_config


    def _extract_nodes(
        self,
        doc: Pdf,
        ocr: bool,
        temp_config: Config
    ) -> List[Node]:
        """Extract and process nodes from document."""
        text_nodes = self._extract_text_nodes(doc, ocr)
        table_nodes = self._extract_table_nodes(doc, temp_config)
        nodes = text_nodes + table_nodes
        return self.processing_pipeline.run(nodes)

    def _extract_text_nodes(self, doc: Pdf, ocr: bool) -> List[Node]:
        """Extract text nodes from document."""
        text_engine: Literal["pdfminer", "pymupdf"] = (
            "pdfminer" if not ocr else "pymupdf"
        )
        text_elems = text.ingest(doc, parsing_method=text_engine)
        return self._elems_to_nodes(text_elems)


    def _extract_table_nodes(
        self,
        doc: Pdf,
        temp_config: Config
    ) -> List[Node]:
        """Extract table nodes if enabled."""
        if not self.table_args or not temp_config._parse_elements.get("tables", True):
            return []
        table_args_obj = _table_args_dict_to_model(self.table_args)
        table_elems = tables.ingest(doc, table_args_obj, verbose=self._verbose)
        return self._elems_to_nodes(table_elems)

    def _get_table_kwargs(self) -> Optional[Dict]:
        """Get table kwargs if table args present."""
        if not hasattr(self, 'table_args_obj'):
            return None
        return self.table_args_obj.model_dump()


    def parse(
        self,
        file: Union[str, Path],
        ocr: bool = False,
        parse_elements: Optional[Dict[str, bool]] = None,
        embeddings_provider: Optional[Literal["openai", "ollama", "cloudflare"]] = None,
        batch_size: int = 1
    ) -> Union[ParsedDocument, List[ParsedDocument]]:
        """Parse document using configured parser."""
        file_path = Path(file)
        
        if self.use_markitdown:
            if file_path.is_dir():
                files = list(file_path.glob("*"))
                return self._process_directory(files, batch_size)
            
            nodes, metadata = self.markitdown_parser.parse(file_path)
            return self._process_markitdown(file_path, nodes, metadata)
            
        return self._process_pdfminer(
            file_path,
            parse_elements,
            embeddings_provider,
            ocr
        )

    @staticmethod
    def _elems_to_nodes(
        elems: Union[List[TextElement], List[TableElement], List[ImageElement]],
    ) -> List[Node]:
        return [
            Node(
                elements=(e,),
            )
            for e in elems
        ]
