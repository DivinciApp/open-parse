"""
Tests for DocumentParser class in OpenParse.

This test suite verifies:
- Parser initialization
- PDF document processing
- MarkItDown integration
- Configuration handling
- Error cases

Run tests with:
    pytest test_docparser_0_7_2.py -v
    
With coverage:
    pytest test_docparser_0_7_2.py -v --cov=openparse

Requirements:
    pip install pytest pytest-cov
    pip install openparse[test]
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, date

from openparse import DocumentParser, ParsedDocument, Node
from openparse.processing import BasicIngestionPipeline
from openparse.schemas import TextElement, Bbox, NodeVariant

@pytest.fixture
def sample_pdf():
    return Path(__file__).parent / "sample_data" / "pdf-with-image.pdf"

@pytest.fixture
def sample_docx():
    return Path(__file__).parent / "sample_data" / "docx-test-data.docx"

@pytest.fixture
def mock_markitdown():
    """Mock MarkItDown parser for testing."""
    with patch('openparse.processing.markitdown_doc_parser.DocumentParser') as mock:
        mock.return_value.parse.return_value = (
            [Node(elements=(TextElement(
                text="Test content",
                lines=(),
                bbox=Bbox(
                    page=1,
                    page_height=1000,
                    page_width=1000,
                    x0=0, y0=0, x1=100, y1=100
                ),
                variant=NodeVariant.TEXT),))],
            {"creation_date": None}
        )
        yield mock

class TestDocumentParser:
    def test_init_default(self):
        """Test default initialization."""
        parser = DocumentParser()
        assert parser._verbose is False
        assert isinstance(parser.processing_pipeline, BasicIngestionPipeline)
        assert parser.table_args is None
        assert parser.use_markitdown is False

    def test_init_with_markitdown(self, mock_markitdown):
        """Test initialization with MarkItDown."""
        parser = DocumentParser(use_markitdown=True)
        assert parser.use_markitdown is True
        assert hasattr(parser, 'markitdown_parser')

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_setting(self, verbose):
        """Test verbose flag propagation."""
        parser = DocumentParser(verbose=verbose)
        assert parser._verbose == verbose
        assert parser.processing_pipeline.verbose == verbose

    def test_parse_pdf(self, sample_pdf):
        """Test PDF parsing."""
        parser = DocumentParser()
        result = parser.parse(sample_pdf)
        assert isinstance(result, ParsedDocument)
        assert result.filename == sample_pdf.name
        assert len(result.nodes) > 0

    def test_parse_with_markitdown(self, sample_docx):
        """Test non-PDF parsing with MarkItDown."""
        # mock_nodes = [Node(elements=(TextElement(
        #     text="Test content",
        #     lines=(),
        #     bbox=Bbox(
        #         page=1,
        #         page_height=1000,
        #         page_width=1000,
        #         x0=0, y0=0,
        #         x1=1000, y1=1000
        #     ),
        #     variant=NodeVariant.TEXT
        # ),))]
        
        # mock_metadata = {
        #     "creation_date": None,
        #     "last_modified_date": date.today(),
        #     "last_accessed_date": date.today(),
        #     "file_size": 100,
        #     "file_type": ".docx"
        # }

        # with patch('openparse.processing.markitdown_doc_parser.MarkItDown') as mock:
            # mock_instance = mock.return_value
            # mock_instance.convert.return_value.text_content = "Test content"
            
        parser = DocumentParser(use_markitdown=True)
        result = parser.parse(sample_docx)
        
        assert isinstance(result, ParsedDocument)
        assert result.filename == sample_docx.name
        assert len(result.nodes) == 1
        # assert result.nodes[0].text == "Test content"

    @staticmethod
    def create_test_docx():
        """Create valid test DOCX."""
        content = (
            b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00"
            b"[Content_Types].xml"
        )
        return content

    @staticmethod
    def create_test_pdf():
        """Create minimal valid PDF with correct /Pages tree."""
        return (
            b"%PDF-1.7\n"
            b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type/Pages/Count 1/Kids[3 0 R]>>\nendobj\n"
            b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>\nendobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000056 00000 n\n" 
            b"0000000111 00000 n\n"
            b"trailer\n"
            b"<</Size 4/Root 1 0 R>>\n"
            b"startxref\n"
            b"180\n"
            b"%%EOF"
        )

    def test_directory_processing(self, tmp_path, mock_markitdown):
        """Test directory batch processing."""
        # Create test files
        doc1 = tmp_path / "test1.docx" 
        doc2 = tmp_path / "test2.pdf"
        
        doc1.write_bytes(self.create_test_docx())
        doc2.write_bytes(self.create_test_pdf())

        # Configure mock
        mock_markitdown.return_value.parse.return_value = (
            [Node(elements=(TextElement(
                text="Test content",
                lines=(),
                bbox=Bbox(
                    page=1,
                    page_height=1000,
                    page_width=1000,
                    x0=0, y0=0,
                    x1=1000, y1=1000
                ),
                variant=NodeVariant.TEXT
            ),))],
            {
                "creation_date": None,
                "last_modified_date": date.today(),
                "last_accessed_date": date.today(),
                "file_size": 100,
                "file_type": ".docx"
            }
        )

        parser = DocumentParser(use_markitdown=True)
        results = parser.parse(tmp_path, batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ParsedDocument) for r in results)

    def test_config_update(self):
        """Test configuration updates."""
        parser = DocumentParser()
        config = parser._update_config(
            parse_elements={"tables": False},
            embeddings_provider="cloudflare"
        )
        assert config._parse_elements["tables"] is False
        assert config._embeddings_provider == "cloudflare"

    def test_error_handling(self, tmp_path):
        """Test error handling for invalid files."""
        non_existent = tmp_path / "not_exists.pdf"
        parser = DocumentParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse(non_existent)