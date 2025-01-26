import pytest
from openparse import DocumentParser
from openparse.config import config

def test_parse_with_ollama_embeddings():
    basic_doc_path = "src/evals/data/full-pdfs/mock-1-page-lease.pdf"
    parser = DocumentParser()
    
    # Test with Ollama embeddings provider
    parsed_doc = parser.parse(
        basic_doc_path,
        embeddings_provider="ollama"
    )
    assert len(parsed_doc.nodes) >= 1
    assert parsed_doc.nodes[0].text.startswith("**MOCK LEASE AGREEMENT**")

def test_parse_with_config_overrides():
    basic_doc_path = "src/evals/data/full-pdfs/mock-1-page-lease.pdf"
    parser = DocumentParser()
    
    # Test with multiple config overrides
    parsed_doc = parser.parse(
        basic_doc_path,
        parse_elements={"images": False, "tables": True},
        embeddings_provider="ollama"
    )
    assert len(parsed_doc.nodes) >= 1
    
    # Verify images were skipped
    assert all(node.variant != "image" for node in parsed_doc.nodes)

@pytest.mark.integration
def test_ollama_connection():
    """Test Ollama API connection - requires running Ollama service"""
    basic_doc_path = "src/evals/data/full-pdfs/mock-1-page-lease.pdf"
    parser = DocumentParser()
    
    try:
        parsed_doc = parser.parse(
            basic_doc_path,
            embeddings_provider="ollama"
        )
        assert len(parsed_doc.nodes) >= 1
    except ConnectionError as e:
        pytest.skip(f"Ollama service not available: {str(e)}")