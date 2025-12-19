import pytest

# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from ingestion.chunker import Chunker


def test_chunker():
    try:
        chunker = Chunker()

        test_pdf = ".test_contract.pdf"
        texts, metadatas, ids = chunker.parse_contracts(test_pdf)
        
        assert texts is not None, "Chunker failed to parse contract PDF."
        assert metadatas is not None, "Chunker failed to extract metadata from contract PDF."
        assert ids is not None, "Chunker failed to generate IDs for contract PDF."

    except Exception as e:
        pytest.fail(f"Chunker test failed: {e}")


