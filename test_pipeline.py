from tests.test_ingestion import test_chunking, test_ingestion
from tests.test_connection import test_llm_connection, test_db_connection


def run_tests():
    """
    Run all tests for the Policy Compliant Agent project.
    """

    # Test LLM Connections
    test_llm_connection()

    # Test Database Connection
    test_db_connection()

    # Test Chunker
    texts, metadatas, ids = test_chunking()

    # # Test Ingestion
    # test_ingestion(texts, metadatas, ids)



if __name__ == "__main__":
    run_tests()