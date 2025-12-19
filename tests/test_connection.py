import os
import sys
import pytest
from dotenv import load_dotenv

# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.utils import get_llm, db_client_connect

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
contract_collection = os.getenv("CONTRACT_COLLECTION_NAME")
policy_collection = os.getenv("POLUCY_COLLECTION_NAME")


def test_llm_connection(llm_type, model_name):
    """
    Test if the LLM can be instantiated and respond.
    """
    try:
        llm = get_llm(llm_type, model_name=model_name)
        response = llm.predict("Hello, how are you?")
        assert response is not None, "LLM did not return a response."
    except Exception as e:
        pytest.fail(f"Failed to connect to LLM '{llm_type}': {e}")

def test_db_connection():
    """
    Test the db_client_connect function.
    """
    try:
        contract_client = db_client_connect(contract_collection)
        policy_client = db_client_connect(policy_collection)
        assert contract_client is not None, "Failed to connect to the contract collection client."
        assert policy_client is not None, "Failed to connect to the policy collection client."

    except Exception as e:
        pytest.fail(f"Failed to connect to the database client: {e}")