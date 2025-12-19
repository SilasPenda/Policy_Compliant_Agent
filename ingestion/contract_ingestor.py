import os
import sys
from dotenv import load_dotenv

from src.exception import CustomException
from ingestion.chunker import Chunker
from ingestion.embed_upsert import EmbedUpsert

load_dotenv()


class ContractIngestor:
    def __init__(self, url, api_key):
        self.chunker = Chunker()
        self.embed_upsert = EmbedUpsert(url=url, api_key=api_key)

        contract_collection = os.getenv("CONTRACT_COLLECTION_NAME")
        self.contracts_dir = os.path.join(os.getcwd(), contract_collection)

    def run_pipeline(self):
        try:
            texts_list = []
            metadatas_list = []
            ids_list = []

            for root, dirs, files in os.walk(self.contracts_dir):
                for file in files:

                    if file.enswith((".pdf", ".PDF")):
                        pdf_path = os.path.join(root, file)

                        texts, metadatas, ids = self.chunker.parse_contracts(pdf_path)

                        texts_list.extend(texts)
                        metadatas_list.extend(metadatas)
                        ids_list.extend(ids)

            embedings = self.embed_upsert.embedder(texts_list)
            self.embed_upsert.upsert(texts_list, metadatas_list, ids_list, embedings)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    ingestor = ContractIngestor(url=qdrant_url, api_key=qdrant_api_key)
    ingestor.run_pipeline()    