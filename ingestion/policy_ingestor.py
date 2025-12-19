import os
import sys
from dotenv import load_dotenv

from ingestion.chunker import Chunker
from ingestion.embed_upsert import EmbedUpsert
from src.exception import CustomException

load_dotenv()


class PolicyIngestor:
    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        policy_collection = os.getenv("POLICY_COLLECTION_NAME")

        self.policies_dir = os.path.join(os.getcwd(), policy_collection)
        self.chunker = Chunker()
        self.embed_upsert = EmbedUpsert(url=qdrant_url, api_key=qdrant_api_key, collection_prefix=policy_collection)

    def run_pipeline(self):
        try:
            texts_list = []
            metadatas_list = []
            ids_list = []

            for root, dirs, files in os.walk(self.policies_dir):
                for file in files:
                    if file.enswith(".yaml"):
                        yaml_path = os.path.join(root, file)
                        texts, metadatas, ids = self.chunker.parse_policies(yaml_path)

                        texts_list.extend(texts)
                        metadatas_list.extend(metadatas)
                        ids_list.extend(ids)

            embedings = self.embed_upsert.embedder(texts_list)
            self.embed_upsert.upsert(texts_list, metadatas_list, ids_list, embedings)

        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    

    ingestor = PolicyIngestor()
    ingestor.run_pipeline()    

  