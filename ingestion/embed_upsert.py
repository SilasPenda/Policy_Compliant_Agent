import os
import re
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.exception import CustomException
from src.logger import logging
from src.utils import get_embedding_model

load_dotenv()


class EmbedUpsert:
    def __init__(self, url, api_key, collection_prefix):
        self.pdf_dir = os.path.join(os.getcwd(), collection_prefix)

        # Load sentence transformer model
        self.model = get_embedding_model()

        # Qdrant setup
        self.client = QdrantClient(
            url=url,
            api_key=api_key
        )
        self.collection_name = self.get_next_collection_name(self.client, collection_prefix)

    def get_latest_collection_version(self, client, base_name: str) -> int:
        """
        Returns the highest version number for collections named base_name_vN.
        If none exist, returns 0.
        """
        pattern = re.compile(rf"^{base_name}_v(\d+)$")

        versions = []

        collections = client.get_collections().collections
        for c in collections:
            match = pattern.match(c.name)
            if match:
                versions.append(int(match.group(1)))

        return max(versions) if versions else 0
    
    def get_next_collection_name(self, client, base_name: str) -> str:
        latest_version = self.get_latest_collection_version(client, base_name)
        next_version = latest_version + 1
        return f"{base_name}_v{next_version}"

    def embedder(self,texts):
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
            return embeddings
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def upsert(self, texts, metadatas, ids, embeddings):
        try:
            vector_size = embeddings.shape[1]

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

            points = [
                qmodels.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload={
                        "text": texts[i],
                        "metadata": {
                            **metadatas[i],
                            "collection_version": self.collection_name,
                            "active": True,
                        },
                    },
                )
                for i in range(len(texts))
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logging.info(
                f"Upserted {len(texts)} points into {self.collection_name}"
            )

        except Exception as e:
            raise CustomException(e, sys)

            




    
    def upsert(self, texts, metadatas, ids, embeddings):
        try:
            vector_size = embeddings.shape[1]
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE)
            )

            points = [
                qmodels.PointStruct(
                    id=ids[i],
                    vector=embeddings[i],
                    payload={
                        "text": texts[i],
                        "metadata": metadatas[i]
                    }
                )
                for i in range(len(texts))
            ]

            self.client.upsert(collection_name=self.collection_name, points=points)

            logging.info(f"Data ingestion completed successfully. Added {len(texts)} complaints to collection {self.collection_name}")    
            
        except Exception as e:
            raise CustomException(e, sys)


