import os
import sys
import yaml
from pypdf import PdfReader
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.exception import CustomException
from src.logger import logging
from src.utils import get_device, read_yaml, read_pdf



class Chunker:
    def __init__(self):
        root_dir = os.getcwd()
        self.contracts_dir = os.path.join(root_dir, "contracts")
        self.policies_dir = os.path.join(root_dir, "policies")
    
    def parse_contracts(self, pdf_path):
        try:
            texts = []
            metadatas = []
            ids = []

            reader_pages = read_pdf(pdf_path)
            for i, page in enumerate(reader_pages):
                text = page.extract_text()
                if text:
                    chunks = self.splitter.split_text(text)
                    for idx, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadatas.append({"page": i, "source": os.path.basename(pdf_path)})
                        ids.append(f"{os.path.basename(pdf_path)}_page{i}_chunk{idx}")

            return texts, metadatas, ids
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def parse_policies(self, yaml_path):
        try:
            texts = []
            metadatas = []
            ids = []

            data = read_yaml(yaml_path)
            
            # Iterate through all top-level keys (policy categories)
            for policy_category, rules in data.items():
                for i, rule in enumerate(rules):
                    content = rule.get("content")
                    metadata = rule.get("metadata", {}).copy()

                    # Add document-level metadata
                    metadata.update({
                        "page": i,
                        "source": os.path.basename(yaml_path)
                    })
                    
                    texts.append(content)
                    metadatas.append(metadata)
                    ids.append(f"{policy_category}_{i}")

            return texts, metadatas, ids
        
        except Exception as e:
            raise CustomException(e, sys)