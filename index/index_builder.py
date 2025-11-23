from config import get_vector_store
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import pandas as pd
import os

# Load index
def load_rag_index(collection_name:str, embed_model):

    vector_store = get_vector_store(collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage_context)
    return index

def build_index(csv_path:str, collection_name:str, embed_model):
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    
    documents = []
    for _, row in df.iterrows():
        # Create text content
        text = (
            f"Món: {row.get('dish_name', '')}. "
            f"Mô tả: {row.get('description', '')}. "
            f"Thành phần: {row.get('ingredients', '')}. "
            f"Loại: {row.get('dish_type', '')}."
        )
        
        # Create metadata
        metadata = {
            "calories": row.get('calories'),
            "protein": row.get('protein'),
            "fat": row.get('fat'),
            "fiber": row.get('fiber'),
            "sugar": row.get('sugar'),
            "image_link": row.get('image_link')
        }
        
        # Create Document
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)

    # Text splitter
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    # chunking & indexing
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[text_splitter]
    )
    
    return index