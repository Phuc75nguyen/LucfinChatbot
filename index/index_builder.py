from config import get_vector_store 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# Load index
def load_rag_index(collection_name:str, embed_model):

    vector_store = get_vector_store(collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model, storage_context=storage_context)
    return index

def build_index(path:str, collection_name:str, embed_model):

    # load documents & add doc_id metadata
    documents = SimpleDirectoryReader("./data").load_data()

    for i, doc in enumerate(documents):
        file_name = doc.metadata.get("file_name")
        # Assign a simple doc_id based on enumeration
        doc.metadata["doc_id"] = i
    
    # Text splitter
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    # chunking & indexing
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(documents,
                            storage_context=storage_context,
                            embed_model=embed_model,
                            text_splitter = text_splitter
                            )
    
    return index