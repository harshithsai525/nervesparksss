import tempfile
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False}
    )

    temp_dir = tempfile.mkdtemp()  # In-memory DuckDB safe temp dir
    return Chroma(
        persist_directory=temp_dir,
        embedding_function=embedding,
        collection_name="legal_temp",
        client_settings={
            "chroma_db_impl": "duckdb+parquet",
            "persist_directory": temp_dir
        }
    )
