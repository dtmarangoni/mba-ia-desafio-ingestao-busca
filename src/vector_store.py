from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from pydantic import SecretStr

from config import load_get_config

config = load_get_config()


def get_vector_store() -> PGVector:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        google_api_key=SecretStr(config.api_key)
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=config.pgvector_collection,
        connection=config.pgvector_url,
        use_jsonb=True
    )
