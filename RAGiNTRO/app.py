import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embedding functions with the API key and model name
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"  # example model name, adjust as needed
)

# Now `openai_ef` can be used to generate embeddings
# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chromadb_store")

collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=openai_ef
)

