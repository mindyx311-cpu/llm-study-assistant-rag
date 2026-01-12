import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.2"
VECTOR_DIR = "vectorstore"
