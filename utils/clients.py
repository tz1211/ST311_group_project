import os 
from dotenv import load_dotenv

from openai import OpenAI
from FlagEmbedding import BGEM3FlagModel

load_dotenv(override=True)

def create_chat_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("MODEL_ENDPOINT"),
        api_key=os.getenv("API_KEY")
    )

def create_embed_client() -> BGEM3FlagModel:
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)