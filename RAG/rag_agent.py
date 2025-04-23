import logging 
import pathlib
import pandas as pd
from openai import OpenAI
from openai_messages_token_helper import build_messages 
from FlagEmbedding import BGEM3FlagModel

from .rag_searcher import RAGSearcher
from utils.embeddings_utils import compute_text_embedding

logger = logging.getLogger("rag_agent")

class RAGAgent: 
    def __init__(
            self, 
            chat_client: OpenAI, 
            embed_client: BGEM3FlagModel,  
            chat_model: str, 
            searcher: RAGSearcher,
            max_tokens: int = 128000,
            temperature: float = 0.0,
            ):
        self.chat_client = chat_client
        self.embed_client = embed_client
        self.chat_model = chat_model
        self.searcher = searcher
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # system prompt(s) 
        current_dir = pathlib.Path(__file__).parent
        self.system_prompt = open(current_dir / "prompts/system_prompt.txt").read()

    def generate_response(self, user_query: str, df_context: pd.DataFrame, question_id: str, top: int = 3): 
        logger.info(f"Entering vector search with query text: {user_query}")
        q_vector = compute_text_embedding(
            user_query,
            self.embed_client
        )

        search_results = self.searcher.search(user_query, q_vector, df_context, question_id, top)

        messages = build_messages(
            model=self.chat_model,
            system_prompt=self.system_prompt,
            new_user_content=user_query + "\n\nSources:\n" + search_results,
            max_tokens=self.max_tokens, 
            fallback_to_default=True,
        )

        chat_completion_response = self.chat_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )

        return chat_completion_response.choices[0].message.content