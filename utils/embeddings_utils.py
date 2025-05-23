import logging
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

logging.getLogger("transformers").setLevel(logging.ERROR)

embed_model_instance = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct", trust_remote_code=True)

def compute_text_embedding(text: str, embed_model: str = embed_model_instance): 
    """
    This function takes a text and returns an embedding for the text. 
    """
    return [embed_model.encode(text)["dense_vecs"].tolist()]


def count_tokens(text: str, tokenizer: AutoTokenizer = DEFAULT_TOKENIZER):
    return len(tokenizer.encode(text))