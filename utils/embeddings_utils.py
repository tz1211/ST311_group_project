import logging
from FlagEmbedding import BGEM3FlagModel

logging.getLogger("transformers").setLevel(logging.ERROR)

embed_model_instance = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

def compute_text_embedding(text: str, embed_model: str = embed_model_instance): 
    """
    This function takes a text and returns an embedding for the text. 
    """
    return [embed_model.encode(text)["dense_vecs"].tolist()]
