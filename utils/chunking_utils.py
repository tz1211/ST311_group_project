import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core.node_parser import SentenceSplitter

from .embeddings_utils import compute_text_embedding

load_dotenv()

logger = logging.getLogger(__name__)

default_chunk_size = int(os.getenv("CHUNK_SIZE", 256))
default_chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 20))
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def adaptive_semantic_chunking(text, similarity_threshold=0.8, chunk_size=default_chunk_size): 
    """
    This function takes a long body of text and returns a list of chunks based on the adaptive semantic chunking algorithm. 
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    text_chunks = splitter.split_text(text)
    logger.info(f"text_chunks length {len(text_chunks)}")
    embeddings_list = [compute_text_embedding(chunk) for chunk in text_chunks] 

    output_chunks_list = []
    output_embeddings_list = []
    cosine_similarity_list = []

    prev_chunk = text_chunks[0]
    prev_embedding = embeddings_list[0]
    for i in range(1, len(embeddings_list)): 
        cosine_similarity_list.append(cosine_similarity(prev_embedding, embeddings_list[i])[0][0])
        # Appending the chunk to output if its cosine similarity with the previous chunk is below the specified threshold
        if cosine_similarity(prev_embedding, embeddings_list[i])[0][0] < similarity_threshold: 
            output_chunks_list.append(prev_chunk)
            output_embeddings_list.append(prev_embedding) 
            prev_chunk = text_chunks[i]
            prev_embedding = embeddings_list[i]
        else: 
            # Merging the 2 chunks if their cosine similarity is above the specified threshold
            prev_chunk = " ".join([prev_chunk, text_chunks[i]])
            prev_embedding = compute_text_embedding(prev_chunk)
    
    # Appending the last chunk to output
    output_chunks_list.append(prev_chunk)
    output_embeddings_list.append(prev_embedding)
    logger.info(f"output_chunks_list length {len(output_chunks_list)}")

    return output_chunks_list, output_embeddings_list, cosine_similarity_list



def simple_chunking(text, chunk_size=default_chunk_size, chunk_overlap=default_chunk_overlap): 
    """
    This function takes a text and a chunk size and returns a list of chunks strictly adhering to the chunk size (without regards for sentence or paragraph boundaries). 
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap  # step forward, retaining overlap
    
    return chunks



def sentence_chunking(text, chunk_size=default_chunk_size, chunk_overlap=default_chunk_overlap): 
    """
    This function takes a text and a chunk size and returns a list of chunks, respecting sentence and paragraph boundaries. 
    """
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_chunks = splitter.split_text(text)
    embeddings_list = [compute_text_embedding(chunk) for chunk in text_chunks] 

    return text_chunks, embeddings_list