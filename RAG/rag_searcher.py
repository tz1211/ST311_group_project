import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

class RAGSearcher:
    def __init__(self):
        pass 


    def calculate_text_score(self, text, query_terms):
        text = str(text).lower()
        # Count how many query terms appear in the text
        return sum(term in text for term in query_terms)


    def search(self, q_text: str, q_vector: list[float], df_context: pd.DataFrame, question_id: str, top: int = 3):
        # filter df_context by question_id
        df_context_filtered = df_context[df_context["_id"] == question_id]

        # Calculate text match score and cosine similarity in one pass
        query_terms = str(q_text).lower().split()
        query = np.array(q_vector)
        
        # Compute cosine similarity array 
        embeddings_array = np.stack([np.array(eval(x))[0] for x in df_context_filtered["embeddings"]])
        similarities = cosine_similarity(query.reshape(1,-1), embeddings_array)[0]

        # Calculate text match score and cosine similarity 
        scores = pd.DataFrame({
            'text_rank': df_context_filtered['chunk_text'].apply(lambda x: self.calculate_text_score(x, query_terms)),
            'cosine_similarity': similarities,
            'chunk_text': df_context_filtered['chunk_text'].values
        })

        # Normalize scores
        scores['text_rank_norm'] = 1.0 / (1.0 + scores['text_rank'].rank(ascending=False))
        scores['cosine_rank_norm'] = 1.0 / (1.0 + scores['cosine_similarity'].rank(ascending=False))
        
        # Combine scores and sort once
        scores['combined_score'] = scores['text_rank_norm'] + scores['cosine_rank_norm']
        top_results = scores.nlargest(top, 'combined_score')

        # Format results
        results = [f"Source {i+1}: {text}\n" for i, text in enumerate(top_results['chunk_text'])]
        return "\n".join(results)
