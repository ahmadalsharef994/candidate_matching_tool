"""
Cosine similarity implementation
"""
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logger
logger = logging.getLogger(__name__)

def cosine_sim(X, Y=None):
    """
    Compute cosine similarity between vectors X and Y
    
    Args:
        X: TF-IDF matrix for document 1
        Y: TF-IDF matrix for document 2 (optional)
        
    Returns:
        Cosine similarity score matrix
    """
    try:
        return cosine_similarity(X, Y)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return None
