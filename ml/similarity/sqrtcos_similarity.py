"""
Square root cosine similarity implementation
"""
from sklearn.metrics import pairwise
import numpy as np
import logging

# Configure logger
logger = logging.getLogger(__name__)

def sqrtcos_sim(X, Y=None, dense_output=True):
    """
    Compute square root cosine similarity between samples in X and Y
    
    Args:
        X: TF-IDF matrix for document 1
        Y: TF-IDF matrix for document 2 (optional)
        dense_output: Whether to return dense output
        
    Returns:
        Square root cosine similarity score matrix
    """
    try:
        # Apply square root transformation to X
        X = np.sqrt(X)
        
        # Check if Y is provided and transform it too
        if Y is not None:
            Y = np.sqrt(Y)
        
        X, Y = pairwise.check_pairwise_arrays(X, Y)

        # Normalize both matrices
        X_normalized = pairwise.normalize(X, copy=True, norm='l2')
        if X is Y:
            Y_normalized = X_normalized
        else:
            Y_normalized = pairwise.normalize(Y, copy=True, norm='l2')

        # Calculate dot product
        K = pairwise.safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

        return K
    except Exception as e:
        logger.error(f"Error calculating sqrtcos similarity: {e}")
        return None
