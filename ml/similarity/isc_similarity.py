"""
Improved Square root Cosine (ISC) similarity implementation
"""
from sklearn.metrics import pairwise
import logging

# Configure logger
logger = logging.getLogger(__name__)

def isc_sim(X, Y=None, dense_output=True):
    """
    Compute improved square root cosine similarity between samples in X and Y
    
    This implementation uses L1 normalization instead of L2 used in sqrtcos_similarity
    
    Args:
        X: TF-IDF matrix for document 1
        Y: TF-IDF matrix for document 2 (optional)
        dense_output: Whether to return dense output
        
    Returns:
        ISC similarity score matrix
    """
    try:
        X, Y = pairwise.check_pairwise_arrays(X, Y)

        # Use L1 normalization for ISC
        X_normalized = pairwise.normalize(X, copy=True, norm='l1')
        if X is Y:
            Y_normalized = X_normalized
        else:
            Y_normalized = pairwise.normalize(Y, copy=True, norm='l1')

        # Calculate dot product
        K = pairwise.safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

        return K
    except Exception as e:
        logger.error(f"Error calculating ISC similarity: {e}")
        return None
