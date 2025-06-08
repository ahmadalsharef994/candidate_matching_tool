"""
Initialization file for similarity module
"""
from ml.similarity.cosine_similarity import cosine_sim
from ml.similarity.sqrtcos_similarity import sqrtcos_sim
from ml.similarity.isc_similarity import isc_sim

__all__ = ['cosine_sim', 'sqrtcos_sim', 'isc_sim']
