"""
Custom similarity algorithms for resume matching
"""
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """
    A class for calculating similarities between job descriptions and resumes
    """
    
    def __init__(self):
        """Initialize the similarity calculator"""
        self.vectorizer = None
        self.job_vector = None
        self.algorithm_times = {
            'cosine': [],
            'sqrtcos': [],
            'isc': []
        }
    
    def create_vectorizer(self, preprocessed_texts):
        """
        Create TF-IDF vectorizer from texts
        
        Args:
            preprocessed_texts: List of preprocessed texts
            
        Returns:
            Fitted vectorizer
        """
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                max_df=0.9
            )
            
            self.vectorizer.fit(preprocessed_texts)
            logger.info(f"Vectorizer created with {len(self.vectorizer.get_feature_names_out())} features")
            return self.vectorizer
        
        except Exception as e:
            logger.error(f"Error creating vectorizer: {str(e)}")
            raise
    
    def vectorize_job_description(self, preprocessed_job_text):
        """
        Vectorize the job description
        
        Args:
            preprocessed_job_text: Preprocessed job description text
            
        Returns:
            Job description vector
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer not initialized. Call create_vectorizer first.")
        
        try:
            self.job_vector = self.vectorizer.transform([preprocessed_job_text])
            logger.info(f"Job description vectorized with shape {self.job_vector.shape}")
            return self.job_vector
        
        except Exception as e:
            logger.error(f"Error vectorizing job description: {str(e)}")
            raise
    
    def vectorize_resumes(self, preprocessed_resume_texts):
        """
        Vectorize resume texts
        
        Args:
            preprocessed_resume_texts: List of preprocessed resume texts
            
        Returns:
            Resume vectors
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer not initialized. Call create_vectorizer first.")
        
        try:
            resume_vectors = self.vectorizer.transform(preprocessed_resume_texts)
            logger.info(f"Vectorized {len(preprocessed_resume_texts)} resumes with shape {resume_vectors.shape}")
            return resume_vectors
        
        except Exception as e:
            logger.error(f"Error vectorizing resumes: {str(e)}")
            raise
    
    def calculate_cosine_similarity(self, resume_vectors):
        """
        Calculate cosine similarity between job description and resumes
        
        Args:
            resume_vectors: Vectorized resume texts
            
        Returns:
            Array of cosine similarity scores
        """
        if self.job_vector is None:
            raise ValueError("Job vector not initialized. Call vectorize_job_description first.")
        
        start_time = time.time()
        try:
            # Calculate cosine similarity
            cosine_similarities = cosine_similarity(self.job_vector, resume_vectors).flatten()
            
            # Record processing time
            proc_time = time.time() - start_time
            self.algorithm_times['cosine'].append(proc_time)
            
            return cosine_similarities
        
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            raise
    
    def calculate_sqrtcos_similarity(self, resume_vectors):
        """
        Calculate square root cosine similarity between job description and resumes
        
        Args:
            resume_vectors: Vectorized resume texts
            
        Returns:
            Array of square root cosine similarity scores
        """
        if self.job_vector is None:
            raise ValueError("Job vector not initialized. Call vectorize_job_description first.")
        
        start_time = time.time()
        try:
            # Convert sparse matrices to dense for element-wise operations
            job_vector_dense = self.job_vector.toarray()
            resume_vectors_dense = resume_vectors.toarray()
            
            # Apply square root to TF-IDF values (element-wise)
            job_vector_sqrt = np.sqrt(job_vector_dense)
            resume_vectors_sqrt = np.sqrt(resume_vectors_dense)
            
            # Calculate cosine similarity using the square root vectors
            sqrtcos_similarities = cosine_similarity(job_vector_sqrt, resume_vectors_sqrt).flatten()
            
            # Record processing time
            proc_time = time.time() - start_time
            self.algorithm_times['sqrtcos'].append(proc_time)
            
            return sqrtcos_similarities
        
        except Exception as e:
            logger.error(f"Error calculating sqrt cosine similarity: {str(e)}")
            raise
    
    def calculate_isc_similarity(self, resume_vectors):
        """
        Calculate Improved Similarity Coefficient (ISC) between job description and resumes.
        ISC is a custom algorithm that combines cosine similarity with word overlap measures.
        
        Args:
            resume_vectors: Vectorized resume texts
            
        Returns:
            Array of ISC similarity scores
        """
        if self.job_vector is None:
            raise ValueError("Job vector not initialized. Call vectorize_job_description first.")
        
        start_time = time.time()
        try:
            # Calculate standard cosine similarity first
            cos_similarities = cosine_similarity(self.job_vector, resume_vectors).flatten()
            
            # Get non-zero elements in the job vector (job description terms)
            job_terms = set(self.job_vector.indices)
            
            # Calculate term overlap for each resume
            isc_similarities = np.zeros(resume_vectors.shape[0])
            
            for i, vec in enumerate(resume_vectors):
                # Get non-zero elements in the resume vector (resume terms)
                resume_terms = set(vec.indices)
                
                # Calculate term overlap coefficients
                overlap_size = len(job_terms.intersection(resume_terms))
                job_unique = len(job_terms)
                
                # Coverage (how much of the job description is covered by the resume)
                coverage = overlap_size / job_unique if job_unique > 0 else 0
                
                # ISC formula: enhance cosine similarity with coverage factor
                isc_similarities[i] = cos_similarities[i] * (0.5 + 0.5 * coverage)
            
            # Record processing time
            proc_time = time.time() - start_time
            self.algorithm_times['isc'].append(proc_time)
            
            return isc_similarities
        
        except Exception as e:
            logger.error(f"Error calculating ISC similarity: {str(e)}")
            raise
    
    def calculate_all_similarities(self, resume_vectors):
        """
        Calculate all similarity measures at once
        
        Args:
            resume_vectors: Vectorized resume texts
            
        Returns:
            Dictionary with arrays of similarity scores
        """
        cosine_scores = self.calculate_cosine_similarity(resume_vectors)
        sqrtcos_scores = self.calculate_sqrtcos_similarity(resume_vectors)
        isc_scores = self.calculate_isc_similarity(resume_vectors)
        
        return {
            'cosine': cosine_scores,
            'sqrtcos': sqrtcos_scores,
            'isc': isc_scores
        }
    
    def get_average_processing_times(self):
        """Get average processing times for algorithms"""
        return {
            'cosine': sum(self.algorithm_times['cosine']) / len(self.algorithm_times['cosine']) if self.algorithm_times['cosine'] else 0,
            'sqrtcos': sum(self.algorithm_times['sqrtcos']) / len(self.algorithm_times['sqrtcos']) if self.algorithm_times['sqrtcos'] else 0,
            'isc': sum(self.algorithm_times['isc']) / len(self.algorithm_times['isc']) if self.algorithm_times['isc'] else 0
        }
    
    def reset_metrics(self):
        """Reset the algorithm processing time metrics"""
        self.algorithm_times = {
            'cosine': [],
            'sqrtcos': [],
            'isc': []
        }

# Create a singleton instance to be used throughout the application
similarity_calculator = SimilarityCalculator()