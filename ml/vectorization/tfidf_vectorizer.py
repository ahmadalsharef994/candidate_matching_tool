"""
Text vectorization module using TF-IDF
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logger
logger = logging.getLogger(__name__)

class TextVectorizer:
    """
    Vectorize text data using TF-IDF with configurable parameters
    """
    
    def __init__(self, max_features=None, ngram_range=(1, 1), max_df=1.0, min_df=1):
        """
        Initialize the vectorizer with custom parameters
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: The lower and upper boundary of the range of n-values for n-grams
            max_df: When building the vocabulary, ignore terms with a document frequency higher than max_df
            min_df: When building the vocabulary, ignore terms with a document frequency lower than min_df
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.vectorizer = None
        
    def get_vectorizer(self):
        """
        Create and return a TfidfVectorizer with the specified parameters
        
        Returns:
            Configured TF-IDF vectorizer
        """
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                max_df=self.max_df,
                min_df=self.min_df
            )
        return self.vectorizer
    
    def fit_transform(self, documents):
        """
        Fit the vectorizer on documents and transform them to TF-IDF vectors
        
        Args:
            documents: List of text documents
            
        Returns:
            TF-IDF matrix for the documents
        """
        try:
            vectorizer = self.get_vectorizer()
            return vectorizer.fit_transform(documents)
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            return None
    
    def transform(self, documents):
        """
        Transform documents to TF-IDF vectors using the fitted vectorizer
        
        Args:
            documents: List of text documents
            
        Returns:
            TF-IDF matrix for the documents
        """
        try:
            if self.vectorizer is None:
                logger.error("Vectorizer not fitted yet. Call fit_transform first.")
                return None
                
            return self.vectorizer.transform(documents)
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            return None
            
    def get_feature_names(self):
        """
        Get feature names (vocabulary) from the vectorizer
        
        Returns:
            List of feature names
        """
        if self.vectorizer is None:
            logger.error("Vectorizer not fitted yet. Call fit_transform first.")
            return []
            
        return self.vectorizer.get_feature_names_out()
