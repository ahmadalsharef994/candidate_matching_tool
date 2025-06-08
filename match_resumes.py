#!/usr/bin/env python3
"""
Production-ready Resume Matcher
- Works directly with files in the CVs folder
- Matches against job_description.pdf file
- Outputs results to CSV
"""
import os
import sys
import time
import pandas as pd
import pdftotext
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_matcher.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CV_FOLDER = os.path.join(BASE_DIR, 'CVs')
JOB_DESC_PATH = os.path.join(BASE_DIR, 'job_description.pdf')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'results')
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, 'matching_results.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Text processing functions
def pdftotext_converter(pdf_path):
    """Extract text from PDF using pdftotext"""
    try:
        with open(pdf_path, "rb") as f:
            pdf = pdftotext.PDF(f)
            return "\n\n".join(pdf)
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def clean_text(text):
    """Basic text cleaning operations"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces between words
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """Process text for similarity comparison"""
    cleaned_text = clean_text(text)
    # Additional preprocessing can be added here
    return cleaned_text

# Similarity measurement functions
def calculate_cosine_similarity(job_vector, resume_vector):
    """Calculate cosine similarity between job and resume vectors"""
    return cosine_similarity(job_vector, resume_vector)[0][0]

def calculate_sqrtcos_similarity(job_vector, resume_vector):
    """Calculate square root cosine similarity (custom algorithm)"""
    # Convert to dense arrays
    job_array = job_vector.toarray()
    resume_array = resume_vector.toarray()
    
    # Apply element-wise square root
    job_sqrt = np.sqrt(job_array)
    resume_sqrt = np.sqrt(resume_array)
    
    # Calculate cosine similarity on the transformed vectors
    return cosine_similarity(job_sqrt, resume_sqrt)[0][0]

def calculate_isc_similarity(job_vector, resume_vector, vectorizer):
    """
    Calculate Improved Similarity Coefficient (ISC)
    This custom algorithm enhances cosine similarity by including term coverage
    """
    # Base cosine similarity
    cos_sim = calculate_cosine_similarity(job_vector, resume_vector)
    
    # Get job terms (non-zero features)
    job_terms = set(job_vector.indices)
    resume_terms = set(resume_vector.indices)
    
    # Calculate term overlap
    overlap_size = len(job_terms.intersection(resume_terms))
    job_unique = len(job_terms)
    
    # Calculate term coverage - how much of the job description is covered by the resume
    coverage = overlap_size / job_unique if job_unique > 0 else 0
    
    # ISC formula: enhance cosine similarity with the coverage factor
    isc = cos_sim * (0.5 + 0.5 * coverage)
    
    return isc

def main():
    """Main function to run the resume matcher"""
    start_time = time.time()
    logger.info("Starting resume matching process")
    
    # Check if job description file exists
    if not os.path.exists(JOB_DESC_PATH):
        logger.error(f"Job description file not found: {JOB_DESC_PATH}")
        return
    
    # Extract and preprocess job description
    logger.info(f"Processing job description: {JOB_DESC_PATH}")
    job_text = pdftotext_converter(JOB_DESC_PATH)
    if not job_text:
        logger.error("Failed to extract text from job description")
        return
    
    job_processed = preprocess_text(job_text)
    
    # Get CV files
    cv_files = [f for f in os.listdir(CV_FOLDER) 
               if os.path.isfile(os.path.join(CV_FOLDER, f)) and 
               (f.endswith('.pdf') or f.endswith('.docx'))]
    
    if not cv_files:
        logger.error(f"No CV files found in {CV_FOLDER}")
        return
    
    logger.info(f"Found {len(cv_files)} CV files")
    
    # Process resumes
    resume_texts = []
    resume_names = []
    
    for cv_file in cv_files:
        cv_path = os.path.join(CV_FOLDER, cv_file)
        logger.info(f"Processing CV: {cv_file}")
        
        if cv_file.endswith('.pdf'):
            text = pdftotext_converter(cv_path)
        else:
            logger.warning(f"Skipping non-PDF file: {cv_file}")
            continue
        
        if not text:
            logger.warning(f"Could not extract text from {cv_file}")
            continue
        
        processed_text = preprocess_text(text)
        resume_texts.append(processed_text)
        resume_names.append(cv_file)
    
    if not resume_texts:
        logger.error("No resume texts were successfully extracted")
        return
    
    logger.info(f"Successfully processed {len(resume_texts)} CVs")
    
    # Create vectorizer
    all_texts = [job_processed] + resume_texts
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.9
    )
    
    # Fit and transform all texts
    all_vectors = vectorizer.fit_transform(all_texts)
    
    # Split job and resume vectors
    job_vector = all_vectors[0:1]
    resume_vectors = all_vectors[1:]
    
    # Calculate similarity scores for each resume
    results = []
    
    for i, resume_name in enumerate(resume_names):
        resume_vector = resume_vectors[i:i+1]
        
        # Calculate similarity scores using different algorithms
        cosine_score = calculate_cosine_similarity(job_vector, resume_vector)
        sqrtcos_score = calculate_sqrtcos_similarity(job_vector, resume_vector)
        isc_score = calculate_isc_similarity(job_vector, resume_vector, vectorizer)
        
        results.append({
            'Filename': resume_name,
            'Cosine_Score': cosine_score,
            'SqrtCos_Score': sqrtcos_score,
            'ISC_Score': isc_score
        })
    
    # Create DataFrame and sort by ISC score
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='ISC_Score', ascending=False)
    
    # Save results to CSV
    df_results.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")
    
    # Print top 10 results
    print("\nTop 10 matching candidates:")
    print(df_results.head(10)[['Filename', 'ISC_Score']].to_string(index=False))
    
    total_time = time.time() - start_time
    logger.info(f"Matching process completed in {total_time:.2f} seconds")
    
    return df_results

if __name__ == "__main__":
    main()
