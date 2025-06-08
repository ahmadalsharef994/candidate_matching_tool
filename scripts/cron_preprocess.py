#!/usr/bin/env python3
"""
CRON script to preprocess resumes from raw folder to processed folder
"""
import os
import sys
import logging
from pathlib import Path
import shutil
import time

# Set up project path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Import environment variables
from dotenv import load_dotenv
load_dotenv()

# Import from project
from ml.preprocessing.pdf_parser import process_pdf, batch_process_pdfs
from config.logger import logger

# Define paths
RAW_FOLDER = os.environ.get('RAW_FOLDER', os.path.join(PROJECT_ROOT, 'data', 'raw'))
CV_FOLDER = os.environ.get('CV_FOLDER', os.path.join(PROJECT_ROOT, 'data', 'processed', 'cvs'))

def main():
    """Main function to run preprocessing"""
    logger.info("Starting preprocessing job")
    
    # Create necessary directories
    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs(CV_FOLDER, exist_ok=True)
    
    # Get files from raw folder
    raw_files = [os.path.join(RAW_FOLDER, f) for f in os.listdir(RAW_FOLDER) 
                if os.path.isfile(os.path.join(RAW_FOLDER, f)) and 
                (f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.doc') or f.endswith('.txt'))]
    
    if not raw_files:
        logger.info("No new files to process")
        return
    
    logger.info(f"Found {len(raw_files)} files to process")
    
    # Process files
    processed_results = batch_process_pdfs(raw_files, doc_type="resume")
    
    if not processed_results:
        logger.error("Failed to process any files")
        return
    
    # Move processed files to CV folder
    for result in processed_results:
        source_path = result['original_path']
        filename = os.path.basename(source_path)
        dest_path = os.path.join(CV_FOLDER, filename)
        
        # Copy the file to CV folder
        shutil.copy2(source_path, dest_path)
        logger.info(f"Copied {filename} to {CV_FOLDER}")
        
        # Remove from raw folder
        os.remove(source_path)
        logger.info(f"Removed {filename} from {RAW_FOLDER}")
    
    logger.info(f"Preprocessing complete. Processed {len(processed_results)} files")

if __name__ == "__main__":
    main()