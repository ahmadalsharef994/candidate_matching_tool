"""
PDF parsing module for extracting text from resume and job description PDFs
"""
import os
import time
import logging
import fitz  # PyMuPDF
from tika import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
import re
import string
import numpy as np

from config.logger import logger

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyMuPDF
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        # First try with PyMuPDF
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        
        if len(text.strip()) < 100:
            logger.warning(f"Short text from PyMuPDF for {pdf_path}, trying tika parser")
            return pdftotext_converter(pdf_path)
            
        return text
    
    except Exception as e:
        logger.warning(f"PyMuPDF failed for {pdf_path}, trying tika parser: {e}")
        return pdftotext_converter(pdf_path)

def pdftotext_converter(filename):
    """
    Extract text from PDF using Tika parser
    
    Args:
        filename: Path to PDF file
        
    Returns:
        Extracted text or empty string if extraction failed
    """
    try:
        raw = parser.from_file(filename)
        return raw['content'] if raw['content'] else ""
    except Exception as e:
        logger.error(f"Tika extraction failed for {filename}: {e}")
        return ""

# Keep the rest of your functions unchanged
# cleanText, convert_lower_case, remove_stop_words, etc.