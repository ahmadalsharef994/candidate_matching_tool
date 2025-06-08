"""
Main application file for the Candidate Matching Tool
Simplified version with straightforward interface
"""
import os
import time
import pandas as pd
import subprocess
from flask import Flask, request, render_template, jsonify, send_file
from pathlib import Path
from werkzeug.utils import secure_filename

# Import ML components
from ml.preprocessing.pdf_parser import process_pdf, batch_process_pdfs
from ml.similarity.similarity_engine import similarity_calculator

# Import configuration
from config.config import active_config
from config.logger import logger

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(active_config)

# Define paths
CV_FOLDER = os.environ.get('CV_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'cvs'))
JOB_DESC_PATH = os.environ.get('JOB_DESC_PATH', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'job_description.pdf'))
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed'))
RAW_FOLDER = os.environ.get('RAW_FOLDER', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw'))
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_first_request
def initialize_app():
    """Initialize application dependencies before first request"""
    logger.info("Initializing application...")
    
    # Create necessary directories
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(CV_FOLDER, exist_ok=True)
    os.makedirs(RAW_FOLDER, exist_ok=True)
    
    # Run CRON job on startup to process any raw files
    try:
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts', 'preprocess_raw.py')
        subprocess.run(['python', script_path])
    except Exception as e:
        logger.error(f"Error running preprocessing on startup: {e}")
    
    # Check if CV folder exists and count files
    if os.path.exists(CV_FOLDER):
        cv_count = len([f for f in os.listdir(CV_FOLDER) 
                      if os.path.isfile(os.path.join(CV_FOLDER, f)) and 
                      (f.endswith('.pdf') or f.endswith('.docx'))])
        logger.info(f"Found {cv_count} CVs in {CV_FOLDER}")
    
    # Check if job description file exists
    if os.path.exists(JOB_DESC_PATH):
        logger.info(f"Found job description file: {JOB_DESC_PATH}")
    
    logger.info("Application initialized successfully")

@app.route('/')
def index():
    """Render the home page with stats about available CVs"""
    try:
        # Count available CVs
        cv_files = [f for f in os.listdir(CV_FOLDER) 
                   if os.path.isfile(os.path.join(CV_FOLDER, f)) and 
                   (f.endswith('.pdf') or f.endswith('.docx'))]
        
        cv_count = len(cv_files)
        
        # Check if job description exists
        job_desc_exists = os.path.exists(JOB_DESC_PATH)
        
        # Check if results exist
        results_path = os.path.join(RESULTS_FOLDER, 'matching_results.csv')
        has_results = os.path.exists(results_path)
        
        return render_template('index.html', 
                              cv_count=cv_count,
                              job_desc_exists=job_desc_exists,
                              has_results=has_results)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return render_template('index.html', error=str(e))

@app.route('/upload/job', methods=['POST'])
def upload_job():
    """Handle job description file or text upload"""
    # Handle text input
    if 'jobText' in request.form and request.form['jobText'].strip():
        job_text = request.form['jobText'].strip()
        with open(JOB_DESC_PATH, 'w') as f:
            f.write(job_text)
        return jsonify({'status': 'success', 'message': 'Job description text saved'})
    
    # Handle file upload
    if 'jobFile' in request.files and request.files['jobFile'].filename:
        file = request.files['jobFile']
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = JOB_DESC_PATH
            file.save(filepath)
            return jsonify({'status': 'success', 'message': 'Job description file uploaded'})
    
    return jsonify({'status': 'error', 'message': 'No valid job description provided'})

@app.route('/match', methods=['POST'])
def match_resumes():
    """Match CVs against job description"""
    try:
        logger.info("Starting matching process")
        start_time = time.time()
        
        # Process job description from file
        job_result = process_pdf(JOB_DESC_PATH, doc_type="job_description")
        if not job_result:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process job description file.'
            }), 500
        
        # Get job description preprocessed text
        preprocessed_job = job_result['preprocessed_text']
        
        # Get all CV files
        cv_files = [os.path.join(CV_FOLDER, f) for f in os.listdir(CV_FOLDER) 
                   if os.path.isfile(os.path.join(CV_FOLDER, f)) and 
                   (f.endswith('.pdf') or f.endswith('.docx'))]
        
        if not cv_files:
            return jsonify({
                'status': 'error',
                'message': 'No CV files found in the CVs folder.'
            }), 400
        
        # Process CVs in batch
        logger.info(f"Processing {len(cv_files)} CVs")
        cv_results = batch_process_pdfs(cv_files, doc_type="resume")
        
        if not cv_results:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process any CV files.'
            }), 500
        
        logger.info(f"Successfully processed {len(cv_results)} CVs")
        
        # Extract preprocessed texts and filenames
        cv_texts = [result['preprocessed_text'] for result in cv_results]
        cv_filenames = [result['file_name'] for result in cv_results]
        
        # Track algorithm performance
        perf_metrics = {}
        
        # Initialize vectorizer with all texts (job + CVs)
        all_texts = [preprocessed_job] + cv_texts
        
        # Vectorization time
        vec_start = time.time()
        vectorizer = similarity_calculator.create_vectorizer(all_texts)
        job_vector = similarity_calculator.vectorize_job_description(preprocessed_job)
        cv_vectors = similarity_calculator.vectorize_resumes(cv_texts)
        perf_metrics['vectorization'] = time.time() - vec_start
        
        # Calculate cosine similarity
        cosine_start = time.time()
        cosine_scores = similarity_calculator.calculate_cosine_similarity(cv_vectors)
        perf_metrics['cosine'] = time.time() - cosine_start
        
        # Calculate sqrtcos similarity
        sqrtcos_start = time.time()
        sqrtcos_scores = similarity_calculator.calculate_sqrtcos_similarity(cv_vectors)
        perf_metrics['sqrtcos'] = time.time() - sqrtcos_start
        
        # Calculate ISC similarity
        isc_start = time.time()
        isc_scores = similarity_calculator.calculate_isc_similarity(cv_vectors)
        perf_metrics['isc'] = time.time() - isc_start
        
        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'Filename': cv_filenames,
            'Cosine_Score': cosine_scores,
            'SqrtCos_Score': sqrtcos_scores,
            'ISC_Score': isc_scores,
            'Word_Count': [result['word_count'] for result in cv_results]
        })
        
        # Sort by ISC score (highest first)
        results_df = results_df.sort_values('ISC_Score', ascending=False)
        
        # Save results to CSV
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        results_path = os.path.join(RESULTS_FOLDER, 'matching_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Save performance metrics
        perf_metrics['total'] = time.time() - start_time
        perf_df = pd.DataFrame({
            'Algorithm': list(perf_metrics.keys()),
            'Time_Seconds': list(perf_metrics.values())
        })
        perf_path = os.path.join(RESULTS_FOLDER, 'perf_metrics.csv')
        perf_df.to_csv(perf_path, index=False)
        
        logger.info(f"Matching completed in {perf_metrics['total']:.2f}s")
        
        # Return results
        return jsonify({
            'status': 'success',
            'message': f'Matched {len(cv_results)} CVs against job description',
            'results_url': '/results',
            'download_url': '/download/results',
            'performance': {k: f"{v:.4f}s" for k, v in perf_metrics.items()}
        })
        
    except Exception as e:
        logger.error(f"Error matching resumes: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/results')
def show_results():
    """Display matching results"""
    results_path = os.path.join(RESULTS_FOLDER, 'matching_results.csv')
    perf_path = os.path.join(RESULTS_FOLDER, 'perf_metrics.csv')
    
    if not os.path.exists(results_path):
        return render_template('results.html', has_results=False)
    
    try:
        # Read results
        results_df = pd.read_csv(results_path)
        
        # Limit to top 20
        results_df = results_df.head(20)
        
        # Format scores for better display
        results_df['Cosine_Score'] = results_df['Cosine_Score'].apply(lambda x: f"{x:.4f}")
        results_df['SqrtCos_Score'] = results_df['SqrtCos_Score'].apply(lambda x: f"{x:.4f}")
        results_df['ISC_Score'] = results_df['ISC_Score'].apply(lambda x: f"{x:.4f}")
        
        # Convert DataFrame to list of dictionaries for template
        results = results_df.to_dict('records')
        
        # Read performance metrics if available
        performance = None
        if os.path.exists(perf_path):
            perf_df = pd.read_csv(perf_path)
            performance = {row['Algorithm']: f"{row['Time_Seconds']:.4f}s" 
                          for _, row in perf_df.iterrows()}
        
        # Calculate summary stats
        summary = {
            'total_cvs': len(results),
            'avg_isc_score': results_df['ISC_Score'].astype(float).mean(),
            'median_isc_score': results_df['ISC_Score'].astype(float).median(),
            'max_isc_score': results_df['ISC_Score'].astype(float).max(),
            'min_isc_score': results_df['ISC_Score'].astype(float).min()
        }
        
        return render_template('results.html', 
                              has_results=True, 
                              results=results,
                              summary=summary,
                              performance=performance)
        
    except Exception as e:
        logger.error(f"Error displaying results: {e}")
        return render_template('results.html', has_results=False, error=str(e))

@app.route('/download/results')
def download_results():
    """Download matching results as CSV"""
    results_path = os.path.join(RESULTS_FOLDER, 'matching_results.csv')
    
    if not os.path.exists(results_path):
        return jsonify({
            'status': 'error',
            'message': 'No results found. Please run matching first.'
        }), 404
    
    try:
        return send_file(
            results_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cv_matching_results.csv'
        )
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# Run the application if file is executed directly
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True if active_config.FLASK_CONFIG == 'development' else False
    )