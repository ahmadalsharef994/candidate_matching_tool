# Candidate Matching Tool

A simple resume matching and ranking tool that compares job candidates against job descriptions using multiple similarity metrics.

## Overview

This system matches and ranks job candidates against job descriptions using three text similarity techniques: cosine similarity, square root cosine similarity, and Improved Similarity Coefficient (ISC). It provides a straightforward web interface for uploading job descriptions, and displays the top 20 candidates ranked by similarity scores.

 Built to process large candidate pools (20k–30k resumes) in seconds and return top-ranked profiles.

## Key Features

- **Multiple Similarity Algorithms**:
  - Cosine Similarity: Standard TF-IDF vector comparison
  - SqrtCos Similarity: Square root cosine similarity for better term weighting
  - ISC Similarity: Improved Similarity Coefficient combining cosine similarity with term overlap measures

- **Text Processing**:
  - PDF parsing with Tika fallback
  - NLP-based text cleaning and preprocessing
  - TF-IDF vectorization with domain-specific parameters

- **Simple Web Interface**:
  - Upload job descriptions (PDF or text)
  - View top 20 candidates in a ranked table
  - Performance metrics for each similarity algorithm
  - Download results as CSV

- **Automated Processing**:
  - CRON job for preprocessing new resumes weekly
  - Automatic processing on application startup

## Architecture

## 🧩 System Components & Technology stack

| Component             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Web Interface         | Flask-based web application                                                 |
| PDF Parsing           | pdftotext for primary parsing, with Apache Tika as fallback               |
| Text Processing       | NLTK for tokenization, stemming, and redundancy removal                     |
| Similarity Engine     | Cosine Similarity (Scikit-learn), SqrtCos & ISC (custom algorithms)         |
| Vectorization         | TF-IDF (Scikit-learn), with logic to penalize overly long profiles          |
| Data Handling         | Pandas for result processing, PyMongo for optional data storage             |
| Automation            | Weekly CRON jobs for CV preprocessing from S3 or local sources              |
| Deployment            | Hosted on DigitalOcean                                                      |
| Secrets Management    | AWS (S3) and MongoDB credentials                                             |
| Planned Upgrades      | FAISS for semantic similarity; Docker for containerized deployment          |


## 🔄 Job-CV Matching Workflow

1. **Raw Data Processing (Weekly CRON Job)**
   - CVs (from S3 or local storage) are processed into a standardized format
   - Processed CVs are saved in a dedicated folder (processed/)

2. **Job Description Input**
   - Users upload a job description as a PDF or paste text directly into the system

3. **Candidate Selection**
   - The system automatically selects and loads CVs from the processed folder

4. **Similarity Analysis**
   - Each CV is compared to the job description using three similarity algorithms:
     - **Cosine Similarity** (via scikit-learn)
     - **SqrtCos Similarity** (custom)
     - **ISC Similarity** (custom)

5. **Results Display**
   - Top 20 candidates are ranked based on aggregated similarity scores
   - Performance metrics and comparison across all three algorithms are shown

6. **Export Results**
   - Users can export the ranked candidate list and scores as a CSV file


## Metrics
📊
| Type | Examples |
|------|----------|
| Model Metrics | Cosine score, custom similarity |
| Software Metrics | Latency, throughput |
| Business Impact | Reduced screening time, improved matching quality, recruiter trust |

## Data & Ethics
📂 
- **Data Sources**: Scraping, uploads, open-source test sets, collective data
- **Privacy**: No private data input to AI; only metadata (link to profile) stored
- **Labeling & Validation**: Human feedback, blind testing (recruiter vs model scores)
- **Ethical Checks**: Monitored for bias, removed deleted profiles following GDPR
- **Interoperability**: Modular components with ETL-style architecture

## Architecture Principles
🔧 
- Unified data structure with reduced dimensionality (via feature selection)
- Preprocessing prioritized over post-model interpretation
- Emphasis on ETL over ELT
- Dynamic shifting enabled through modular orchestration

## Timeline
🧪
- Development Completed in: 1 week

## Planned Improvements
📈
- Use FAISS for vector storage (better scalability)
- Full Dockerization for reproducibility and deployment
- CI/CD integration for automatic update checks
- F1-score


## Installation and Setup


### Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd production-candidate-matcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables example and edit as needed
cp .env.example .env

# Create necessary directories
mkdir -p data/raw data/processed/cvs data/models logs
```

### Configuration

Edit the `.env` file to configure basic application settings:

```bash
# Application
FLASK_CONFIG=development
FLASK_APP=app.main:app
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO

# File paths
CV_FOLDER=./data/processed/cvs
JOB_DESC_PATH=./job_description.pdf
PROCESSED_FOLDER=./data/processed
RAW_FOLDER=./data/raw
RESULTS_FOLDER=./data/processed
```

## Usage

### Start the application

```bash
./start.sh
```

This will start the web application at http://localhost:5001 and run the initial CV processing job.

### Using the web interface

1. **Upload Job Description**:
   - Use the form to upload a PDF file or paste job description text

2. **View Top Candidates**:
   - After uploading, the system automatically processes and displays the top 20 candidates
   - Each candidate is ranked using three similarity metrics (Cosine, SqrtCos, ISC)
   - Performance metrics (latency) for each algorithm are displayed

3. **Download Results**:
   - Download the full results as a CSV file for further analysis

### Automated Processing with CRON

The system includes a CRON job that runs weekly to process new resumes:

```bash
# Added automatically to crontab on first run
0 0 * * 0 cd /path/to/production-candidate-matcher && python3 cron_preprocess.py >> logs/cron.log 2>&1
```

This will:
1. Check the `data/raw` folder weekly for new resume files
2. Process and move them to the `data/processed/cvs` folder
3. Remove processed files from raw folder


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and scikit-learn communities for text processing tools
- Flask framework for the web interface
- Tika and pdftotext libraries for PDF parsing


