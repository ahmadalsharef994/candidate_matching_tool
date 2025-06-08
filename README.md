# Candidate Matching Tool

A simple resume matching and ranking tool that compares job candidates against job descriptions using multiple similarity metrics.

## Overview

This system matches and ranks job candidates against job descriptions using three text similarity techniques: cosine similarity, square root cosine similarity, and Improved Similarity Coefficient (ISC). It provides a straightforward web interface for uploading job descriptions, and displays the top 20 candidates ranked by similarity scores.


A scalable web-based tool for screening and ranking job candidates against job descriptions using text similarity techniques. Built to process large candidate pools (20k–30k resumes) and return top-ranked profiles.

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

### Components

- **Web Interface**: Flask web application
- **PDF Parsing**: pdftotext with Tika parser fallback
- **Text Processing**: NLTK for tokenization, stemming, and redundancy removal
- **Similarity Engine**: Scikit-learn with custom algorithms
- **Data Handling**: Pandas for results processing

### Workflow

1. **Raw Data Processing**: CRON job processes raw CVs into processed format weekly
2. **Job Description Input**: Upload a job description (PDF) or paste text
3. **Candidate Selection**: System automatically selects CVs from processed folder
4. **Similarity Analysis**: Each CV compared to the job description using all three similarity algorithms
5. **Results Display**: Top 20 candidates ranked based on similarity scores with performance metrics
6. **Export Results**: Download matches as CSV

## Installation and Setup

### Prerequisites

- Python 3.8+
- NLTK data
- Tika Parser

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

## Testing

```bash
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and scikit-learn communities for text processing tools
- Flask framework for the web interface
- Tika and pdftotext libraries for PDF parsing



## Tools & Technologies 
🛠
| Component | Tools Used |
|-----------|------------|
| Web Backend | Flask |
| PDF Parsing | Pdf2Text |
| Text Processing | NLTK (manual tokenization, stemming, redundancy removal) |
| Similarity Engine | Scikit-learn (Cosine), Custom Algorithms (SqrtCos, ISC) |
| Vectorization | Scikit-learn TF-IDF (penalizes overlong profiles) |
| Data Handling | Pandas, PyMongo |
| Automation | CRON jobs for S3 polling |
| Planned Upgrades | FAISS for embeddings, Dockerization |
| Deployment | DigitalOcean |
| Secrets | AWS (S3), MongoDB credentials |

## Workflow
🔄
1. **Job Description Input**
   - Users upload job description (or paste in text field)
   
2. **Text Preprocessing**
   - Using cron jobs (weekly) the raw data from S3 (or local storage) will be processed and stored in processed folder (S3 or local storage)
   
3. **Similarity Analysis**
   - Job description compared to each candidate using 3 similarity algorithms:
     - Cosine Similarity (scikit-learn)
     - SqrtCos Similarity (custom)
     - ISC Similarity (custom)
   
4. **Results Display**
   - Ranked candidates shown based on similarity score

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