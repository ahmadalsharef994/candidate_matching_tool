"""
Configuration settings for the Candidate Matching Tool
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    TESTING = False
    
    # MongoDB settings
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'resumes_db')
    MONGO_COLLECTION = os.getenv('MONGO_COLLECTION', 'resumes')
    
    # AWS settings
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY', '')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY', '')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET = os.getenv('S3_BUCKET', 'resume-bucket')
    
    # File paths
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
    PROCESSED_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'csv', 'xlsx'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # MLflow tracking
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MONGO_DB_NAME = 'test_resumes_db'


# Create config dictionary to easily access configurations
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# The active configuration
active_config = config[os.getenv('FLASK_CONFIG', 'default')]()
