#!/bin/bash
# Start the simplified Candidate Matching Application

# Load environment variables properly
set -a
source .env
set +a


echo "Starting Candidate Matching application..."

# For development, use Flask's built-in server
if [ "$FLASK_CONFIG" = "development" ]; then
    flask run --host 0.0.0.0 --port 5001
else
    # For production, use gunicorn
    gunicorn \
      --bind 0.0.0.0:5001 \
      --workers 4 \
      --threads 2 \
      --timeout 120 \
      --log-file logs/gunicorn.log \
      --access-logfile logs/access.log \
      --log-level info \
      app.main:app
fi

echo "Application started"