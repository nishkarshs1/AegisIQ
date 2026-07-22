#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI Backend on port 8000..."
python -m uvicorn mlApp:app --host 0.0.0.0 --port 8000 &

# Wait briefly for FastAPI to initialize
sleep 2

# Start Streamlit frontend in the foreground
echo "Starting Streamlit Frontend on port 8501..."
exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
