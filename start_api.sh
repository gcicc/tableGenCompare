#!/bin/bash

echo "Starting Synthetic Data Generation API..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Install API requirements if needed
echo "Installing API requirements..."
pip3 install -r requirements-api.txt

# Start the API server
echo
echo "Starting API server on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

python3 run_api.py --env development --reload