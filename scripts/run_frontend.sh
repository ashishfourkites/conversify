#!/bin/bash

# Script to run the Conversify frontend

cd "$(dirname "$0")/../frontend" || exit

echo "Starting Conversify Frontend Server..."
echo "--------------------------------------"
echo "Frontend will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "Using Python HTTP server..."
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    echo "Using Python HTTP server..."
    python -m http.server 8000
elif command -v npx &> /dev/null; then
    echo "Using npx serve..."
    npx serve -p 8000
else
    echo "Error: No suitable server found!"
    echo "Please install Python or Node.js"
    exit 1
fi