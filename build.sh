#!/bin/bash

# Build script for Vercel deployment
echo "Starting Django build process..."

# Install dependencies
pip install -r requirements.txt

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Create database and run migrations
echo "Running database migrations..."
python manage.py migrate --noinput

echo "Build process completed successfully!"