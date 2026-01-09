# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the dashboard server and static files
COPY serve_dashboard.py .
COPY dashboard/ ./dashboard/

# Copy all results files
COPY comparison_results/ ./comparison_results/
COPY translated_results/ ./translated_results/

# Copy samples directory (if needed by the dashboard)
COPY samples/ ./samples/

# Expose port 8000 (default port for the dashboard)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the dashboard server
CMD ["python", "serve_dashboard.py", "--host", "0.0.0.0", "--port", "8000"]
