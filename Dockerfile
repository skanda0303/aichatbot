# Use a lightweight python base image
FROM python:3.10-slim

# Install system dependencies needed for compiling python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create documents folder and database folder
RUN mkdir -p docs_multi chroma_db_multi

# Set default port to 7860 (Hugging Face Spaces default, main.py reads $PORT dynamically)
EXPOSE 7860
ENV PORT=7860

# Command to run the application
CMD ["python", "-m", "multi_agent.main"]
