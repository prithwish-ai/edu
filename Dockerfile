FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV PORT=5000

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p temp/uploads logs temp_audio

# Set permissions
RUN chmod -R 755 temp temp_audio logs

# Expose the port
EXPOSE 5000

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT 'server:create_app()' 