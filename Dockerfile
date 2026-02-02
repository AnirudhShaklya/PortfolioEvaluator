# Use the official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your local folder to the container
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Required by Cloud Run)
EXPOSE 8080

# Command to run the app
# We explicitly set the server port to 8080 to match Cloud Run's expectation
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]