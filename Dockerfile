# Dockerfile 
# 1. Use a lightweight official Python base image 
FROM python:3.10-slim

# 2. Set the working directory inside the container
# All subsequent commands will run from this directory
WORKDIR /usr/src/app

# 3. Copy the entire project directory contents into the container's working directory
COPY . .

# 4. Upgrade pip and install Python dependencies listed in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. Set environment variable to ensure Python can locate the `scheduler/` and `app/` modules
ENV PYTHONPATH=/usr/src/app

# 6. Expose the port that the FastAPI server will run on (default: 8000)
EXPOSE 8000

# 7. Start the FastAPI application using Uvicorn as the ASGI server
# It will serve the app from app.main:app on all available IPs
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
