# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /code

# Install ffmpeg (Crucial for YouTube audio processing)
RUN apt-get update && apt-get install -y ffmpeg

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the app
COPY . /code

# Create templates directory just in case
RUN mkdir -p /code/templates

# Start the app on port 7860 (Hugging Face default)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
