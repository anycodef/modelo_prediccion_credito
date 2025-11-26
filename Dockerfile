
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements_vertex.txt .

# Install any needed packages specified in requirements_vertex.txt
RUN pip install --no-cache-dir -r requirements_vertex.txt

# Copy the rest of the application's code
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Set environment variables
# PYTHONUNBUFFERED ensures that python output is sent straight to terminal (e.g. your container logs)
ENV PYTHONUNBUFFERED=1

# Define a default command. This can be overridden when running the container.
# For example, to run the training script:
# docker run <image_name> python src/entrenar_modelo.py
CMD ["python", "src/entrenar_modelo.py"]
