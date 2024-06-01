# Base image with CUDA and Python
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /usr/src/app

# Copy the local directory contents to the container
COPY . .

# Install Python 3.10 and pip
RUN apt-get update; \
    apt-get install python3.11; \
    apt-get install -y python3-pip;

# Install Poetry
RUN pip3 install poetry

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false; \
    poetry env use python3.11; \
    poetry install --no-root;

# Expose the port the app runs on
EXPOSE 80

# Command to run on container start
CMD ["python3.10", "tfg/run.py"]
