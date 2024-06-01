# Base image with CUDA and Python
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /usr/src/app

# Copy the local directory contents to the container
COPY . .

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Install Poetry
ARG POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
   
# Set Poetry configuration: Do not create a virtual environment
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies using Poetry. Exclude development dependencies if not needed
RUN poetry install --no-root

# Expose the port the app runs on
EXPOSE 80

# Command to run on container start
CMD ["python3", "tfg/run.py"]
