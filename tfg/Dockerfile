# Base image with CUDA and Python
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

# Set the working directory
WORKDIR /usr/src/app

# Copy the local directory contents to the container
COPY . .

# Install Python 3.11 and pip
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip && \
    apt-get clean

# Verify versions of Python and pip
RUN python3 --version && pip3 --version

# Install Poetry
RUN pip3 install poetry

# Configure Poetry to not use virtual environments and install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Expose the port the app runs on
EXPOSE 80

# Command to run on container start
CMD ["python3", "tfg/run.py"]
