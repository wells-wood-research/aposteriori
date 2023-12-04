FROM python:3.8

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Set work directory
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY . /code

# Upgrade pip and install Cython explicitly
RUN pip install --upgrade pip && pip install Cython

# Install build dependencies from pyproject.toml and runtime dependencies from setup.py
RUN pip install .

RUN pip install -r dev-requirements.txt