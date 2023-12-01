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

# Copy pyproject.toml, setup.py, and README.md (and possibly other necessary files)
COPY pyproject.toml setup.py README.md /code/

# Install build dependencies from pyproject.toml and runtime dependencies from setup.py
RUN pip install .

# Copy the rest of your application
COPY . /code