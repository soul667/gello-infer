# Dockerfile for the simple client.

# Build the container:
# docker build . -t simple_client -f examples/simple_client/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app simple_client /bin/bash

FROM python:3.7-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./openpi/examples/simple_client/requirements.txt /tmp/requirements.txt
COPY ./openpi/packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/openpi-client/pyproject.toml

# ----------------[Install gello]-----------------------------
COPY ./gello_software/requirements.txt /tmp/requirements.txt

RUN uv pip install -r /tmp/requirements.txt

COPY ./gello_software /app/gello_software
RUN cd /app/gello_software && uv pip install -e .
RUN cd /app/gello_software/third_party/DynamixelSDK/python && uv pip install -e .

# 清理gello代码和/tmp
RUN rm -rf /app/gello_software && rm -f /tmp/requirements.txt

