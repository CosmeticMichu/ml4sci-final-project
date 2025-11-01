FROM mcr.microsoft.com/devcontainers/python:3.8

# installs packages
RUN apt-get update && \
    apt-get install -y git-lfs && \
    apt-get clean

# installs uv
RUN pip install uv