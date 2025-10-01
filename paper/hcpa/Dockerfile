FROM python:3.11-slim
LABEL authors="david"

# install R and dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
RUN pip install --no-cache-dir numpy scipy pandas nibabel nilearn

S