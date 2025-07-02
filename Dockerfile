FROM python:3.11-slim

# copy current project
COPY . /app

# install project as a package
RUN pip install --no-cache-dir /app

ENV NETWORKX_AUTOMATIC_BACKENDS="parallel"

ENTRYPOINT ["python", "/app/csmooth/fmriprep.py"]
