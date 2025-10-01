FROM python:3.11-slim
LABEL authors="david"

# Install R and dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir numpy pandas nibabel networkx pyarrow templateflow tqdm

# Install R dependencies
RUN Rscript -e "install.packages(c('ggplot2', 'dplyr', 'tidyr', 'arrow', 'mgcv', 'lme4', 'broom', 'readr'), repos='http://cran.r-project.org')"

# Set working directory
WORKDIR /app/hcpa

# Copy and set executable permissions for the script
COPY run_connectivity_stats.sh /app/hcpa/
RUN chmod +x /app/hcpa/run_connectivity_stats.sh

COPY compute_connectivity_smoothing_stats.py /app/hcpa/
RUN chmod +x /app/hcpa/compute_connectivity_smoothing_stats.py

COPY connectivity_distance_analysis.R /app/hcpa/
RUN chmod +x /app/hcpa/connectivity_distance_analysis.R

COPY connectivity_smoothing_analysis.R /app/hcpa/
RUN chmod +x /app/hcpa/connectivity_smoothing_analysis.R

ENTRYPOINT ["bash", "/app/hcpa/run_connectivity_stats.sh"]