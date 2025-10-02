FROM rocker/r-ver:4.3.3

# Install Python 3.11 and pip
# The rocker images are Debian-based, so you can use apt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir numpy pandas nibabel networkx pyarrow templateflow tqdm

# Install R packages
RUN R -e "options(warn=2); install.packages(c('ggplot2', 'dplyr', 'tidyr', 'arrow', 'mgcv', 'lme4', 'broom', 'broom.mixed', 'readr', 'patchwork'))"

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