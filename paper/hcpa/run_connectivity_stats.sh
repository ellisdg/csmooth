start_time=$(date +%s)
echo "Starting connectivity stats computation at $(date)"

CONN_DIR="/data2/david.ellis/public/HCPA/myderivatives/connectivity"
OUTPUT_DIR="/data2/david.ellis/public/HCPA/myderivatives/connectivity_stats"
mkdir -p $OUTPUT_DIR
METRICS_FILE="$OUTPUT_DIR/conn_smoothing_graph_metrics.csv"
DATA_FILE="$OUTPUT_DIR/conn_smoothing_distance_data.parquet"

echo "Computing connectivity stats"
python3 /app/hcpa/compute_connectivity_smoothing_stats.py\
  --connectivity_dir $CONN_DIR\
  --output_metrics $METRICS_FILE\
  --output_data $DATA_FILE\
  --graph_density 0.2\
  --n_subjects 100
conn_stats_time=$(date +%s)
echo "Connectivity stats computation completed at $(date), time taken: $(($conn_stats_time - start_time)) seconds"

# Now run the R scripts to run statistical analyses and generate figures
echo "Running statistical analyses and generating figures"
Rscript /app/hcpa/connectivity_distance_analysis.R $DATA_FILE $OUTPUT_DIR
Rscript /app/hcpa/connectivity_smoothing_analysis.R $METRICS_FILE $OUTPUT_DIR
analysis_time=$(date +%s)
echo "Statistical analyses and figure generation completed at $(date), time taken: $(($analysis_time - conn_stats_time)) seconds"