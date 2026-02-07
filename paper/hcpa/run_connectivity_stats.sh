#!/usr/bin/env bash
set -euo pipefail

# Default directories (fallbacks)
CONN_DIR="/data2/david.ellis/public/HCPA/myderivatives/connectivity"
OUTPUT_DIR="/data2/david.ellis/public/HCPA/myderivatives/connectivity_stats"

usage() {
  echo "Usage: $0 [--conn-dir DIR] [--output-dir DIR]"
  echo ""
  echo "Options:"
  echo "  --conn-dir DIR      Path to connectivity directory (default: $CONN_DIR)"
  echo "  --output-dir DIR    Path to output directory (default: $OUTPUT_DIR)"
  echo "  -h, --help          Show this help message"
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --conn-dir)
      CONN_DIR="$2"
      shift 2
      ;;
    --conn-dir=*)
      CONN_DIR="${1#*=}"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

start_time=$(date +%s)
echo "Starting connectivity stats computation at $(date)"

mkdir -p "$OUTPUT_DIR"
METRICS_FILE="$OUTPUT_DIR/conn_smoothing_graph_metrics.csv"
DATA_FILE="$OUTPUT_DIR/conn_smoothing_distance_data.parquet"

echo "Computing connectivity stats"
python3 /app/hcpa/compute_connectivity_smoothing_stats.py \
  --connectivity_dir "$CONN_DIR" \
  --output_metrics "$METRICS_FILE" \
  --output_data "$DATA_FILE" \
  --graph_density 0.2 \
  --n_subjects 100

conn_stats_time=$(date +%s)
echo "Connectivity stats computation completed at $(date), time taken: $(($conn_stats_time - start_time)) seconds"

echo "Running statistical analyses and generating figures"
Rscript /app/hcpa/connectivity_distance_analysis.R "$DATA_FILE" "$OUTPUT_DIR"
Rscript /app/hcpa/connectivity_smoothing_analysis.R "$METRICS_FILE" "$OUTPUT_DIR"

analysis_time=$(date +%s)
echo "Statistical analyses and figure generation completed at $(date), time taken: $(($analysis_time - conn_stats_time)) seconds"