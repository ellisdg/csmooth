#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=20G  # Memory per CPU in MB
#SBATCH --ntasks-per-node=2  # Number of CPU cores per node
#SBATCH --time=1-00:00:00  # Requested Runtime (in D-HH:MM:SS)
#SBATCH --error=/data2/david.ellis/public/HCPA/cache/job_logs/matrix.%J.err  # File to which STDERR will be written
#SBATCH --output=/data2/david.ellis/public/HCPA/cache/job_logs/matrix.%J.out  # File to which STDOUT will be written

input_files=$1
output_file=$2
start_time=`date +%s`
echo "Input files: ${input_files}"
echo "Output file: ${output_file}"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: `hostname`"
echo "Starting at: `date`"

apptainer exec -B /data -B /data2 /data/david.ellis/apptainer/csmooth.sif python /data2/david.ellis/public/HCPA/code/connectivity/compute_connectivity_matrix.py --input_files ${input_files} --output_file "${output_file}"

end_time=`date +%s`
elapsed=$(( end_time - start_time ))
echo "Elapsed time: $elapsed seconds"
echo "Finished at: `date`"