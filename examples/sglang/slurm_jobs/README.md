# SLURM Jobs for Dynamo Serve Benchmarking

This folder contains SLURM job scripts designed to launch Dynamo Serve service on SLURM cluster nodes and monitor GPU activity. The primary purpose is to automate the process of starting prefill and decode nodes to enable running benchmarks.

## Overview

The scripts in this folder orchestrate the deployment of Dynamo Serve across multiple cluster nodes, with separate nodes handling prefill and decode operations. The system uses a Python-based job submission system with Jinja2 templates for flexible configuration.

## Scripts

- **`submit_job_script.py`**: Main script for generating and submitting SLURM job scripts from templates
- **`job_script_template.j2`**: Jinja2 template for generating SLURM job scripts
- **`scripts/worker_setup.py`**: Worker script that handles the actual Dynamo Serve setup on each node
- **`scripts/monitor_gpu_utilization.sh`**: Script for monitoring GPU utilization during benchmarks

## Logs Folder Structure

Each SLURM job creates a unique log directory under `logs/` using the job ID. For example, job ID `3062824` creates the directory `logs/3062824/`.

### Log File Structure

```
logs/
├── 3062824/                    # Job ID directory
│   ├── log.out                 # Main job output (node allocation, IP addresses, launch commands)
│   ├── log.err                 # Main job errors
│   ├── node0197_prefill.out     # Prefill node stdout (node0197)
│   ├── node0197_prefill.err     # Prefill node stderr (node0197)
│   ├── node0200_prefill.out     # Prefill node stdout (node0200)
│   ├── node0200_prefill.err     # Prefill node stderr (node0200)
│   ├── node0201_decode.out      # Decode node stdout (node0201)
│   ├── node0201_decode.err      # Decode node stderr (node0201)
│   ├── node0204_decode.out      # Decode node stdout (node0204)
│   ├── node0204_decode.err      # Decode node stderr (node0204)
│   ├── node0197_prefill_gpu_utilization.log    # GPU utilization monitoring (node0197)
│   ├── node0200_prefill_gpu_utilization.log    # GPU utilization monitoring (node0200)
│   ├── node0201_decode_gpu_utilization.log     # GPU utilization monitoring (node0201)
│   └── node0204_decode_gpu_utilization.log     # GPU utilization monitoring (node0204)
├── 3063137/                    # Another job ID directory
├── 3062689/                    # Another job ID directory
└── ...
```

## Usage

1. **Submit a benchmark job**:
   ```bash
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri
   ```

   **Required arguments**:
   - `--template`: Path to Jinja2 template file
   - `--model-dir`: Model directory path
   - `--config-dir`: Config directory path
   - `--container-image`: Container image URI (e.g., `registry/repository:tag`)
   - `--account`: SLURM account

   **Optional arguments**:
   - `--prefill-nodes`: Number of prefill nodes (default: `2`)
   - `--decode-nodes`: Number of decode nodes (default: `2`)
   - `--gpus-per-node`: Number of GPUs per node (default: `8`)
   - `--network-interface`: Network interface to use (default: `eth3`)
   - `--job-name`: SLURM job name (default: `dynamo_setup`)
   - `--time-limit`: Time limit in HH:MM:SS format (default: `01:00:00`)

   **Note**: The script automatically calculates the total number of nodes needed based on `--prefill-nodes` and `--decode-nodes` parameters.

2. **Monitor job progress**:
   ```bash
   squeue -u $USER
   ```

3. **Check logs in real-time**:
   ```bash
   tail -f logs/{JOB_ID}/log.out
   ```

4. **Monitor GPU utilization**:
   ```bash
   tail -f logs/{JOB_ID}/{node}_prefill_gpu_utilization.log
   ```

## Outputs

Benchmark results and outputs are stored in the `outputs/` directory, which is mounted into the container.
