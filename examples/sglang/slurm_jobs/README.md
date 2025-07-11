# Example: Deploy Multi-node SGLang with Dynamo on SLURM

This folder implements the example of [SGLang DeepSeek-R1 Disaggregated with WideEP](../dsr1-wideep.md) on a SLURM cluster.

## Overview

The scripts in this folder set up multiple cluster nodes to run the [SGLang DeepSeek-R1 Disaggregated with WideEP](../dsr1-wideep.md) example, with separate nodes handling prefill and decode.
The node setup is done using Python job submission scripts with Jinja2 templates for flexible configuration. The setup also includes GPU utilization monitoring capabilities to track performance during benchmarks.

## Scripts

- **`submit_job_script.py`**: Main script for generating and submitting SLURM job scripts from templates with interactive workflow capabilities
- **`job_script_template.j2`**: Jinja2 template for generating SLURM job scripts
- **`scripts/worker_setup.py`**: Worker script that handles the setup on each node
- **`scripts/monitor_gpu_utilization.sh`**: Script for monitoring GPU utilization during benchmarks
- **`requirements.txt`**: Python dependencies for the job submission script

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Logs Folder Structure

Each SLURM job creates a unique log directory under `logs/` using the job ID. For example, job ID `3062824` creates the directory `logs/3062824/`.

### Log File Structure

```
logs/
├── 3062824/                    # Job ID directory
│   ├── job_info.json           # Job information for interactive workflow
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
│   ├── node0204_decode_gpu_utilization.log     # GPU utilization monitoring (node0204)
│   ├── warmup/                  # Interactive workflow step logs
│   │   ├── warmup.log
│   │   └── warmup.err
│   └── flush_cache/             # Interactive workflow step logs
│       ├── flush_cache.log
│       └── flush_cache.err
├── 3063137/                    # Another job ID directory
├── 3062689/                    # Another job ID directory
└── ...
```

## Setup

For simplicity of the example, we will make some assumptions about your SLURM cluster:
1. We assume you have access to a SLURM cluster with multiple GPU nodes
   available. For functional testing, most setups should be fine. For performance
   testing, you should aim to allocate groups of nodes that are performantly
   inter-connected, such as those in an NVL72 setup.
2. We assume this SLURM cluster has the [Pyxis](https://github.com/NVIDIA/pyxis)
   SPANK plugin setup. In particular, the `job_script_template.j2` template in this
   example will use `srun` arguments like `--container-image`,
   `--container-mounts`, and `--container-env` that are added to `srun` by Pyxis.
   If your cluster supports similar container based plugins, you may be able to
   modify the template to use that instead.
3. We assume you have already built a recent Dynamo+SGLang container image as
   described [here](../dsr1-wideep.md#instructions).
   This is the image that can be passed to the `--container-image` argument in later steps.
4. The required Python dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Job Submission

1. **Submit a benchmark job**:

   > **Note:** The logic for finding the prefill and decode node IP addresses in [`job_script_template.j2`](job_script_template.j2) is still a work in progress. You may need to tweak the `srun`/`ip route`/`getent`/`awk` commands for your cluster setup, especially if your networking or hostname conventions differ. PRs and suggestions welcome.

   ```bash
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account
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
   - `--time-limit`: Time limit in HH:MM:SS format (default: `04:00:00`)
   - `--gpu-type`: GPU type to use, choices: `h100`, `gb200` (default: `h100`)
   - `--use-sglang-commands`: Use SGLang commands instead of Dynamo (default: `false`)

   **Note**: The script automatically calculates the total number of nodes needed based on `--prefill-nodes` and `--decode-nodes` parameters.

### Interactive Workflow

The script now supports an interactive workflow mode that automatically manages the job lifecycle:

1. **Submit and run interactively**:
   ```bash
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account \
     --interactive
   ```

   The interactive mode performs the following steps automatically:
   - **Wait for Allocation**: Waits for SLURM to allocate the requested nodes
   - **Wait for Setup**: Monitors all nodes until they complete initialization and GPUs are idle
   - **Warmup**: Runs a warmup benchmark on the prefill host
   - **Flush Cache**: Calls the flush cache endpoint to clear any cached data
   - **Cleanup**: Automatically cancels the job when the workflow completes

2. **Example with different GPU types**:
   ```bash
   # For H100 with Dynamo (default)
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account \
     --gpu-type h100 \
     --interactive

   # For GB200 with SGLang
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account \
     --gpu-type gb200 \
     --use-sglang-commands \
     --gpus-per-node 4 \
     --interactive
   ```

### Manual Monitoring (Non-Interactive Mode)

If not using interactive mode, you can manually monitor the job:

3. **Monitor job progress**:
   ```bash
   squeue -u $USER
   ```

4. **Check logs in real-time**:
   ```bash
   tail -f logs/{JOB_ID}/log.out
   ```

   You can view logs of all prefill or decode workers simultaneously by running:
   ```bash
   # prefill workers err (or .out)
   tail -f logs/{JOB_ID}/*_prefill.err

   # decode workers err (or .out)
   tail -f logs/{JOB_ID}/*_decode.err
   ```

5. **Monitor GPU utilization**:
   ```bash
   tail -f logs/{JOB_ID}/{node}_prefill_gpu_utilization.log
   ```

## Interactive Workflow Features

The enhanced `submit_job_script.py` includes several new features:

- **Automatic Job Lifecycle Management**: The interactive mode handles the complete job lifecycle from submission to cleanup
- **Node Status Monitoring**: Automatically waits for nodes to be allocated and initialized
- **GPU Utilization Tracking**: Monitors GPU utilization to ensure nodes are ready before proceeding
- **Structured Logging**: Each step of the interactive workflow is logged separately with timestamps
- **Error Handling**: Comprehensive error handling with automatic cleanup on failures
- **Job Information Persistence**: Job details are stored in `job_info.json` for communication between steps

## Outputs

Benchmark results and outputs are stored in the `outputs/` directory, which is mounted into the container.
