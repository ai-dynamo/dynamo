# Example: Deploy Multi-node SGLang with Dynamo on SLURM

This folder implements the example of [SGLang DeepSeek-R1 Disaggregated with WideEP](../docs/dsr1-wideep-h100.md) on a SLURM cluster.

## Overview

The scripts in this folder set up multiple cluster nodes to run the [SGLang DeepSeek-R1 Disaggregated with WideEP](../docs/dsr1-wideep-h100.md) example, with separate nodes handling prefill and decode.
The node setup is done using Python job submission scripts with Jinja2 templates for flexible configuration. The setup also includes GPU utilization monitoring capabilities to track performance during benchmarks.

## Scripts

- **`submit_job_script.py`**: Main script for generating and submitting SLURM job scripts from templates
- **`job_script_template.j2`**: Jinja2 template for generating SLURM job scripts
- **`scripts/worker_setup.py`**: Worker script that handles the setup on each node
- **`scripts/monitor_gpu_utilization.sh`**: Script for monitoring GPU utilization during benchmarks
- **`submit_disagg.sh`**: A simple one-liner script that invokes the `submit_job_script.py`

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
   described [here](../docs/dsr1-wideep-h100.md#instructions).
   This is the image that can be passed to the `--container-image` argument in later steps.

## Usage

> [!NOTE]
> The logic for finding prefill and decode node IPs in [`job_script_template.j2`](job_script_template.j2) is still a work in progress. You may need to tweak the `srun`/`ip route`/`getent`/`awk` bits for your cluster, especially if your networking or hostname conventions differ. PRs and suggestions welcome.

1. **Submit a benchmark job**:

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
   - `--time-limit`: Time limit in HH:MM:SS format (default: `01:00:00`)
   - `--gpu-type`: GPU type to use, choices: `h100`, `gb200` (default: `h100`)
   - `--use-sglang-commands`: Use SGLang commands instead of Dynamo (default: `false`)

   **Note**: The script automatically calculates the total number of nodes needed based on `--prefill-nodes` and `--decode-nodes` parameters.

2. **Example with different GPU types**:

   ```bash
   # For H100 with Dynamo (default)
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account \
     --gpu-type h100

   # For GB200 with SGLang
   python submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir /path/to/model \
     --config-dir /path/to/configs \
     --container-image container-image-uri \
     --account your-slurm-account \
     --gpu-type gb200 \
     --use-sglang-commands
     --gpus-per-node 4
   ```

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

## Outputs

Benchmark results and outputs are stored in the `outputs/` directory, which is mounted into the container.
