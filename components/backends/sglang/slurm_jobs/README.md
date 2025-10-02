# Example: Deploy DeepSeek R1 - FP8 with Dynamo and SGLang on SLURM

This folder allows you to deploy the SGLang DeepSeek-R1 Disaggregated with WideEP on a GB200 SLURM cluster.

## SLURM Prerequisites

For this example, we will make some assumptions about your SLURM cluster:

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
   described [here](../docs/dsr1-wideep-gb200.md#instructions).
   This is the image that can be passed to the `--container-image` argument in later steps.

## Scripts Overview

- **`submit_job_script.py`**: Main script for generating and submitting SLURM job scripts from templates
- **`job_script_template.j2`**: Jinja2 template for generating SLURM sbatch scripts
- **`scripts/worker_setup.py`**: Worker script that handles the setup on each node
- **`submit_disagg.sh`**: A simple one-liner script that invokes the `submit_job_script.py`

## Logs Folder Structure

Each SLURM job creates a unique log directory under `logs/` using the job ID. For example, job ID `3062824` creates the directory `logs/3062824/`.

## Usage

> [!NOTE]
> The logic for finding prefill and decode node IPs in [`job_script_template.j2`](job_script_template.j2) is still a work in progress. You may need to tweak the `ip addr show $NETWORK_INTERFACE` bits for your cluster, especially if your networking or hostname conventions differ. PRs and suggestions are always welcome.

1. **Submit a benchmark job**:

   ```bash
   python3 submit_job_script.py \
     --template job_script_template.j2 \
     --model-dir <path-to>/deepseek-r1-0528 \
     --container-image <path-to>/dynamo-sglang+v0.5.3rc1-v0.3.12.sqsh \
     --gpus-per-node 4 \
     --config-dir <path-to>/klconfigs \
     --gpu-type gb200-fp8 \
     --network-interface enP6p9s0np0 \
     --prefill-nodes 6 \
     --decode-nodes 12 \
     --prefill-workers 3 \
     --decode-workers 1 \
     --account <account> \
     --partition <partition> \
     --time-limit 4:00:00 \
     --enable-multiple-frontends \
     --num-additional-frontends 9 \
     --profiler "type=vllm; isl=8192; osl=1024; concurrencies=16x2048x4096x8192; req-rate=inf"
   ```

   This command will deploy 3 prefill workers and 1 decode worker with 9 additional frontends load-balanced by nginx. Diving deeper into the command:

   - `--template job_script_template.j2`: Path to Jinja2 template file (this shouldn't change unless you want to modify the template)
   - `--model-dir <path-to>/deepseek-r1-0528`: Path to DSR1-FP8 model directory
   - `--container-image <path-to>/dynamo-sglang+v0.5.3rc1-v0.3.12.sqsh`: Enroot container image URI
   - `--gpus-per-node 4`: Number of GPUs per node (each GB200 tray has 4 GPUs)
   - `--config-dir <path-to>/klconfigs`: Various configs (see explanation below)
   - `--gpu-type gb200-fp8`: GPU type to use, choices: `gb200-fp8`
   - `--network-interface enP6p9s0np0`: Network interface to use (depends on your cluster)
   - `--prefill-nodes 6`: Number of prefill nodes
   - `--decode-nodes 12`: Number of decode nodes
   - `--prefill-workers 3`: Number of prefill workers
   - `--decode-workers 1`: Number of decode workers
   - `--account <account>`: SLURM account
   - `--partition <partition>`: SLURM partition
   - `--time-limit 4:00:00`: Time limit in HH:MM:SS format
   - `--enable-multiple-frontends`: Enable multiple frontend architecture with nginx load balancer
   - `--num-additional-frontends 9`: Number of additional frontends
   - `--profiler "type=vllm; isl=8192; osl=1024; concurrencies=16x2048x4096x8192; req-rate=inf"`: Profiler configurations (see explanation below)

   **Note**: The script automatically calculates the total number of nodes needed based on `--prefill-nodes` and `--decode-nodes` parameters.

2. **Check logs in real-time**:

   ```bash
   cd logs/{JOB_ID}
   tail -f *_prefill_*.err *_decode_*.err
   ```

## Configs directory

The `--config-dir` argument is used to specify the directory containing the various configs that are used when running this model. Here are the current configs that are in our directory.

```bash
klconfigs/
├── decode_dsr1-0528_loadgen_in1024out1024_num2000_2p12d.json
├── deepep_config.json
├── dgcache/
└── prefill_dsr1-0528_in1000out1000_num40000.json
```

1. `decode_dsr1-0528_loadgen_in1024out1024_num2000_2p12d.json`: `init-expert-location` for decode worker
2. `deepep_config.json`: DeepEP config file for GB2009
3. `dgcache/`: DeepGEMM kernel cache directory. Instructions for creating this can be found [here](https://github.com/sgl-project/sglang/issues/9867#issuecomment-3336551174)
4. `prefill_dsr1-0528_in1000out1000_num40000.json`: `init-expert-location` for prefill worker

**Note**: The expert locations are collected using the instructions [here](https://github.com/sgl-project/sglang/issues/6017). See the section titled "Create expert distribution data". Note that this is sensitive to your data and performance results may differ if you dont benchmark with the same data that was used to collect the expert locations.

## Profiler

If you provide the `--profiler` command, the sbatch script will automatically warmup the model and run the vllm benchmarking script. Benchmark results and outputs are stored in the `outputs/` directory, which is mounted into the container.
