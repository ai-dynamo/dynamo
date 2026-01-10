# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to generate SLURM job scripts from Jinja2 templates.
"""

import argparse
import logging
import os
import pathlib
import subprocess
import tempfile
from datetime import datetime

from jinja2 import Template


def print_welcome_message(job_ids: list[str], log_dir_name: str):
    """Print a clean welcome message with job information."""

    _ = f"{', '.join(job_ids)}"
    print(
        f"""
🚀 Welcome! We hope you enjoy your time on our GB200 NVL72.

Your logs for this submitted job will be available in logs/{log_dir_name}
You can access them by running:

    cd logs/{log_dir_name}

You can view all of the prefill/decode worker logs by running:

    tail -f *_decode_*.err *_prefill_*.err

To kick off the benchmark we suggest opening up a new terminal, SSH-ing
into the login node, and running the srun command that is found at the
bottom of the log.out. You can find it by running:

    cat log.out

Enjoy :)
- NVIDIA
"""
    )


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_job_script(template_path, output_path, **kwargs):
    """Generate a job script from template with given parameters."""
    with open(template_path, "r") as f:
        template = Template(f.read())

    rendered_script = template.render(**kwargs)
    with open(output_path, "w") as f:
        f.write(rendered_script)

    return output_path, rendered_script


def submit_job(job_script_path, extra_slurm_args=[]):
    """
    Submit the job script to SLURM and extract the job ID from the output.

    Returns:
        The job ID of the submitted job.
    """
    try:
        command = (
            ["sbatch"]
            + ["--" + x for x in extra_slurm_args]
            + [
                job_script_path,
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split("\n")

        # sbatch typically outputs: "Submitted batch job JOBID"
        job_id = output_lines[-1].split()[-1]
        logging.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logging.error(f"Error submitting job: {e}")
        logging.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logging.error(f"Error parsing job ID from sbatch output: {result.stdout}")
        raise


def _get_available_gpu_types() -> list[str]:
    """Discover available GPU types by scanning scripts directory structure.

    Looks for scripts in: scripts/{gpu_type}/{agg,disagg}/*.sh
    """
    script_dir = pathlib.Path(__file__).parent / "scripts"
    gpu_types = set()

    # Scan for GPU type directories (directories that contain agg/ or disagg/)
    for gpu_dir in script_dir.iterdir():
        if not gpu_dir.is_dir():
            continue

        # Check if this directory has agg/ or disagg/ subdirectories
        has_agg = (gpu_dir / "agg").is_dir()
        has_disagg = (gpu_dir / "disagg").is_dir()

        if has_agg or has_disagg:
            gpu_types.add(gpu_dir.name)

    return sorted(list(gpu_types))


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM job scripts"
    )

    # Get available GPU types dynamically
    available_gpu_types = _get_available_gpu_types()

    # Template parameters
    parser.add_argument("--job-name", default="dynamo_setup", help="SLURM job name")
    parser.add_argument("--account", required=True, help="SLURM account")
    parser.add_argument("--model-dir", required=True, help="Model directory path")
    parser.add_argument("--config-dir", required=True, help="Config directory path")
    parser.add_argument("--container-image", required=True, help="Container image")
    parser.add_argument(
        "--time-limit", default="04:00:00", help="Time limit (HH:MM:SS)"
    )
    parser.add_argument(
        "--prefill-nodes", type=int, default=None, help="Number of prefill nodes"
    )
    parser.add_argument(
        "--decode-nodes", type=int, default=None, help="Number of decode nodes"
    )
    parser.add_argument(
        "--prefill-workers", type=int, default=None, help="Number of prefill workers"
    )
    parser.add_argument(
        "--decode-workers", type=int, default=None, help="Number of decode workers"
    )
    parser.add_argument(
        "--agg-nodes", type=int, default=None, help="Number of aggregated worker nodes"
    )
    parser.add_argument(
        "--agg-workers", type=int, default=None, help="Number of aggregated workers"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--network-interface", default="eth3", help="Network interface to use"
    )
    parser.add_argument(
        "--gpu-type",
        choices=available_gpu_types,
        default=available_gpu_types[0] if available_gpu_types else None,
        help=f"GPU type to use. Available types: {', '.join(available_gpu_types)}",
    )
    parser.add_argument(
        "--script-variant",
        type=str,
        default="default",
        help="Script variant to use (e.g., 'default', 'optim', 'decode-optim'). Defaults to 'default.sh'",
    )

    parser.add_argument(
        "--partition",
        default="batch",
        help="SLURM partition to use",
    )
    parser.add_argument(
        "--enable-multiple-frontends",
        action="store_true",
        help="Enable multiple frontend architecture with nginx load balancer",
    )
    parser.add_argument(
        "--num-additional-frontends",
        type=int,
        default=0,
        help="Number of additional frontend nodes (beyond the first frontend on node 1)",
    )

    parser.add_argument(
        "--use-init-location",
        action="store_true",
        help="Whether we use '--init-expert-locations' json files",
    )

    parser.add_argument(
        "--profiler",
        type=str,
        help="Profiler configurations. Example: "
        + '"type=vllm; isl=8192; osl=1024; concurrencies=16x2048x4096x8192; req-rate=inf"',
    )

    parser.add_argument(
        "--extra-slurm-args",
        action="append",
        default=[],
        help="Extra slurm arguments, remove the '--' prefix. Example: --extra-slurm-args dependency=afterok:<x>",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Tries to launch the job multiple times to catch transient errors",
    )

    parser.add_argument(
        "--disable-config-dump",
        action="store_false",
        dest="enable_config_dump",
        default=True,
        help="Disable dumping config to file on each node (default: config dump is enabled)",
    )

    parser.add_argument(
        "--run-in-ci",
        action="store_true",
        help="Run in CI mode - use binaries from /configs/ for nats/etcd and install dynamo wheel",
    )

    return parser.parse_args(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments and ensure aggregated and disaggregated args are mutually exclusive."""
    has_disagg_args = any(
        [
            args.prefill_nodes is not None,
            args.decode_nodes is not None,
            args.prefill_workers is not None,
            args.decode_workers is not None,
        ]
    )
    has_agg_args = any(
        [
            args.agg_nodes is not None,
            args.agg_workers is not None,
        ]
    )

    if has_disagg_args and has_agg_args:
        raise ValueError(
            "Cannot specify both aggregated (--agg-nodes, --agg-workers) and "
            "disaggregated (--prefill-nodes, --decode-nodes, --prefill-workers, --decode-workers) arguments"
        )

    if has_disagg_args:
        # Validate disaggregated args
        if args.prefill_nodes is None or args.decode_nodes is None:
            raise ValueError(
                "Disaggregated mode requires both --prefill-nodes and --decode-nodes"
            )
        if args.prefill_workers is None or args.decode_workers is None:
            raise ValueError(
                "Disaggregated mode requires both --prefill-workers and --decode-workers"
            )
        if args.prefill_nodes % args.prefill_workers != 0:
            raise ValueError(
                f"Prefill nodes ({args.prefill_nodes}) must be divisible by prefill workers ({args.prefill_workers})"
            )
        if args.decode_nodes % args.decode_workers != 0:
            raise ValueError(
                f"Decode nodes ({args.decode_nodes}) must be divisible by decode workers ({args.decode_workers})"
            )
        # Validate GPU script exists for disaggregated mode
        script_dir = pathlib.Path(__file__).parent / "scripts"
        disagg_dir = script_dir / args.gpu_type / "disagg"
        # Use script variant (defaults to "default")
        script_name = f"{args.script_variant}.sh"
        gpu_script = disagg_dir / script_name
        if not gpu_script.exists():
            raise ValueError(
                f"Disaggregated GPU script not found: {gpu_script}. Available GPU types: {', '.join(_get_available_gpu_types())}"
            )

    if has_agg_args:
        # Validate aggregated args
        if args.agg_nodes is None or args.agg_workers is None:
            raise ValueError(
                "Aggregated mode requires both --agg-nodes and --agg-workers"
            )
        if args.agg_nodes % args.agg_workers != 0:
            raise ValueError(
                f"Aggregated nodes ({args.agg_nodes}) must be divisible by aggregated workers ({args.agg_workers})"
            )
        # Validate aggregated GPU script exists
        script_dir = pathlib.Path(__file__).parent / "scripts"
        # Remove any -prefill or -decode suffix if present
        base_gpu_type = args.gpu_type.replace("-prefill", "").replace("-decode", "")
        agg_dir = script_dir / base_gpu_type / "agg"
        # Use script variant (defaults to "default")
        script_name = f"{args.script_variant}.sh"
        agg_gpu_script = agg_dir / script_name
        if not agg_gpu_script.exists():
            raise ValueError(
                f"Aggregated GPU script not found: {agg_gpu_script}. Available GPU types: {', '.join(_get_available_gpu_types())}"
            )

    if not has_disagg_args and not has_agg_args:
        raise ValueError(
            "Must specify either aggregated (--agg-nodes, --agg-workers) or "
            "disaggregated (--prefill-nodes, --decode-nodes, --prefill-workers, --decode-workers) arguments"
        )


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)

    # Validate arguments
    _validate_args(args)

    # Determine mode and set defaults
    is_aggregated = args.agg_nodes is not None

    if is_aggregated:
        agg_nodes = args.agg_nodes
        agg_workers = args.agg_workers
        prefill_nodes = 0
        decode_nodes = 0
        prefill_workers = 0
        decode_workers = 0
        total_nodes = agg_nodes
    else:
        prefill_nodes = args.prefill_nodes
        decode_nodes = args.decode_nodes
        prefill_workers = args.prefill_workers
        decode_workers = args.decode_workers
        agg_nodes = 0
        agg_workers = 0
        total_nodes = prefill_nodes + decode_nodes

    # Validation for multiple frontends
    if args.enable_multiple_frontends:
        if args.num_additional_frontends < 0:
            raise ValueError("Number of additional frontends cannot be negative")

    # parse profiler configs
    profiler_config = {}
    if args.profiler:
        for key_val_pair in args.profiler.split("; "):
            key, val = key_val_pair.split("=")
            profiler_config[key] = val

    # validate profiler configs
    if profiler_config == {} or profiler_config["type"] == "manual":
        parsable_config = ""
        profiler_config["type"] = "manual"
    elif profiler_config["type"] in ["sglang", "vllm", "gap"]:
        parsable_config = ""
        need_keys = ["isl", "osl", "concurrencies"]
        assert all([key in profiler_config for key in need_keys])
        assert profiler_config["isl"].isnumeric()
        parsable_config = f"{parsable_config} {profiler_config['isl']}"
        assert profiler_config["osl"].isnumeric()
        parsable_config = f"{parsable_config} {profiler_config['osl']}"
        assert all([x.isnumeric() for x in profiler_config["concurrencies"].split("x")])
        parsable_config = f"{parsable_config} {profiler_config['concurrencies']}"

        if profiler_config["type"] in ["sglang", "vllm"]:
            assert "req-rate" in profiler_config
            assert (
                profiler_config["req-rate"] == "inf"
                or profiler_config["req-rate"].isnumeric()
            )
            parsable_config = f"{parsable_config} {profiler_config['req-rate']}"
    else:
        assert False, profiler_config["type"]

    # Generate timestamp for log directory naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Select template based on mode
    if is_aggregated:
        template_path = "job_script_template_agg.j2"
    else:
        template_path = "job_script_template_disagg.j2"

    template_vars = {
        "job_name": args.job_name,
        "total_nodes": total_nodes,
        "account": args.account,
        "time_limit": args.time_limit,
        "prefill_nodes": prefill_nodes,
        "decode_nodes": decode_nodes,
        "prefill_workers": prefill_workers,
        "decode_workers": decode_workers,
        "agg_nodes": agg_nodes,
        "agg_workers": agg_workers,
        "is_aggregated": is_aggregated,
        "model_dir": args.model_dir,
        "config_dir": args.config_dir,
        "container_image": args.container_image,
        "gpus_per_node": args.gpus_per_node,
        "network_interface": args.network_interface,
        "gpu_type": args.gpu_type,
        "script_variant": args.script_variant,
        "partition": args.partition,
        "enable_multiple_frontends": args.enable_multiple_frontends,
        "num_additional_frontends": args.num_additional_frontends,
        "use_init_location": args.use_init_location,
        "do_profile": profiler_config["type"] != "manual",
        "profiler_type": profiler_config["type"],
        "profiler_arg": parsable_config,
        "timestamp": timestamp,
        "enable_config_dump": args.enable_config_dump,
        "run_in_ci": args.run_in_ci,
    }

    # Create temporary file for sbatch script
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        _, rendered_script = generate_job_script(
            template_path, temp_path, **template_vars
        )

        submitted_job_ids = []
        job_id = submit_job(temp_path, args.extra_slurm_args)
        submitted_job_ids.append(job_id)

        # Create log directory with new naming format IMMEDIATELY after submission
        # SLURM will write log.out/log.err to this directory when job starts
        if is_aggregated:
            log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
        else:
            log_dir_name = f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"
        log_dir_path = os.path.join("logs", log_dir_name)
        os.makedirs(log_dir_path, exist_ok=True)

        # Save rendered sbatch script
        sbatch_script_path = os.path.join(log_dir_path, "sbatch_script.sh")
        with open(sbatch_script_path, "w") as f:
            f.write(rendered_script)
        logging.info(f"Saved rendered sbatch script to {sbatch_script_path}")

        # retries logic
        if args.retries > 0:
            extra_slurm_args_without_dependencies = [
                x for x in args.extra_slurm_args if "dependency" not in x
            ]
            for _ in range(args.retries):
                dependencies = ",".join(
                    [f"afternotok:{job}" for job in submitted_job_ids]
                )
                slurm_args = extra_slurm_args_without_dependencies + [
                    f"dependency={dependencies}"
                ]
                job_id = submit_job(temp_path, slurm_args)
                submitted_job_ids.append(job_id)

                # Save script for retry job as well
                if is_aggregated:
                    retry_log_dir_name = f"{job_id}_{agg_workers}A_{timestamp}"
                else:
                    retry_log_dir_name = (
                        f"{job_id}_{prefill_workers}P_{decode_workers}D_{timestamp}"
                    )
                retry_log_dir_path = os.path.join("logs", retry_log_dir_name)
                os.makedirs(retry_log_dir_path, exist_ok=True)
                retry_sbatch_script_path = os.path.join(
                    retry_log_dir_path, "sbatch_script.sh"
                )
                with open(retry_sbatch_script_path, "w") as f:
                    f.write(rendered_script)
                logging.info(
                    f"Saved rendered sbatch script to {retry_sbatch_script_path}"
                )

        print_welcome_message(submitted_job_ids, log_dir_name)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
