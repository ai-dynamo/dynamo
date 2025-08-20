# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import tempfile

from jinja2 import Template


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

    return output_path


def submit_job(job_script_path):
    """
    Submit the job script to SLURM and extract the job ID from the output.

    Returns:
        The job ID of the submitted job.
    """
    try:
        result = subprocess.run(
            ["sbatch", job_script_path], capture_output=True, text=True, check=True
        )
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


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM job scripts"
    )
    parser.add_argument(
        "--template", required=True, help="Path to Jinja2 template file"
    )

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
        "--prefill-nodes", type=int, default=2, help="Number of prefill nodes"
    )
    parser.add_argument(
        "--decode-nodes", type=int, default=2, help="Number of decode nodes"
    )
    parser.add_argument(
        "--prefill-workers", type=int, default=1, help="Number of prefill workers"
    )
    parser.add_argument(
        "--decode-workers", type=int, default=1, help="Number of decode workers"
    )
    parser.add_argument(
        "--gpus-per-node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--network-interface", default="eth3", help="Network interface to use"
    )
    parser.add_argument(
        "--gpu-type",
        choices=["h100", "gb200-fp8"],
        default="h100",
        help="GPU type to use",
    )

    parser.add_argument(
        "--partition",
        default="batch",
        help="SLURM partition to use",
    )
    return parser.parse_args(args)


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)

    # Validation
    if args.prefill_nodes % args.prefill_workers != 0:
        raise ValueError(
            f"Prefill nodes ({args.prefill_nodes}) must be divisible by prefill workers ({args.prefill_workers})"
        )

    if args.decode_nodes % args.decode_workers != 0:
        raise ValueError(
            f"Decode nodes ({args.decode_nodes}) must be divisible by decode workers ({args.decode_workers})"
        )

    total_nodes = args.prefill_nodes + args.decode_nodes
    template_vars = {
        "job_name": args.job_name,
        "total_nodes": total_nodes,
        "account": args.account,
        "time_limit": args.time_limit,
        "prefill_nodes": args.prefill_nodes,
        "decode_nodes": args.decode_nodes,
        "prefill_workers": args.prefill_workers,
        "decode_workers": args.decode_workers,
        "model_dir": args.model_dir,
        "config_dir": args.config_dir,
        "container_image": args.container_image,
        "gpus_per_node": args.gpus_per_node,
        "network_interface": args.network_interface,
        "gpu_type": args.gpu_type,
        "partition": args.partition,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as temp_file:
        generate_job_script(args.template, temp_file.name, **template_vars)
        job_id = submit_job(temp_file.name)
        logging.info(f"Job logs will be available in: logs/{job_id}/")


if __name__ == "__main__":
    main()
