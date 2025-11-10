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


def print_welcome_message(job_ids: list[str]):
    """Print a clean welcome message with job information."""

    job_id = f"<{', '.join(job_ids)}>"
    print(
        f"""
🚀 Welcome! We hope you enjoy your time on our GB200 NVL72.

Your logs for this submitted job will be available in logs/{job_id}
You can access them by running:

    cd logs/{job_id}

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

    return output_path


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
        choices=["gb200-fp8", "gb200-fp4"],
        default="gb200-fp8",
        help="GPU type to use. You can choose between gb200-fp8 and gb200-fp4.",
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

    # Validation for multiple frontends
    if args.enable_multiple_frontends:
        if args.num_additional_frontends < 0:
            raise ValueError("Number of additional frontends cannot be negative")

    total_nodes = args.prefill_nodes + args.decode_nodes

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
        "enable_multiple_frontends": args.enable_multiple_frontends,
        "num_additional_frontends": args.num_additional_frontends,
        "use_init_location": args.use_init_location,
        "do_profile": profiler_config["type"] != "manual",
        "profiler_type": profiler_config["type"],
        "profiler_arg": parsable_config,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as temp_file:
        generate_job_script(args.template, temp_file.name, **template_vars)

        submitted_job_ids = []
        job_id = submit_job(temp_file.name, args.extra_slurm_args)
        submitted_job_ids.append(job_id)
        # retries logic
        extra_slurm_args_without_dependencies = [
            x for x in args.extra_slurm_args if "dependency" not in x
        ]
        for _ in range(args.retries):
            dependencies = ",".join([f"afternotok:{job}" for job in submitted_job_ids])
            slurm_args = extra_slurm_args_without_dependencies + [
                f"dependency={dependencies}"
            ]
            job_id = submit_job(temp_file.name, slurm_args)
            submitted_job_ids.append(job_id)

        print_welcome_message(submitted_job_ids)


if __name__ == "__main__":
    main()
