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
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import List, Optional

from jinja2 import Template
from pydantic import BaseModel

FLUSH_CACHE_PORT = 9001

logger = logging.getLogger("slurm_job")


class JobInfo(BaseModel):
    """
    Pydantic model for job information.

    The job information is stored in a JSON file and is used as a form of
    communication between the main script running on the head node
    and the job script running on the prefill and decode nodes.
    """

    slurm_job_id: str
    prefill_host_ip: str
    decode_host_ip: str
    nodes: List[str]
    enroot_args: str
    prefill_nodes: int
    decode_nodes: int


class SlurmStep(BaseModel):
    """An execution step in the interactive workflow."""

    name: str
    job_id: str
    description: str
    command: Optional[tuple[str, ...]] = None
    print_command: bool = False

    def log(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")

    def describe(self) -> None:
        self.log(self.description)
        if self.print_command and self.command:
            self.log(f"    {' '.join(self.command)}")

    def execute(self) -> None:
        if self.command:
            log_dir = os.path.join(
                os.path.dirname(__file__), "logs", self.job_id, self.name
            )
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, f"{self.name}.log")
            err_file = os.path.join(log_dir, f"{self.name}.err")

            self.log(f"Executing command: {' '.join(self.command)}")
            self.log(f"Logging stdout to: {log_file}")
            self.log(f"Logging stderr to: {err_file}")

            result = subprocess.run(
                self.command,
                stdout=open(log_file, "w"),
                stderr=open(err_file, "w"),
                text=True,
                check=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Command failed with return code {result.returncode}"
                )

            self.log("Command completed successfully")
        else:
            self.log("Nothing to execute.")


class WaitForAllocation(SlurmStep):
    """Wait for the nodes to be allocated."""

    wait_time: int = 10  # seconds

    def execute(self) -> None:
        while True:
            result = subprocess.run(
                ["squeue", "-j", self.job_id, "-h", "-o", "%t"],
                capture_output=True,
                text=True,
                check=True,
            )
            output_lines = result.stdout.strip().split("\n")
            if len(output_lines) < 1:
                raise RuntimeError("No output from squeue. Exiting.")
            elif output_lines[0] == "R":
                self.log("Nodes allocated. Exiting.")
                break
            elif output_lines[0].startswith("P"):
                self.log("Waiting for nodes to be allocated...")
            elif output_lines[0].startswith("C"):
                raise RuntimeError("Job was cancelled")
            else:
                raise RuntimeError(f"Unknown state: {output_lines[0]}")
            time.sleep(self.wait_time)


class WaitForSetup(SlurmStep):
    """Wait for the setup to complete and for GPUs to be idle."""

    nodes: List[str]
    prefill_nodes: int
    decode_nodes: int
    retries: int = 5 * 60  # 5 minutes

    def _wait_for_file(self, filename: str) -> None:
        self.log(f"Waiting for file {filename} to be created...")
        for _ in range(self.retries):
            if os.path.exists(filename):
                self.log(f"File {filename} exists")
                break
            else:
                time.sleep(1)
        else:
            raise RuntimeError(
                f"File {filename} does not exist after {self.retries} retries"
            )

    def _wait_for_idle_gpu(self, node: str, worker_type: str) -> None:
        filename = f"{node}_{worker_type}_gpu_utilization.log"
        self.log(f"Waiting for GPUs to be idle on {worker_type} node {node}")

        log_path = os.path.join(
            os.path.dirname(__file__), "logs", self.job_id, filename
        )
        self._wait_for_file(log_path)

        with open(log_path, "r") as f:
            last_line = f.readlines()[-1]
        utilization = last_line.split("utilization.gpu [%]")[-1].strip()
        for _ in range(self.retries):
            if all(float(i) == 0 for i in utilization.split()):
                self.log(f"GPU is idle on {node}")
                break
            else:
                time.sleep(1)
        else:
            raise RuntimeError(f"GPU is no idle on {node}")
        self.log(f"GPUs are idle on {node}")

    def _wait_for_complete_setup(self, node: str, worker_type: str) -> None:
        # NOTE: for some reason the sglang setup is saved in the stderr file
        # TODO: find the root cause and fix it
        setup_complete_phrases = [
            "dummy health check server scheduled",
            "request handler initialized",
        ]
        filename = f"{node}_{worker_type}.err"
        self.log(f"Waiting for setup to complete on {worker_type} node {node}")

        log_path = os.path.join(
            os.path.dirname(__file__), "logs", self.job_id, filename
        )
        self._wait_for_file(log_path)

        for _ in range(self.retries):
            with open(log_path, "r") as f:
                for line in f:
                    if any(
                        phrase.lower() in line.lower()
                        for phrase in setup_complete_phrases
                    ):
                        self.log(f"Setup complete on {worker_type} node {node}")
                        break
                else:
                    time.sleep(1)
                    continue
                break
        else:
            raise RuntimeError(f"Setup is not complete on {worker_type} node {node}")
        self.log(f"{worker_type} node {node} is ready")

    def execute(self) -> None:
        self.log("Waiting for setup to complete and for GPUs to be idle")
        for worker_type, node in zip(
            self.prefill_nodes * ["prefill"] + self.decode_nodes * ["decode"],
            self.nodes,
        ):
            self._wait_for_complete_setup(node, worker_type)
            self._wait_for_idle_gpu(node, worker_type)

        self.log("Setup complete and GPUs are idle.")


class SlurmConfig(BaseModel):
    """Configuration class for SLURM job management."""

    job_info: JobInfo
    timeout: int = 3600  # seconds # TODO implement

    def create_execution_steps(self) -> List[SlurmStep]:
        """Create the sequence of SLURM steps for the interactive workflow."""
        return [
            WaitForAllocation(
                name="wait_for_allocation",
                job_id=self.job_info.slurm_job_id,
                description="Wait for the nodes to be allocated",
            ),
            WaitForSetup(
                name="wait_for_setup",
                job_id=self.job_info.slurm_job_id,
                description="Wait for setup to complete and for GPUs to be idle",
                nodes=self.job_info.nodes,
                prefill_nodes=self.job_info.prefill_nodes,
                decode_nodes=self.job_info.decode_nodes,
            ),
            SlurmStep(
                name="warmup",
                job_id=self.job_info.slurm_job_id,
                description="Run warmup on the prefill host",
                command=(
                    "srun",
                    *self.job_info.enroot_args.split(),
                    "--jobid",
                    self.job_info.slurm_job_id,
                    "-w",
                    self.job_info.nodes[0],
                    "--overlap",
                    "bash",
                    "utils/bench.sh",
                    self.job_info.prefill_host_ip,
                    "--type",
                    "warmup",
                ),
                print_command=True,
            ),
            SlurmStep(
                name="flush_cache",
                job_id=self.job_info.slurm_job_id,
                description="Call the flush cache endpoint on the prefill host",
                command=(
                    "curl",
                    "-X",
                    "POST",
                    f"http://{self.job_info.prefill_host_ip}:{FLUSH_CACHE_PORT}/flush_cache",
                ),
                print_command=True,
            ),
        ]

    def create_cleanup_steps(self) -> List[SlurmStep]:
        """Create the cleanup steps for the interactive workflow."""
        return [
            SlurmStep(
                name="cancel_job",
                job_id=self.job_info.slurm_job_id,
                description="Cancel the job:",
                command=("scancel", self.job_info.slurm_job_id),
                print_command=True,
            ),
        ]

    def describe_execution_steps(self) -> None:
        logger.info("--------------------------------")
        logger.info("The following steps will be executed:")
        for step_id, step in enumerate(self.create_execution_steps()):
            logger.info(f"Step {step_id+1}")
            logger.info(f"    Name: {step.name}")
            logger.info(f"    Description: {step.description}")

    def describe_cleanup_steps(self) -> None:
        logger.info("--------------------------------")
        logger.info("The following steps will be executed when the job is done:")
        for step_id, step in enumerate(self.create_cleanup_steps()):
            logger.info(f"Step {step_id+1}")
            logger.info(f"    Name: {step.name}")
            logger.info(f"    Description: {step.description}")
        logger.info("--------------------------------")

    def execute_interactive_workflow(self) -> None:
        """Execute the complete interactive workflow."""
        steps, cleanup_steps = None, None
        try:
            logger.info("Entering interactive mode")
            logger.info(
                "If the connection is lost, make sure to cancel the job manually:"
            )
            logger.info(f"    scancel {self.job_info.slurm_job_id}")

            steps = self.create_execution_steps()
            cleanup_steps = self.create_cleanup_steps()

            self.describe_execution_steps()
            self.describe_cleanup_steps()
            for step in steps:
                step.execute()
        finally:
            logger.info("--------------------------------")
            logger.info("Running cleanup steps:")
            if not cleanup_steps:
                cleanup_steps = self.create_cleanup_steps()
            for step in cleanup_steps:
                step.execute()
            logger.info("--------------------------------")
            logger.info("Interactive mode completed")


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
        logger.info(f"Job submitted successfully with ID: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        logger.error(f"Error submitting job: {e}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except (IndexError, ValueError):
        logger.error(f"Error parsing job ID from sbatch output: {result.stdout}")
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
        "--gpus-per-node", type=int, default=8, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--network-interface", default="eth3", help="Network interface to use"
    )
    parser.add_argument(
        "--gpu-type", choices=["h100", "gb200"], default="h100", help="GPU type to use"
    )
    parser.add_argument(
        "--use-sglang-commands",
        action="store_true",
        default=False,
        help="Use SGLang commands instead of Dynamo",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run the job interactively",
    )
    return parser.parse_args(args)


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)

    total_nodes = args.prefill_nodes + args.decode_nodes
    template_vars = {
        "job_name": args.job_name,
        "total_nodes": total_nodes,
        "account": args.account,
        "time_limit": args.time_limit,
        "prefill_nodes": args.prefill_nodes,
        "decode_nodes": args.decode_nodes,
        "model_dir": args.model_dir,
        "config_dir": args.config_dir,
        "container_image": args.container_image,
        "gpus_per_node": args.gpus_per_node,
        "network_interface": args.network_interface,
        "gpu_type": args.gpu_type,
        "use_sglang_commands": args.use_sglang_commands,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh") as temp_file:
        generate_job_script(args.template, temp_file.name, **template_vars)
        job_id = submit_job(temp_file.name)
        logger.info(f"Job logs will be available in: logs/{job_id}/")
        logger.info("To cancel the job, run:")
        logger.info(f"    scancel {job_id}")
        logger.info("")

    if args.interactive:
        WaitForAllocation(
            name="wait_for_allocation",
            job_id=job_id,
            description="Wait for the nodes to be allocated",
        ).execute()

        job_info_path = os.path.join(
            os.path.dirname(__file__), "logs", job_id, "job_info.json"
        )
        if not os.path.exists(job_info_path):
            raise FileNotFoundError(f"Job info file not found at {job_info_path}")

        with open(job_info_path) as f:
            job_info_data = json.load(f)
        job_info = JobInfo(
            **job_info_data,
            prefill_nodes=args.prefill_nodes,
            decode_nodes=args.decode_nodes,
        )
        logger.info(f"Job info loaded from {job_info_path}: {job_info}")

        slurm_config = SlurmConfig(
            job_info=job_info,
        )

        slurm_config.execute_interactive_workflow()


if __name__ == "__main__":
    main()
