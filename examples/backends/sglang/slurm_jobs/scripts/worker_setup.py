# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker setup script for Slurm nodes.
This script will be running on the prefill and decode nodes, and will be called by the
benchmark_dynamo.sh script.

The script will:
- Setup the environment
- Generate the python3 command to run the prefill or decode worker
- Start dynamo (or sglang)
- Monitor the GPU utilization
"""

import argparse
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import requests

# Network configurations
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222
DIST_INIT_PORT = 29500
ETCD_LISTEN_ADDR = "http://0.0.0.0"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s| %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_gpu_utilization(log_file: Path) -> None:
    """
    Log GPU utilization for all GPUs in the node.
    Format: utilization.gpu [%] x y z
    """
    util_script = Path(__file__).parent / "monitor_gpu_utilization.sh"
    util_process = run_command(
        f"bash {util_script}",
        background=True,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
    )
    if not util_process:
        logging.warning("Failed to start GPU utilization monitoring")
    else:
        logging.info("Started GPU utilization monitoring in the background")


def check_etcd_health(etcd_url: str) -> bool:
    """Check if etcd is healthy"""
    health_url = f"{etcd_url}/health"
    try:
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_for_etcd(etcd_url: str, max_retries: int = 1000) -> bool:
    """Wait for etcd to be ready"""
    logging.info(f"Waiting for etcd to be ready on {etcd_url}...")

    for attempt in range(max_retries):
        try:
            if check_etcd_health(etcd_url):
                logging.info("Etcd is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logging.info(
            f"Etcd not ready yet, retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})"
        )
        time.sleep(2)

    logging.error("Etcd failed to become ready within the timeout period")
    return False


def run_command(
    cmd: str, background: bool = False, shell: bool = True, stdout=None, stderr=None
):
    """
    Run a command either in background or foreground.

    Args:
        cmd: Command to run
        background: If True, run in background and return Popen object. If False, wait for
            completion and return exit code.
        shell: Whether to run command through shell

    Returns:
        If background=True: subprocess.Popen
        If background=False: int (exit code)
    """
    logging.info(f"Running command (background={background}, shell={shell}): {cmd}")
    if background:
        process = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=stdout if stdout else subprocess.PIPE,
            stderr=stderr if stderr else subprocess.PIPE,
        )  # noqa: S603
        return process
    else:
        result = subprocess.run(cmd, shell=shell, check=True)  # noqa: S603
        return result.returncode


def _parse_command_line_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Worker setup script for Dynamo distributed training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--leader_ip",
        type=str,
        required=False,
        help="IP address of the leader node for this worker group",
    )
    parser.add_argument(
        "--master_ip",
        type=str,
        required=True,
        help="IP address of the master node (first prefill node) for NATS/ETCD",
    )
    parser.add_argument(
        "--worker_idx",
        type=int,
        required=False,
        help="Index of the worker group (0-based)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=False,
        help="Local rank within the worker group (0 for leader)",
    )
    parser.add_argument(
        "--nodes_per_worker",
        type=int,
        required=False,
        help="Number of nodes per worker",
    )
    parser.add_argument(
        "--worker_type",
        choices=["decode", "prefill", "frontend", "nginx"],
        required=True,
        help="Type of worker to run",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=8,
        help="Number of GPUs per node (default: 8)",
    )
    parser.add_argument(
        "--gpu_utilization_log",
        type=str,
        default=None,
        help="File to log GPU utilization (default: None)",
    )

    parser.add_argument(
        "--gpu_type",
        type=str,
        choices=["gb200-fp8", "gb200-fp4"],
        default="gb200-fp8",
        help="Type of GPU to use. You can choose between gb200-fp8 and gb200-fp4.",
    )

    parser.add_argument(
        "--nginx_config",
        type=str,
        help="Path to nginx configuration file (required for nginx worker type)",
    )

    parser.add_argument(
        "--multiple-frontends-enabled",
        action="store_true",
        help="Whether multiple frontend architecture is enabled (affects infrastructure setup)",
    )

    parser.add_argument(
        "--use_init_locations",
        action="store_true",
        help="Whether we add --init-expert-locations to launch commands",
    )

    return parser.parse_args(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.worker_type in ["prefill", "decode"]:
        if args.worker_idx is None or args.worker_idx < 0:
            raise ValueError(
                "Worker index must be provided and non-negative for prefill/decode"
            )

    if args.worker_type in ["prefill", "decode"]:
        if args.local_rank is None or args.local_rank < 0:
            raise ValueError("Local rank must be non-negative")

        if args.nodes_per_worker is None or args.nodes_per_worker < 1:
            raise ValueError("Nodes per worker must be at least 1")

        if args.gpus_per_node < 1:
            raise ValueError("GPUs per node must be at least 1")

        if args.local_rank >= args.nodes_per_worker:
            raise ValueError(
                f"Local rank ({args.local_rank}) must be less than nodes per worker ({args.nodes_per_worker})"
            )

    # Validate nginx-specific arguments
    if args.worker_type == "nginx" and not args.nginx_config:
        raise ValueError("--nginx_config is required for nginx worker type")


def setup_env_vars_for_gpu_script(
    host_ip: str,
    local_rank: int,
    total_gpus: int,
    total_nodes: int,
    port: int = DIST_INIT_PORT,
    use_init_locations: bool = True,
):
    """Setup environment variables required by GPU scripts (gb200-fp8.sh)"""
    os.environ["HOST_IP_MACHINE"] = host_ip
    os.environ["PORT"] = str(port)
    os.environ["TOTAL_GPUS"] = str(total_gpus)
    os.environ["RANK"] = str(local_rank)
    os.environ["TOTAL_NODES"] = str(total_nodes)
    os.environ["USE_INIT_LOCATIONS"] = str(use_init_locations)

    logging.info(f"Set HOST_IP: {host_ip}")
    logging.info(f"Set PORT: {port}")
    logging.info(f"Set TOTAL_GPUS: {total_gpus}")
    logging.info(f"Set RANK: {local_rank}")
    logging.info(f"Set TOTAL_NODES: {total_nodes}")
    logging.info(f"Set USE_INIT_LOCATIONS: {use_init_locations}")


def get_gpu_command(worker_type: str, gpu_type: str) -> str:
    """Generate command to run the appropriate GPU script"""
    script_name = f"{gpu_type}.sh"
    script_path = Path(__file__).parent / script_name
    mode = worker_type  # "prefill" or "decode"

    return f"bash {script_path} {mode}"


def setup_head_prefill_node(prefill_host_ip: str) -> None:
    """
    Setup NATS, etcd, ingress, and http servers on the prefill host node.
    """
    logging.info(f"Starting nats server on node {prefill_host_ip}")

    nats_process = run_command("nats-server -js", background=True)
    if not nats_process:
        raise RuntimeError("Failed to start nats-server")

    logging.info(f"Starting etcd server on node {prefill_host_ip}")
    etcd_cmd = (
        f"etcd --listen-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
        f"--advertise-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
        f"--listen-peer-urls {ETCD_LISTEN_ADDR}:{ETCD_PEER_PORT} "
        f"--initial-cluster default=http://{prefill_host_ip}:{ETCD_PEER_PORT}"
    )

    etcd_process = run_command(etcd_cmd, background=True)
    if not etcd_process:
        raise RuntimeError("Failed to start etcd")


def setup_nginx_worker(master_ip: str, nginx_config: str) -> int:
    """Setup nginx load balancer"""
    logging.info("Setting up nginx load balancer")

    if not nginx_config or not os.path.exists(nginx_config):
        raise ValueError(f"Nginx config file not found: {nginx_config}")

    nginx_cmd = f"apt-get update && apt-get install -y nginx && nginx -c {nginx_config} && sleep 86400"
    return run_command(nginx_cmd)


def setup_frontend_worker(worker_idx: int, master_ip: str) -> int:
    """Setup a frontend worker"""
    logging.info(f"Setting up frontend worker {worker_idx}")

    # First frontend (worker_idx 0) also sets up NATS/ETCD
    if worker_idx == 0:
        setup_head_prefill_node(master_ip)
    else:
        logging.info(f"Setting up additional frontend worker {worker_idx}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # All frontends run the ingress server
    frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
    return run_command(frontend_cmd)


def setup_prefill_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    use_init_locations: bool = True,
) -> int:
    """
    Setup the prefill worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    # Only setup infrastructure in traditional mode (not multiple frontends)
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip)
    else:
        logging.info(f"Setting up prefill worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=use_init_locations,
    )

    # Use appropriate GPU script instead of generating command directly
    cmd_to_run = get_gpu_command("prefill", gpu_type)
    return run_command(cmd_to_run)


def setup_decode_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    use_init_locations: bool = True,
) -> int:
    """
    Setup the decode worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    logging.info(f"Setting up decode worker {worker_idx}, local rank {local_rank}")

    if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
        raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=use_init_locations,
    )

    # Use appropriate GPU script instead of generating command directly
    cmd_to_run = get_gpu_command("decode", gpu_type)
    return run_command(cmd_to_run)


def setup_env(master_ip: str):
    nats_server = f"nats://{master_ip}:{NATS_PORT}"
    etcd_endpoints = f"http://{master_ip}:{ETCD_CLIENT_PORT}"

    os.environ["NATS_SERVER"] = nats_server
    os.environ["ETCD_ENDPOINTS"] = etcd_endpoints

    logging.info(f"set NATS_SERVER: {nats_server}")
    logging.info(f"set ETCD_ENDPOINTS: {etcd_endpoints}")


def main(input_args: list[str] | None = None):
    setup_logging()
    args = _parse_command_line_args(input_args)
    _validate_args(args)

    if args.gpu_utilization_log:
        log_gpu_utilization(args.gpu_utilization_log)

    logging.info(f"{args.worker_type.capitalize()} worker setup started")
    logging.info(f"Hostname: {socket.gethostname()}")
    logging.info(f"Worker type: {args.worker_type}")
    logging.info(f"Worker index: {args.worker_idx}")
    logging.info(f"Local rank: {args.local_rank}")
    logging.info(f"Leader IP: {args.leader_ip}")
    logging.info(f"Master IP: {args.master_ip}")
    logging.info(f"Nodes per worker: {args.nodes_per_worker}")
    logging.info(f"Use init locations?: {args.use_init_locations}")

    setup_env(args.master_ip)

    if args.worker_type == "nginx":
        if not args.nginx_config:
            raise ValueError("--nginx_config is required for nginx worker type")
        setup_nginx_worker(args.master_ip, args.nginx_config)
    elif args.worker_type == "frontend":
        setup_frontend_worker(args.worker_idx, args.master_ip)
    elif args.worker_type == "prefill":
        setup_prefill_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpus_per_node,
            args.gpu_type,
            args.multiple_frontends_enabled,
            args.use_init_locations,
        )
    elif args.worker_type == "decode":
        setup_decode_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpus_per_node,
            args.gpu_type,
            args.use_init_locations,
        )

    logging.info(f"{args.worker_type.capitalize()} worker setup complete")


if __name__ == "__main__":
    main()
