# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        choices=["decode", "prefill", "frontend", "nginx", "aggregated"],
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
        default="gb200-fp8",
        help="Type of GPU to use (script will be validated at runtime)",
    )
    parser.add_argument(
        "--script-variant",
        type=str,
        default="default",
        help="Script variant to use (e.g., 'default', 'optim', 'decode-optim'). Defaults to 'default'",
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

    parser.add_argument(
        "--dump-config-path",
        type=str,
        default=None,
        help="Path to dump config file (e.g., /logs/node_config.json)",
    )

    parser.add_argument(
        "--run-in-ci",
        action="store_true",
        help="Run in CI mode - use binaries from /configs/ for nats/etcd",
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
    dump_config_path: str | None = None,
    run_in_ci: bool = False,
):
    """Setup environment variables required by GPU scripts (gb200-fp8.sh)"""
    os.environ["HOST_IP_MACHINE"] = host_ip
    os.environ["PORT"] = str(port)
    os.environ["TOTAL_GPUS"] = str(total_gpus)
    os.environ["RANK"] = str(local_rank)
    os.environ["TOTAL_NODES"] = str(total_nodes)
    os.environ["USE_INIT_LOCATIONS"] = str(use_init_locations)
    os.environ["RUN_IN_CI"] = str(run_in_ci)
    if dump_config_path:
        os.environ["DUMP_CONFIG_PATH"] = dump_config_path
    else:
        os.environ.pop("DUMP_CONFIG_PATH", None)

    logging.info(f"Set HOST_IP: {host_ip}")
    logging.info(f"Set PORT: {port}")
    logging.info(f"Set TOTAL_GPUS: {total_gpus}")
    logging.info(f"Set RANK: {local_rank}")
    logging.info(f"Set TOTAL_NODES: {total_nodes}")
    logging.info(f"Set USE_INIT_LOCATIONS: {use_init_locations}")
    logging.info(f"Set RUN_IN_CI: {run_in_ci}")
    if dump_config_path:
        logging.info(f"Set DUMP_CONFIG_PATH: {dump_config_path}")


def get_gpu_command(
    worker_type: str, gpu_type: str, script_variant: str = "default"
) -> str:
    """Generate command to run the appropriate GPU script.

    Scripts are organized as: scripts/{gpu_type}/{agg,disagg}/{script_variant}.sh
    """
    script_base = Path(__file__).parent
    script_name = f"{script_variant}.sh"

    if worker_type == "aggregated":
        # Remove any -prefill or -decode suffix if present
        base_gpu_type = gpu_type.replace("-prefill", "").replace("-decode", "")
        script_path = script_base / base_gpu_type / "agg" / script_name
        if not script_path.exists():
            raise ValueError(f"Aggregated GPU script not found: {script_path}")
        return f"bash {script_path}"
    else:
        # Disaggregated mode: scripts/{gpu_type}/disagg/{script_variant}.sh {prefill|decode}
        script_path = script_base / gpu_type / "disagg" / script_name
        if not script_path.exists():
            raise ValueError(f"Disaggregated GPU script not found: {script_path}")
        mode = worker_type  # "prefill" or "decode"
        return f"bash {script_path} {mode}"


def setup_head_prefill_node(prefill_host_ip: str, run_in_ci: bool = False) -> None:
    """
    Setup NATS, etcd, ingress, and http servers on the prefill host node.
    """
    if run_in_ci:
        logging.info(
            f"Starting nats server on node {prefill_host_ip} (CI mode - using /configs/nats-server)"
        )
        nats_cmd = "/configs/nats-server -js"
    else:
        logging.info(f"Starting nats server on node {prefill_host_ip}")
        nats_cmd = "nats-server -js"

    nats_process = run_command(nats_cmd, background=True)
    if not nats_process:
        raise RuntimeError("Failed to start nats-server")

    if run_in_ci:
        logging.info(
            f"Starting etcd server on node {prefill_host_ip} (CI mode - using /configs/etcd)"
        )
        etcd_binary = "/configs/etcd"
    else:
        logging.info(f"Starting etcd server on node {prefill_host_ip}")
        etcd_binary = "etcd"

    etcd_cmd = (
        f"{etcd_binary} --listen-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
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


def setup_frontend_worker(
    worker_idx: int, master_ip: str, run_in_ci: bool = False
) -> int:
    """Setup a frontend worker"""
    logging.info(f"Setting up frontend worker {worker_idx}")

    # First frontend (worker_idx 0) also sets up NATS/ETCD
    if worker_idx == 0:
        setup_head_prefill_node(master_ip, run_in_ci)
    else:
        logging.info(f"Setting up additional frontend worker {worker_idx}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # All frontends run the ingress server
    frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
    if run_in_ci:
        frontend_cmd = "python3 -m pip install /configs/ai_dynamo_runtime-0.7.0-cp310-abi3-manylinux_2_28_aarch64.whl && python3 -m pip install /configs/ai_dynamo-0.7.0-py3-none-any.whl && python3 -m dynamo.frontend --http-port=8000"
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
    dump_config_path: str | None = None,
    script_variant: str = "default",
    run_in_ci: bool = False,
) -> int:
    """
    Setup the prefill worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    # Only setup infrastructure in traditional mode (not multiple frontends)
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip, run_in_ci)
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
        dump_config_path=dump_config_path,
        run_in_ci=run_in_ci,
    )

    # Use appropriate GPU script instead of generating command directly
    cmd_to_run = get_gpu_command("prefill", gpu_type, script_variant)
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
    dump_config_path: str | None = None,
    script_variant: str = "default",
    run_in_ci: bool = False,
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
        dump_config_path=dump_config_path,
        run_in_ci=run_in_ci,
    )

    # Use appropriate GPU script instead of generating command directly
    cmd_to_run = get_gpu_command("decode", gpu_type, script_variant)
    return run_command(cmd_to_run)


def setup_aggregated_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    multiple_frontends_enabled: bool = False,
    dump_config_path: str | None = None,
    script_variant: str = "default",
    run_in_ci: bool = False,
) -> int:
    """
    Setup the aggregated worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    # Only setup infrastructure in traditional mode (not multiple frontends) on first worker, first node
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip, run_in_ci)
    else:
        logging.info(
            f"Setting up aggregated worker {worker_idx}, local rank {local_rank}"
        )
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    # Aggregated mode doesn't use init locations
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=False,
        dump_config_path=dump_config_path,
        run_in_ci=run_in_ci,
    )

    # Use appropriate aggregated GPU script
    cmd_to_run = get_gpu_command("aggregated", gpu_type, script_variant)
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
    logging.info(f"Run in CI mode?: {args.run_in_ci}")
    logging.info(f"Use init locations?: {args.use_init_locations}")

    setup_env(args.master_ip)

    if args.worker_type == "nginx":
        if not args.nginx_config:
            raise ValueError("--nginx_config is required for nginx worker type")
        setup_nginx_worker(args.master_ip, args.nginx_config)
    elif args.worker_type == "frontend":
        setup_frontend_worker(args.worker_idx, args.master_ip, args.run_in_ci)
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
            args.dump_config_path,
            args.script_variant,
            args.run_in_ci,
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
            args.dump_config_path,
            args.script_variant,
            args.run_in_ci,
        )
    elif args.worker_type == "aggregated":
        setup_aggregated_worker(
            args.worker_idx,
            args.local_rank,
            args.leader_ip,
            args.master_ip,
            args.nodes_per_worker,
            args.gpus_per_node,
            args.gpu_type,
            args.multiple_frontends_enabled,
            args.dump_config_path,
            args.script_variant,
            args.run_in_ci,
        )

    logging.info(f"{args.worker_type.capitalize()} worker setup complete")


if __name__ == "__main__":
    main()
