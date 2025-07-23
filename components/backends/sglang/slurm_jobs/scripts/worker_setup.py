# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker setup script for Slurm nodes.
This script will be running on the prefill and decode nodes, and will be called by the
benchmark_dynamo.sh script.

The script will:
- Setup the environment
- Update the YAML config file
- Start Dynamo graphs.disagg service
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
        "--prefill_host_ip",
        type=str,
        required=True,
        help="IP address of the prefill host node",
    )
    parser.add_argument(
        "--decode_host_ip",
        type=str,
        required=True,
        help="IP address of the decode host node",
    )
    parser.add_argument(
        "--rank",
        type=int,
        required=True,
        help="Rank of the current node (0 for host node)",
    )
    parser.add_argument(
        "--total_nodes",
        type=int,
        required=True,
        help="Total number of nodes in the cluster",
    )
    parser.add_argument(
        "--worker_type",
        choices=["decode", "prefill"],
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

    return parser.parse_args(args)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.rank < 0:
        raise ValueError("Rank must be non-negative")

    if args.total_nodes < 1:
        raise ValueError("Total nodes must be at least 1")

    if args.gpus_per_node < 1:
        raise ValueError("GPUs per node must be at least 1")


def setup_prefill_node(
    rank: int, prefill_host_ip: str, total_nodes: int, total_gpus: int
) -> int:
    """
    Setup the prefill node.
    """
    if rank == 0:
        logging.info(f"Setting up host prefill node: {rank}")
        logging.info(f"Starting nats server on node {rank} with IP {prefill_host_ip}")

        nats_process = run_command("nats-server -js", background=True)
        if not nats_process:
            raise RuntimeError("Failed to start nats-server")

        etcd_cmd = (
            f"etcd --listen-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
            f"--advertise-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
            f"--listen-peer-urls {ETCD_LISTEN_ADDR}:{ETCD_PEER_PORT} "
            f"--initial-cluster default=http://{prefill_host_ip}:{ETCD_PEER_PORT}"
        )

        etcd_process = run_command(etcd_cmd, background=True)
        if not etcd_process:
            raise RuntimeError("Failed to start etcd")

        ingress_process = run_command("dynamo run in=http out=dyn", background=True)
        if not ingress_process:
            raise RuntimeError("Failed to start ingress")

    else:
        logging.info(f"Setting up child prefill node: {rank}")
        if not wait_for_etcd(f"http://{prefill_host_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # NOTE: This implements the example in examples/sglang/dsr1-wideep.md
    # For other examples, the command might have to be modified.
    dynamo_cmd = (
        f"python3 -m dynamo.sglang.worker "
        "--model-path /model/ "
        "--served-model-name deepseek-ai/DeepSeek-R1 "
        "--skip-tokenizer-init "
        "--disaggregation-mode prefill "
        "--disaggregation-transfer-backend nixl "
        "--disaggregation-bootstrap-port 30001 "
        f"--dist-init-addr {prefill_host_ip}:{DIST_INIT_PORT} "
        f"--nnodes {total_nodes} "
        f"--node-rank {rank} "
        f"--tp-size {total_gpus} "
        f"--dp-size {total_gpus} "
        "--enable-dp-attention "
        "--decode-log-interval 1 "
        "--enable-deepep-moe "
        "--page-size 1 "
        "--trust-remote-code "
        "--moe-dense-tp-size 1 "
        "--enable-dp-lm-head "
        "--disable-radix-cache "
        "--watchdog-timeout 1000000 "
        "--enable-two-batch-overlap "
        "--deepep-mode normal "
        "--mem-fraction-static 0.85 "
        "--deepep-config /configs/deepep.json "
        "--ep-num-redundant-experts 32 "
        "--ep-dispatch-algorithm dynamic "
        "--eplb-algorithm deepseek "
    )
    return run_command(dynamo_cmd)


def setup_decode_node(
    rank: int,
    decode_host_ip: str,
    prefill_host_ip: str,
    total_nodes: int,
    total_gpus: int,
) -> int:
    """
    Setup the decode node.
    """
    logging.info(f"Setting up child decode node: {rank}")

    if not wait_for_etcd(f"http://{prefill_host_ip}:{ETCD_CLIENT_PORT}"):
        raise RuntimeError("Failed to connect to etcd")

    dynamo_cmd = (
        "python3 -m dynamo.sglang.decode_worker "
        "--model-path /model/ "
        "--served-model-name deepseek-ai/DeepSeek-R1 "
        "--skip-tokenizer-init "
        "--disaggregation-mode decode "
        "--disaggregation-transfer-backend nixl "
        "--disaggregation-bootstrap-port 30001 "
        f"--dist-init-addr {decode_host_ip}:{DIST_INIT_PORT} "
        f"--nnodes {total_nodes} "
        f"--node-rank {rank} "
        f"--tp-size {total_gpus} "
        f"--dp-size {total_gpus} "
        "--enable-dp-attention "
        "--decode-log-interval 1 "
        "--enable-deepep-moe "
        "--page-size 1 "
        "--trust-remote-code "
        "--moe-dense-tp-size 1 "
        "--enable-dp-lm-head "
        "--disable-radix-cache "
        "--watchdog-timeout 1000000 "
        "--enable-two-batch-overlap "
        "--deepep-mode low_latency "
        "--mem-fraction-static 0.835 "
        "--ep-num-redundant-experts 32 "
        "--cuda-graph-bs 256 "
    )

    return run_command(dynamo_cmd)


def setup_env(prefill_host_ip: str):
    nats_server = f"nats://{prefill_host_ip}:{NATS_PORT}"
    etcd_endpoints = f"http://{prefill_host_ip}:{ETCD_CLIENT_PORT}"

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

    logging.info(f"{args.worker_type.capitalize()} node setup started")
    logging.info(f"Hostname: {socket.gethostname()}")
    logging.info(f"Prefill host IP: {args.prefill_host_ip}")
    logging.info(f"Decode host IP: {args.decode_host_ip}")
    logging.info(f"Rank: {args.rank}")

    setup_env(args.prefill_host_ip)
    if args.worker_type == "prefill":
        setup_prefill_node(
            args.rank,
            args.prefill_host_ip,
            args.total_nodes,
            args.total_nodes * args.gpus_per_node,
        )
    else:
        setup_decode_node(
            args.rank,
            args.decode_host_ip,
            args.prefill_host_ip,
            args.total_nodes,
            args.total_nodes * args.gpus_per_node,
        )

    logging.info(f"{args.worker_type.capitalize()} node setup complete")


if __name__ == "__main__":
    main()
