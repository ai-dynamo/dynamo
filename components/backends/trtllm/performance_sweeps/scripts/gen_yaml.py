# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from enum import Enum
import os
import re
from typing import Any, Dict, List

import yaml

class ModelType(Enum):
    """
    Model type.
    """
    GPT_OSS = "gpt_oss"
    DSR1 = "dsr1"
    
def get_model_type(model_path: str) -> str:
    if "r1" in model_path.lower():
        print("Inferring DSR1-type model")
        return ModelType.DSR1
    else:
        print("Inferring GPT-oss-type model")
        return ModelType.GPT_OSS

def generate_dsr1_config(    
    config_path: str,
    decode_config_path: str,
    instance_config_path: str,
    args: argparse.Namespace
):
    gen_cuda_graph_batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        768,
        1024,
        2048,
        args.gen_batch_size,
    ]

    gen_moe_backend = "CUTLASS"
    if args.gen_tp_size >= 16 and args.gen_enable_attention_dp:
        gen_moe_backend = "WIDEEP"
    if not args.gen_enable_attention_dp:
        gen_moe_backend = "TRTLLM"

    prefill_config: Dict[str, Any] = {
        "max_batch_size": args.ctx_batch_size,
        "max_num_tokens": args.ctx_max_num_tokens,
        "max_seq_len": args.ctx_max_seq_len,
        "tensor_parallel_size": args.ctx_tp_size,
        "moe_expert_parallel_size": args.ctx_ep_size,
        "enable_attention_dp": args.ctx_enable_attention_dp,
        "pipeline_parallel_size": 1,
        "cuda_graph_config": None,
        "print_iter_log": True,
        "disable_overlap_scheduler": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": args.ctx_free_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "cache_transceiver_config": {
            "max_tokens_in_buffer": args.cache_transceiver_max_num_tokens,
            "backend": "UCX",
        },
    }

    decode_config: Dict[str, Any] = {
        "tensor_parallel_size": args.gen_tp_size,
        "moe_expert_parallel_size": args.gen_tp_size,
        "enable_attention_dp": args.gen_enable_attention_dp,
        "pipeline_parallel_size": 1,
        "max_batch_size": args.gen_batch_size,
        "max_num_tokens": args.gen_max_num_tokens,
        "max_seq_len": args.gen_max_seq_len,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": gen_cuda_graph_batch_sizes,
        },
        "print_iter_log": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": args.gen_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "moe_config": {
            "backend": gen_moe_backend,
            "use_low_precision_moe_combine": True,
        },
        "cache_transceiver_config": {
            "max_tokens_in_buffer": args.cache_transceiver_max_num_tokens,
            "backend": "UCX",
        },
        "stream_interval": 100,
    }

    if args.gen_tp_size == 8 and not args.gen_enable_attention_dp:
        decode_config["allreduce_strategy"] = "MNNVL"

    if args.eplb_num_slots > 0:
        moe_load_balancer_file = os.path.join(
            os.path.dirname(config_path), "moe_load_balancer.yaml"
        )
        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(moe_load_balancer_file), exist_ok=True)
        moe_load_balancer_config = {
            "num_slots": args.eplb_num_slots,
            "layer_updates_per_iter": 1,
        }
        with open(moe_load_balancer_file, "w") as f:
            yaml.dump(
                moe_load_balancer_config, f, default_flow_style=False, sort_keys=False
            )
        decode_config["moe_config"]["load_balancer"] = moe_load_balancer_file

    if args.mtp_size > 0:
        prefill_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_size,
        }
        decode_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_size,
        }
    
    return prefill_config, decode_config

def generate_gpt_oss_config(
    config_path: str,
    decode_config_path: str,
    instance_config_path: str,
    args: argparse.Namespace
):
    gen_cuda_graph_batch_sizes = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        384,
        512,
        768,
        1024,
        2048,
        args.gen_batch_size,
    ]

    gen_moe_backend = "TRTLLM"

    prefill_config: Dict[str, Any] = {
        "max_batch_size": args.ctx_batch_size,
        "max_num_tokens": args.ctx_max_num_tokens,
        "max_seq_len": args.ctx_max_seq_len,
        "tensor_parallel_size": args.ctx_tp_size,
        "moe_expert_parallel_size": args.ctx_ep_size,
        "enable_attention_dp": args.ctx_enable_attention_dp,
        "pipeline_parallel_size": 1,
        "cuda_graph_config": None,
        "print_iter_log": True,
        "disable_overlap_scheduler": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": args.ctx_free_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "cuda_graph_config": {
            "enable_padding": True,
            "max_batch_size": 30,
        },
        "num_postprocess_workers": 4,
        "cache_transceiver_config": {
            "max_tokens_in_buffer": args.cache_transceiver_max_num_tokens,
            "backend": "UCX",
        },
        "moe_config": {
            "backend": "TRTLLM"
        }
    }

    decode_config: Dict[str, Any] = {
        "allreduce_strategy": "AUTO",
        "attention_dp_config": {
            "enable_balance": True
        },
        "disable_overlap_scheduler": False,
        "tensor_parallel_size": args.gen_tp_size,
        "moe_expert_parallel_size": args.gen_ep_size,
        "enable_attention_dp": args.gen_enable_attention_dp,
        "pipeline_parallel_size": 1,
        "max_batch_size": args.gen_batch_size,
        "max_num_tokens": args.gen_max_num_tokens,
        "max_seq_len": args.gen_max_seq_len,
        "cuda_graph_config": {
            "enable_padding": True,
            "batch_sizes": gen_cuda_graph_batch_sizes,
        },
        "print_iter_log": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": args.gen_gpu_memory_fraction,
            "dtype": "fp8",
        },
        "moe_config": {
            "backend": gen_moe_backend,
        },
        "cache_transceiver_config": {
            "max_tokens_in_buffer": args.cache_transceiver_max_num_tokens,
            "backend": "UCX",
        },
        "stream_interval": 20,
        "num_postprocess_workers": 4
    }

    if args.eplb_num_slots > 0:
        moe_load_balancer_file = os.path.join(
            os.path.dirname(config_path), "moe_load_balancer.yaml"
        )
        # Ensure the directory exists before writing the file
        os.makedirs(os.path.dirname(moe_load_balancer_file), exist_ok=True)
        moe_load_balancer_config = {
            "num_slots": args.eplb_num_slots,
            "layer_updates_per_iter": 1,
        }
        with open(moe_load_balancer_file, "w") as f:
            yaml.dump(
                moe_load_balancer_config, f, default_flow_style=False, sort_keys=False
            )
        decode_config["moe_config"]["load_balancer"] = moe_load_balancer_file

    if args.mtp_size > 0:
        prefill_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_size,
        }
        decode_config["speculative_config"] = {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": args.mtp_size,
        }
    
    return prefill_config, decode_config

CONFIG_MAPPING = {
    ModelType.GPT_OSS: generate_gpt_oss_config,
    ModelType.DSR1: generate_dsr1_config,
}

def process_node_and_task() -> tuple[int, List[str], List[str]]:
    """
    Process SLURM node and task environment variables.

    Returns:
        tuple: (max_tasks_per_node, nodes, task_nodes)
    """
    slurm_job_nodelist = os.getenv("SLURM_JOB_NODELIST", "")
    print(f"SLURM_JOB_NODELIST: {slurm_job_nodelist}")
    if not slurm_job_nodelist:
        raise ValueError("Environment variable SLURM_JOB_NODELIST not found.")

    slurm_tasks_per_node = os.getenv("SLURM_TASKS_PER_NODE", "")
    print(f"SLURM_TASKS_PER_NODE: {slurm_tasks_per_node}")
    if not slurm_tasks_per_node:
        raise ValueError("Environment variable SLURM_TASKS_PER_NODE not found.")

    # Generate list of nodes
    if "[" in slurm_job_nodelist:
        # Handle nodelist with range format (e.g., "ptyche[0065-0066]" or "nvl72029-T[15,18]")
        node_prefix = slurm_job_nodelist.split("[")[0]  # Extract everything before '['
        node_range_match = re.search(r"\[(.*?)\]", slurm_job_nodelist)
        if node_range_match is None:
            raise ValueError(
                f"Invalid nodelist format: expected range in brackets but found '{slurm_job_nodelist}'"
            )
        node_range = node_range_match.group(1)
        nodes = []
        for part in node_range.split(","):
            if "-" in part:
                start, end = part.split("-")
                # Get the width of the number format from the first number
                width = len(start)
                # Convert to integers after getting the width
                start, end = int(start), int(end)
                # Format numbers with leading zeros
                nodes.extend(
                    [
                        f"{node_prefix}{str(i).zfill(width)}"
                        for i in range(start, end + 1)
                    ]
                )
            else:
                # Preserve the original format for single numbers
                nodes.append(f"{node_prefix}{part}")
    else:
        # Handle single node format (e.g., "ptyche0065")
        nodes = [slurm_job_nodelist]
    print(f"Nodes: {nodes}")

    # Generate tasks per node
    tasks_per_node = []
    for part in slurm_tasks_per_node.split(","):
        if "(x" in part:
            count, repeat = map(int, re.findall(r"\d+", part))
            tasks_per_node.extend([count] * repeat)
        else:
            tasks_per_node.append(int(part))
    print(f"Tasks per node: {tasks_per_node}")

    if len(tasks_per_node) != len(nodes):
        raise ValueError(
            f"Number of nodes and tasks per node do not match. Number of nodes: {len(nodes)}, Number of tasks per node: {len(tasks_per_node)}"
        )

    max_tasks_per_node = max(tasks_per_node)
    task_nodes = []
    for node, tasks in zip(nodes, tasks_per_node):
        task_nodes.extend([node] * tasks)

    return max_tasks_per_node, nodes, task_nodes


def generate_urls(
    ctx_or_gen: str,
    num_instances: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    max_tasks_per_node: int,
    nodes: List[str],
    task_nodes: List[str],
    node_to_port: Dict[str, int],
    task_nodes_offset: int = 0,
) -> tuple[List[str], int]:
    """
    Generate URLs for context or generation servers.

    Returns:
        tuple: (urls, updated_task_nodes_offset)
    """
    urls: List[str] = []

    for instance in range(num_instances):
        tasks_needed = tensor_parallel_size * pipeline_parallel_size

        if (task_nodes_offset + tasks_needed) > len(task_nodes):
            print(f"{ctx_or_gen} urls so far: {urls}")
            raise ValueError(
                f"For {ctx_or_gen} instance {instance}, there are not enough tasks available. task_nodes_offset: {task_nodes_offset}, tasks_needed: {tasks_needed}, len(task_nodes): {len(task_nodes)}"
            )

        min_node = (tasks_needed + max_tasks_per_node - 1) // max_tasks_per_node
        instance_nodes = set(
            task_nodes[task_nodes_offset : task_nodes_offset + tasks_needed]
        )
        if len(instance_nodes) > min_node:
            raise ValueError(
                f"Tasks for a instance {instance} of {ctx_or_gen} instances use more node than expected. Nodes used: {instance_nodes}, number of nodes expected: {min_node}, max_tasks_per_node: {max_tasks_per_node}"
            )

        node = task_nodes[task_nodes_offset]
        port = node_to_port[node]
        node_to_port[node] += 1
        task_nodes_offset += tasks_needed

        urls.append(f"{node}:{port}")

    print(f"{ctx_or_gen} urls: {urls}")
    return urls, task_nodes_offset


def gen_config_file(
    config_path: str,
    decode_config_path: str,
    instance_config_path: str,
    model_path: str,
    num_ctx_servers: int,
    ctx_tp_size: int,
    ctx_batch_size: int,
    ctx_max_num_tokens: int,
    ctx_max_seq_len: int,
    ctx_free_gpu_memory_fraction: float,
    ctx_enable_attention_dp: bool,
    num_gen_servers: int,
    gen_tp_size: int,
    gen_ep_size: int,
    gen_batch_size: int,
    gen_max_num_tokens: int,
    gen_max_seq_len: int,
    gen_enable_attention_dp: bool,
    gen_gpu_memory_fraction: float,
    eplb_num_slots: int,
    mtp_size: int = 0,
    worker_start_port: int = 8001,
    server_port: int = 8000,
    cache_transceiver_max_num_tokens: int = 9216,
) -> None:
    """
    Generate configuration YAML file for disaggregated inference.

    Args:
        config_path: Path to save the config file
        model_path: Path to the model
        num_ctx_servers: Number of context servers
        ctx_tp_size: Tensor parallel size for context servers
        ctx_batch_size: Batch size for context servers
        ctx_max_num_tokens: Max number of tokens for context servers
        ctx_max_seq_len: Max sequence length for context servers
        ctx_free_gpu_memory_fraction: Free GPU memory fraction for context servers
        ctx_enable_attention_dp: Enable attention DP for context servers
        num_gen_servers: Number of generation servers
        gen_tp_size: Tensor parallel size for generation servers
        gen_ep_size: Expert parallel size for generation servers
        gen_batch_size: Batch size for generation servers
        gen_max_num_tokens: Max number of tokens for generation servers
        gen_enable_attention_dp: Enable attention DP for generation servers
        gen_gpu_memory_fraction: GPU memory fraction for generation servers
        eplb_num_slots: Number of slots for eplb
        worker_start_port: Start port for workers
        server_port: Server port
    """

    model_type = get_model_type(model_path)

    prefill_config, decode_config = CONFIG_MAPPING[model_type](
        config_path,
        decode_config_path,
        instance_config_path,
        args
    )

    counts = {"prefill_count": num_ctx_servers, "decode_count": num_gen_servers}

    with open(instance_config_path, "w") as f:
        yaml.dump(counts, f, default_flow_style=False, sort_keys=False)

    # Write config to file
    with open(config_path, "w") as f:
        yaml.dump(prefill_config, f, default_flow_style=False, sort_keys=False)

    with open(decode_config_path, "w") as f:
        yaml.dump(decode_config, f, default_flow_style=False, sort_keys=False)


# gen main and args
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/tmp/config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument(
        "--num_ctx_servers", type=int, required=True, help="Number of context servers"
    )
    parser.add_argument(
        "--ctx_tp_size",
        type=int,
        required=True,
        help="Tensor parallel size for context servers",
    )
    parser.add_argument(
        "--ctx_ep_size",
        type=int,
        required=True,
        help="Expert parallel size for context servers",
    )
    parser.add_argument(
        "--ctx_batch_size",
        type=int,
        required=True,
        help="Batch size for context servers",
    )
    parser.add_argument(
        "--ctx_max_num_tokens",
        type=int,
        required=True,
        help="Max number of tokens for context servers",
    )
    parser.add_argument(
        "--ctx_max_seq_len",
        type=int,
        required=True,
        help="Max sequence length for context servers",
    )
    parser.add_argument(
        "--ctx_free_gpu_memory_fraction",
        type=float,
        required=True,
        help="Free GPU memory fraction for context servers",
    )
    parser.add_argument(
        "--ctx_enable_attention_dp",
        dest="ctx_enable_attention_dp",
        action="store_true",
        help="Enable attention DP for context servers",
    )
    parser.add_argument(
        "--num_gen_servers",
        type=int,
        required=True,
        help="Number of generation servers",
    )
    parser.add_argument(
        "--gen_tp_size",
        type=int,
        required=True,
        help="Tensor parallel size for generation servers",
    )
    parser.add_argument(
        "--gen_ep_size",
        type=int,
        required=True,
        help="Expert parallel size for generation servers",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        required=True,
        help="Batch size for generation servers",
    )
    parser.add_argument(
        "--gen_max_num_tokens",
        type=int,
        required=True,
        help="Max number of tokens for generation servers",
    )
    parser.add_argument(
        "--gen_max_seq_len",
        type=int,
        required=True,
        help="Max sequence length for generation servers",
    )
    parser.add_argument(
        "--gen_enable_attention_dp",
        dest="gen_enable_attention_dp",
        action="store_true",
        help="Enable attention DP for generation servers",
    )
    parser.add_argument(
        "--gen_gpu_memory_fraction",
        type=float,
        required=True,
        help="GPU memory fraction for generation servers",
    )
    parser.add_argument(
        "--eplb_num_slots", type=int, default=0, help="Number of slots for eplb"
    )
    parser.add_argument(
        "--mtp_size", type=int, default=0, help="Number of nextn layers for MTP"
    )
    parser.add_argument(
        "--worker_start_port", type=int, default=8336, help="Start port for workers"
    )
    parser.add_argument("--server_port", type=int, default=8333, help="Server port")
    parser.add_argument(
        "--cache_transceiver_max_num_tokens",
        type=int,
        default=4608,
        help="Max number of tokens for cache transceiver",
    )

    args = parser.parse_args()

    prefill_config = args.config.replace("config.yaml", "prefill_config.yaml")
    decode_config = args.config.replace("config.yaml", "decode_config.yaml")
    instance_config = args.config.replace("config.yaml", "instance_config.yaml")

    gen_config_file(
        prefill_config,
        decode_config,
        instance_config,
        args.model,
        args.num_ctx_servers,
        args.ctx_tp_size,
        args.ctx_batch_size,
        args.ctx_max_num_tokens,
        args.ctx_max_seq_len,
        args.ctx_free_gpu_memory_fraction,
        args.ctx_enable_attention_dp,
        args.num_gen_servers,
        args.gen_tp_size,
        args.gen_batch_size,
        args.gen_max_num_tokens,
        args.gen_max_seq_len,
        args.gen_enable_attention_dp,
        args.gen_gpu_memory_fraction,
        args.eplb_num_slots,
        args.mtp_size,
        args.worker_start_port,
        args.server_port,
        args.cache_transceiver_max_num_tokens,
    )
