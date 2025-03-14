# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

from ._parser import parse_known_args
from ._process_manager import Command, ProcessManager

processes = []

# llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Llama-8B dynamo-init.process.chat/completions


def _http_commands(args, unknown_args):
    commands = []

    commands.append(
        Command(
            [
                "http",
                "--port",
                f"{args.http_port}",
            ],
            name="http",
        )
    )

    commands.append(
        Command(
            ["llmctl", "http", "remove", "chat-models", args.model],
            call=True,
            name="llmctl",
        )
    )

    llmctl_args = [
        "llmctl",
        "http",
        "add",
        "chat-models",
        args.model,
    ]

    if args.router:
        llmctl_args.append("dynamo-init.process.chat/completions")
    else:
        llmctl_args.append("dynamo-init.vllm.generate")

    commands.append(Command(args=llmctl_args, call=True, name="llmctl"))
    return commands


def _vllm_worker_commands(args, unknown_args):
    commands = []

    for worker_index in range(args.worker_count):
        command_args = []
        if not args.router:
            command_args.append("vllm-routerless-worker")
        else:
            command_args.append("vllm-worker")
            command_args.append("--enable-prefix-caching")
            command_args.append("--router")
            command_args.append(args.router)

        command_args.extend(unknown_args)
        command_args.append("--model")
        command_args.append(args.model)
        if args.block_size:
            command_args.append("--block-size")
            command_args.append(str(args.block_size))
        command_args.append("--max-model-len")
        command_args.append(str(args.max_model_len))
        command_args.append("--enforce-eager")
        if args.prefill_count:
            command_args.append("--remote-prefill")
            command_args.append("--kv-transfer-config")
            command_args.append('{"kv_connector":"DynamoNixlConnector"}')
            if args.conditional_disagg:
                command_args.append("--conditional-disagg")
                if "--enable-prefix-caching" not in command_args:
                    command_args.append("--enable-prefix-caching")
            if args.max_local_prefill_length:
                command_args.append("--max-local-prefill-length")
                command_args.append(str(args.max_local_prefill_length))

        command_args.append("--tensor-parallel-size")
        command_args.append(str(args.worker_tp))

        cuda_visible_devices = []

        for _ in range(args.worker_tp):
            if not args.reuse_gpus and args._next_gpu >= args.gpu_count:
                raise ValueError("Not enough gpus for configuration")
            cuda_visible_devices.append(args._next_gpu % args.gpu_count)
            args._next_gpu += 1

        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(
                [f"{next_gpu}" for next_gpu in cuda_visible_devices]
            )
        }

        if args.hf_hub_offline:
            env["HF_HUB_OFFLINE"] = "1"

        commands.append(
            Command(
                args=command_args,
                name=f"vllm worker_{worker_index}",
                environment=env,
                delay=10,
            )
        )
    return commands


def _vllm_prefill_worker_commands(args, unknown_args):
    commands = []

    for worker_index in range(args.prefill_count):
        command_args = []
        if not args.router:
            command_args.append("vllm-routerless-prefill-worker")
        else:
            command_args.append("vllm-prefill-worker")
            # command_args.append("--router")
            # command_args.append(args.router)

        command_args.extend(unknown_args)
        command_args.append("--model")
        command_args.append(args.model)
        if args.block_size:
            command_args.append("--block-size")
            command_args.append(str(args.block_size))
        command_args.append("--max-model-len")
        command_args.append(str(args.max_model_len))
        command_args.append("--enforce-eager")
        command_args.append("--kv-transfer-config")
        command_args.append('{"kv_connector":"DynamoNixlConnector"}')

        command_args.append("--tensor-parallel-size")
        command_args.append(str(args.prefill_tp))

        cuda_visible_devices = []

        for _ in range(args.prefill_tp):
            if not args.reuse_gpus and args._next_gpu >= args.gpu_count:
                raise ValueError("Not enough gpus for configuration")
            cuda_visible_devices.append(args._next_gpu % args.gpu_count)
            args._next_gpu += 1

        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(
                [f"{next_gpu}" for next_gpu in cuda_visible_devices]
            )
        }

        if args.hf_hub_offline:
            env["HF_HUB_OFFLINE"] = "1"

        commands.append(
            Command(
                args=command_args,
                name=f"vllm prefill worker_{worker_index}",
                environment=env,
                delay=10,
            )
        )
    return commands


# RUST_LOG=info python3 router/processor.py \
#   --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#  --tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
# --enable-prefix-caching \
# --block-size 64 \
# --max-model-len 16384


def _processor_commands(args, unknown_args):
    command_args = ["vllm-processor"]
    command_args.append("--model")
    command_args.append(args.model)
    command_args.append("--tokenizer")
    command_args.append(args.model)
    command_args.append("--enable-prefix-caching")
    if args.block_size:
        command_args.append("--block-size")
        command_args.append(str(args.block_size))
    command_args.append("--max-model-len")
    command_args.append(str(args.max_model_len))
    command_args.append("--router")
    command_args.append(args.router)

    return [Command(args=command_args, name="processor")]


# RUST_LOG=info python3 router/kv_router.py \
#     --routing-strategy prefix \
#     --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
#     --custom-router \
#     --min-workers 1


def _router_commands(args, unknown_args):
    commands = []

    if args.router:
        commands.extend(_processor_commands(args, unknown_args))

        if args.router == "kv":
            command_args = ["vllm-kv-router"]
            command_args.append("--model-name")
            command_args.append(args.model)
            command_args.append("--custom-router")
            command_args.append("True")
            command_args.append("--min-workers")
            command_args.append("1")
            if args.block_size:
                command_args.append("--block-size")
                command_args.append(str(args.block_size))

            commands.append(Command(args=command_args, name="kv router"))

    return commands


def main():
    known_args, unknown_args = parse_known_args()

    with ProcessManager(known_args, unknown_args) as process_manager:
        process_manager.add_commands(_http_commands(known_args, unknown_args))
        process_manager.add_commands(_router_commands(known_args, unknown_args))
        process_manager.add_commands(
            _vllm_prefill_worker_commands(known_args, unknown_args)
        )
        process_manager.add_commands(_vllm_worker_commands(known_args, unknown_args))
        process_manager.start()

        if not known_args.dry_run:
            while True:
                time.sleep(5)


if __name__ == "__main__":
    main()
