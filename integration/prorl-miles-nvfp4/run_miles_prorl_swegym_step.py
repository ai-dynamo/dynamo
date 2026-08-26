#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train one Qwen3 NVFP4 step from four real ProRL SWE-Gym sessions."""

from __future__ import annotations

import argparse
import shlex

from scripts.run_qwen3_30b_a3b import ScriptArgs, execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--polar-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--router-host", required=True)
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--control-url", required=True)
    return parser.parse_args()


def option(name: str, value: object) -> str:
    return f"{name} {shlex.quote(str(value))}"


def main() -> None:
    cli = parse_args()
    extra_args = " ".join(
        [
            option("--prompt-data", cli.prompt_data),
            "--input-key prompt",
            "--label-key label",
            "--metadata-key metadata",
            "--num-rollout 1",
            "--rollout-batch-size 1",
            "--n-samples-per-prompt 4",
            "--global-batch-size 4",
            "--rollout-max-response-len 4096",
            "--rollout-max-context-len 131072",
            "--rollout-temperature 0.7",
            "--rollout-top-p 0.95",
            "--seq-length 131072",
            "--tensor-model-parallel-size 1",
            "--context-parallel-size 4",
            "--pipeline-model-parallel-size 1",
            "--expert-model-parallel-size 4",
            "--expert-tensor-parallel-size 1",
            "--recompute-granularity full",
            "--recompute-method uniform",
            "--recompute-num-layers 1",
            "--optimizer-cpu-offload",
            "--overlap-cpu-optimizer-d2h-h2d",
            "--use-precision-aware-optimizer",
            "--position-embedding-type yarn",
            "--rotary-scaling-factor 4.0",
            "--yarn-original-max-position-embeddings 32768",
            "--max-tokens-per-gpu 32768",
            "--rollout-function-path slime_bridge.rollout.generate_rollout_polar_async",
            "--custom-rm-path slime_bridge.reward.reward_func",
            (
                "--custom-reward-post-process-path "
                "slime_bridge.reward_post_process.post_process_rewards"
            ),
            "--reward-key score",
            option("--custom-config-path", cli.polar_config),
            "--disable-rewards-normalization",
            "--rollout-external",
            option("--rollout-external-engine-addrs", cli.control_url),
            option("--sglang-router-ip", cli.router_host),
            option("--sglang-router-port", cli.router_port),
            "--rollout-num-gpus-per-engine 2",
            "--update-weight-transfer-mode broadcast",
            "--use-rollout-routing-replay",
            "--save-interval 1",
            "--save-retain-interval 1",
            "--seed 1234",
        ]
    )
    config = ScriptArgs(
        mode="debug_minimal",
        run_id=cli.run_id,
        model_name="Qwen3-30B-A3B",
        megatron_model_type="qwen3-30B-A3B",
        num_gpus_per_node=4,
        actor_num_gpus_per_node=4,
        rollout_num_gpus=2,
        no_colocate=True,
        hardware="B200",
        enable_eval=False,
        model_dir=cli.model_dir,
        output_dir=cli.output_dir,
        megatron_path="/root/Megatron-LM",
        rollout_nvfp4=True,
        train_nvfp4=True,
        enable_megatron_bridge=True,
        extra_args=extra_args,
    )
    execute(config)


if __name__ == "__main__":
    main()
