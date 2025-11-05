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

import asyncio
import logging
import math
import os

import numpy as np
import yaml

from benchmarks.profiler.utils.aiperf import benchmark_decode, benchmark_prefill
from benchmarks.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from benchmarks.profiler.utils.dgd_generation import generate_dgd_config_with_planner
from benchmarks.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from benchmarks.profiler.utils.plot import (
    plot_decode_performance,
    plot_pd_joint_results,
    plot_prefill_performance,
)
from benchmarks.profiler.utils.profile_cache import (
    check_decode_results_exist,
    check_prefill_results_exist,
    load_existing_decode_results,
    load_existing_prefill_results,
)
from benchmarks.profiler.utils.profile_decode import (
    get_num_request_range,
    profile_decode,
    profile_decode_aiconfigurator,
)
from benchmarks.profiler.utils.profile_prefill import (
    profile_prefill,
    profile_prefill_aiconfigurator,
)
from benchmarks.profiler.utils.profiler_argparse import create_profiler_parser
from deploy.utils.dynamo_deployment import (
    DynamoDeploymentClient,
    cleanup_remaining_deployments,
)
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


async def run_profile(args):
    # List to track all created deployment clients for cleanup in case of failure
    deployment_clients = []

    # Inherit aic_backend from backend if not explicitly set
    if not args.aic_backend:
        args.aic_backend = args.backend

    try:
        # Log MoE model support
        if args.is_moe_model:
            logger.info(
                "MoE (Mixture of Experts) model profiling, sweeping TEP size for prefill and DEP size for decode"
            )
            assert args.backend in [
                "sglang"
            ], "MoE model support is only available for SGLang"
            assert (
                not args.use_ai_configurator
            ), "MoE model is not supported in ai-configurator"
        else:
            logger.info(
                "Standard dense model profiling, sweeping TP size for both prefill and decode"
            )

        config_modifier = CONFIG_MODIFIERS[args.backend]

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if args.dgd_image:
            config = config_modifier.update_image(config, args.dgd_image)
            logger.info(f"Using DGD image: {args.dgd_image}")

        profile_num_gpus = [
            2**i
            for i in range(int(math.log2(args.max_num_gpus_per_engine)) + 1)
            if args.min_num_gpus_per_engine <= 2**i <= args.max_num_gpus_per_engine
        ]
        if args.is_moe_model:
            # Filter GPU counts to only include divisors of num_experts
            if hasattr(args, "num_experts") and args.num_experts is not None:
                original_counts = profile_num_gpus.copy()
                profile_num_gpus = [
                    gpu_count
                    for gpu_count in profile_num_gpus
                    if args.num_experts % gpu_count == 0
                ]
                if not profile_num_gpus:
                    error_msg = (
                        f"No valid GPU counts found that divide evenly into num_experts={args.num_experts}. "
                        f"Original candidates were {original_counts}. "
                        f"Valid divisors in range would be: {[d for d in range(args.min_num_gpus_per_engine, args.max_num_gpus_per_engine + 1) if args.num_experts % d == 0]}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if len(profile_num_gpus) < len(original_counts):
                    logger.info(
                        f"Filtered GPU counts from {original_counts} to {profile_num_gpus} "
                        f"(only divisors of num_experts={args.num_experts})"
                    )
            logger.info(f"Profiling MoE GPU counts (TEP/DEP): {profile_num_gpus}")
        else:
            logger.info(f"Profiling dense model GPU counts (TP): {profile_num_gpus}")

        os.makedirs(args.output_dir, exist_ok=True)

        model_name = config_modifier.get_model_name(config)

        # Log skip behavior
        if args.force_rerun:
            logger.info(
                "Force rerun enabled - will re-run all tests even if results exist"
            )
        elif args.skip_existing_results:
            logger.info(
                "Skip existing results enabled - will skip TP sizes with existing results"
            )
        else:
            logger.info("Skip existing results disabled - will re-run all tests")

        if args.use_ai_configurator:
            if not args.aic_system:
                raise ValueError(
                    "Must provide --aic-system when using --use-ai-configurator."
                )
            if not args.aic_model_name:
                raise ValueError(
                    "Must provide --aic-model-name when using --use-ai-configurator."
                )
            if not args.aic_backend_version:
                raise ValueError(
                    "Must provide --aic-backend-version when using --use-ai-configurator."
                )

            logger.info("Will use aiconfigurator to estimate perf.")
            ai_configurator_perf_estimator = AIConfiguratorPerfEstimator(
                args.aic_model_name,
                args.aic_system.lower(),
                args.aic_backend,
                args.aic_backend_version,
            )
        else:
            if args.aic_system or args.aic_model_name or args.aic_backend_version:
                logger.warning(
                    "Will ignore --aic-system, --aic-model-name, and/or --backend-version "
                    "when not using --use-ai-configurator."
                )

        # first profile prefill
        prefill_num_gpus = []
        prefill_ttft = []
        prefill_thpt_per_gpu = []
        logger.info("Profiling prefill...")
        prefill_config = config_modifier.convert_config(
            config, "prefill", is_moe_model=args.is_moe_model
        )
        frontend_port = config_modifier.get_port(config)
        itl: float | None = None
        thpt_per_gpu: float | None = None
        for num_gpus in profile_num_gpus:
            logger.info(f"Profiling prefill with {num_gpus} GPUs...")

            # Check if results already exist for this GPU count
            if (
                args.skip_existing_results
                and not args.force_rerun
                and check_prefill_results_exist(args.output_dir, num_gpus, args.isl)
            ):
                logger.info(
                    f"Skipping prefill {num_gpus} GPU(s) - results already exist"
                )
                ttft, thpt_per_gpu = load_existing_prefill_results(
                    args.output_dir, num_gpus, args.isl
                )
                if ttft is not None and thpt_per_gpu is not None:
                    prefill_num_gpus.append(num_gpus)
                    prefill_ttft.append(ttft)
                    prefill_thpt_per_gpu.append(thpt_per_gpu)
                    logger.info(
                        f"Loaded existing prefill results: {num_gpus} GPU TTFT={ttft:.2f}ms, throughput={thpt_per_gpu:.2f} tokens/s/GPU"
                    )
                continue

            if args.is_moe_model:
                prefill_config = config_modifier.set_config_tep_size(
                    prefill_config, num_gpus, args.num_gpus_per_node
                )
            else:
                prefill_config = config_modifier.set_config_tp_size(
                    prefill_config, num_gpus
                )
            logger.info(f"Dynamo config: {prefill_config}")

            work_dir = f"{args.output_dir}/prefill_{num_gpus}gpus"
            os.makedirs(work_dir, exist_ok=True)

            prefill_config_fn = f"{work_dir}/config.yaml"
            with open(prefill_config_fn, "w") as f:
                yaml.dump(prefill_config, f)

            ttft = None
            if args.dry_run:
                logger.info("Skipping deployment creation in dry run mode")
            elif args.use_ai_configurator:
                logger.info("Using ai-configurator to estimate prefill latency.")
                perf_dict = ai_configurator_perf_estimator.estimate_prefill_perf(
                    args.isl,
                    tp_size=num_gpus,
                )
                ttft = perf_dict["context_latency"]
                logger.info(f"Estimated prefill TTFT: {ttft:.2f}ms")
            else:
                client = DynamoDeploymentClient(
                    namespace=args.namespace,
                    base_log_dir=work_dir,
                    model_name=model_name,
                    service_name=args.service_name,
                    frontend_port=frontend_port,
                    deployment_name=prefill_config["metadata"]["name"],
                )
                logger.info(f"Created client with service_name: {client.service_name}")
                deployment_clients.append(client)  # Track for cleanup
                await client.create_deployment(prefill_config_fn)
                logger.info("Waiting for deployment to be ready...")
                await client.wait_for_deployment_ready()
                logger.info("Deployment is ready")

                logger.info("Getting deployment logs...")
                await client.get_deployment_logs()
                logger.info(
                    f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
                )

                # run ai-perf
                base_url = client.get_service_url()
                ai_perf_artifact_dir = f"{work_dir}/aiperf_isl{args.isl}"
                aiperf_result = benchmark_prefill(
                    args.isl,
                    ai_perf_artifact_dir,
                    model_name,
                    model_name,
                    base_url=base_url,
                )
                if aiperf_result is not None:
                    ttft = aiperf_result["time_to_first_token"]["avg"]

                logger.info("Cleaning up deployment...")
                await client.delete_deployment()
                deployment_clients.remove(client)
                logger.info("Deployment deleted")

            if ttft is not None:
                prefill_num_gpus.append(num_gpus)
                prefill_ttft.append(ttft)
                prefill_thpt_per_gpu.append(args.isl / ttft / num_gpus * 1000)

        # Plot the results as a 2D scatter plot
        prefill_results = None
        if prefill_num_gpus and prefill_ttft and prefill_thpt_per_gpu:
            prefill_results = (prefill_num_gpus, prefill_ttft, prefill_thpt_per_gpu)
            plot_prefill_performance(prefill_results, args.ttft, args.output_dir)

        # then profile decode
        decode_num_gpus = []
        decode_itl = []
        decode_thpt_per_gpu = []
        decode_concurrency = []
        decode_kv_cache_size = []
        decode_results = []  # Store partial results for plotting later
        logger.info("Profiling decode...")
        decode_config = config_modifier.convert_config(
            config, "decode", is_moe_model=args.is_moe_model
        )
        for num_gpus in profile_num_gpus:
            logger.info(f"Profiling decode with {num_gpus} GPUs...")

            # Check if results already exist for this GPU count
            if (
                args.skip_existing_results
                and not args.force_rerun
                and check_decode_results_exist(
                    args.output_dir, num_gpus, args.isl, args.osl
                )
            ):
                logger.info(
                    f"Skipping decode {num_gpus} GPU(s) - results already exist"
                )
                existing_results = load_existing_decode_results(
                    args.output_dir, num_gpus, args.isl, args.osl
                )
                if existing_results:
                    # Add existing results to our arrays
                    engine_decode_itl = []
                    engine_decode_thpt_per_gpu = []
                    for itl, thpt_per_gpu, concurrency in existing_results:
                        decode_num_gpus.append(num_gpus)
                        decode_itl.append(itl)
                        decode_thpt_per_gpu.append(thpt_per_gpu)
                        decode_concurrency.append(concurrency)
                        # We need to get kv_cache_size from existing logs or estimate it
                        estimated_kv_cache = max(
                            100000, concurrency * (args.isl + args.osl) * 2
                        )  # Conservative estimate
                        decode_kv_cache_size.append(estimated_kv_cache)
                        engine_decode_itl.append(itl)
                        engine_decode_thpt_per_gpu.append(thpt_per_gpu)

                    # Store results for plotting
                    decode_results.append(
                        (num_gpus, engine_decode_itl, engine_decode_thpt_per_gpu)
                    )
                    logger.info(
                        f"Loaded {len(existing_results)} existing decode results for {num_gpus} GPU(s)"
                    )
                continue

            if args.is_moe_model:
                decode_config = config_modifier.set_config_dep_size(
                    decode_config, num_gpus, args.num_gpus_per_node
                )
            else:
                decode_config = config_modifier.set_config_tp_size(
                    decode_config, num_gpus
                )
            logger.info(f"Dynamo config: {decode_config}")

            work_dir = f"{args.output_dir}/decode_{num_gpus}gpus"
            os.makedirs(work_dir, exist_ok=True)

            decode_config_fn = f"{work_dir}/config.yaml"
            with open(decode_config_fn, "w") as f:
                yaml.dump(decode_config, f)

            if args.dry_run:
                logger.info("Skipping deployment creation in dry run mode")

            elif args.use_ai_configurator:
                # Compute max_concurrency and max_kv_tokens to know which
                # num_request to sweep over.
                max_concurrency = ai_configurator_perf_estimator.get_max_batch_size(
                    args.isl, args.osl, tp_size=num_gpus
                )
                max_kv_tokens = max_concurrency * (args.isl + args.osl)

            else:
                client = DynamoDeploymentClient(
                    namespace=args.namespace,
                    base_log_dir=work_dir,
                    model_name=model_name,
                    service_name=args.service_name,
                    frontend_port=frontend_port,
                    deployment_name=decode_config["metadata"]["name"],
                )
                deployment_clients.append(client)  # Track for cleanup
                await client.create_deployment(decode_config_fn)
                logger.info("Waiting for deployment to be ready...")
                await client.wait_for_deployment_ready()
                logger.info("Deployment is ready")

                logger.info("Getting deployment logs...")
                await client.get_deployment_logs()
                logger.info(
                    f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
                )

                # Compute max_concurrency and max_kv_tokens to know which
                # num_request to sweep over.
                # For MoE models, attention_dp_size = DEP size (num_gpus), for dense models = 1
                attention_dp_size = num_gpus if args.is_moe_model else 1
                max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
                    f"{work_dir}/{client.deployment_name}/{WORKER_COMPONENT_NAMES[args.backend].decode_worker_k8s_name.lower()}/0.log",
                    attention_dp_size=attention_dp_size,
                )
                max_concurrency = max_kv_tokens // (args.isl + args.osl)

            if not args.dry_run:
                attention_dp_size = num_gpus if args.is_moe_model else 1
                sweep_num_request = get_num_request_range(
                    attention_dp_size,
                    max_concurrency,
                    args.decode_interpolation_granularity,
                )
                logger.info(
                    f"Sweeping num_request range based on maximum number of kv tokens: {sweep_num_request}"
                )

                engine_decode_itl = []
                engine_decode_thpt_per_gpu = []
                for num_request in sweep_num_request:
                    itl = thpt_per_gpu = None
                    if args.use_ai_configurator:
                        logger.info("Using ai-configurator to estimate decode latency.")
                        perf_dict = ai_configurator_perf_estimator.estimate_perf(
                            args.isl,
                            args.osl,
                            num_request,
                            mode="decode",
                            tp_size=num_gpus,
                        )

                        itl = perf_dict["tpot"]
                        thpt_per_gpu = perf_dict["tokens/s/gpu"]
                        logger.info(f"Estimated decode ITL: {itl:.2f}ms")
                        logger.info(
                            f"Estimated decode throughput per GPU: {thpt_per_gpu:.2f} tokens/s/GPU"
                        )
                    else:
                        base_url = client.get_service_url()
                        ai_perf_artifact_dir = f"{work_dir}/aiperf_request{num_request}_isl{args.isl}_osl{args.osl}_n{num_request}"
                        aiperf_result = benchmark_decode(
                            args.isl,
                            args.osl,
                            num_request,
                            ai_perf_artifact_dir,
                            model_name,
                            model_name,
                            base_url=base_url,
                        )
                        if aiperf_result is not None:
                            itl = aiperf_result["inter_token_latency"]["avg"]
                            thpt_per_gpu = (
                                aiperf_result["output_token_throughput"]["avg"]
                                / num_gpus
                            )

                    if itl is not None and thpt_per_gpu is not None:
                        engine_decode_itl.append(itl)
                        engine_decode_thpt_per_gpu.append(thpt_per_gpu)
                        decode_num_gpus.append(num_gpus)
                        decode_itl.append(itl)
                        decode_thpt_per_gpu.append(thpt_per_gpu)
                        decode_concurrency.append(num_request)
                        decode_kv_cache_size.append(max_kv_tokens)

                # Store partial results for plotting later
                decode_results.append(
                    (num_gpus, engine_decode_itl, engine_decode_thpt_per_gpu)
                )

            if not args.dry_run and not args.use_ai_configurator:
                logger.info("Cleaning up deployment...")
                await client.delete_deployment()
                deployment_clients.remove(client)
                logger.info("Deployment deleted")

        # Plot all decode results after profiling is complete
        if decode_results:
            plot_decode_performance(decode_results, args.itl, args.output_dir)

        if prefill_results and decode_results:
            plot_pd_joint_results(
                args.isl, args.osl, prefill_results, decode_results, args.output_dir
            )

        if args.dry_run:
            logger.info("Skipping recommendations in dry run mode")
        else:
            logger.info("Analyzing results and generate recommendations...")
            # Safety guards: no results → exit early with a clear message
            if not (prefill_num_gpus and prefill_ttft and prefill_thpt_per_gpu):
                logger.error("No prefill results produced; skipping recommendations.")

            # select best tp size for prefill
            if min(prefill_ttft) > args.ttft:
                logger.info(
                    "No TP size satisfies the TTFT requirement, please try a smaller model or a more powerful GPU SKU"
                )
                selected_prefill_idx = int(np.argmin(np.array(prefill_ttft)))
            else:
                valid_indices = [
                    i for i, ttft in enumerate(prefill_ttft) if ttft <= args.ttft
                ]
                # Among valid TP sizes, select the one with highest throughput per GPU
                valid_thpts = [prefill_thpt_per_gpu[i] for i in valid_indices]
                max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
                selected_prefill_idx = max_thpt_idx
            logger.info(
                f"Suggested number of GPUs for prefill: {prefill_num_gpus[selected_prefill_idx]} (TTFT {prefill_ttft[selected_prefill_idx]:.2f} ms, throughput {prefill_thpt_per_gpu[selected_prefill_idx]:.2f} tokens/s/GPU)"
            )

            # scale up if estimated TTFT is 120% of target TTFT
            prefill_queue_size_upper_bound = max(
                0.1, args.ttft * 1.2 / prefill_ttft[selected_prefill_idx] - 1
            )
            # scale down if estimated TTFT is 80% of target TTFT
            prefill_queue_size_lower_bound = max(
                0.1, args.ttft * 0.8 / prefill_ttft[selected_prefill_idx] - 1
            )
            logger.info(
                f"Suggested planner upper/lower bound for prefill queue size: {prefill_queue_size_upper_bound:.2f}/{prefill_queue_size_lower_bound:.2f}"
            )

            # select best gpu count for decode
            if not (
                decode_num_gpus
                and decode_itl
                and decode_thpt_per_gpu
                and decode_concurrency
                and decode_kv_cache_size
            ):
                logger.error("No decode results produced; skipping recommendations.")
                return
            if min(decode_itl) > args.itl:
                logger.info(
                    "No TP size satisfies the ITL requirement, please try a smaller model or a more powerful GPU SKU"
                )
                selected_decode_idx = int(np.argmin(np.array(decode_itl)))
            else:
                valid_indices = [
                    i for i, itl in enumerate(decode_itl) if itl <= args.itl
                ]
                # Among valid TP sizes, select the one with highest throughput per GPU
                valid_thpts = [decode_thpt_per_gpu[i] for i in valid_indices]
                max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
                selected_decode_idx = max_thpt_idx
            logger.info(
                f"Suggested number of GPUs for decode: {decode_num_gpus[selected_decode_idx]} (ITL {decode_itl[selected_decode_idx]:.2f} ms, throughput {decode_thpt_per_gpu[selected_decode_idx]:.2f} tokens/s/GPU)"
            )

            # calculate kv cache utlization for the selected TP and concurrency
            selected_decode_kv_cache_utilization = (
                decode_concurrency[selected_decode_idx]
                * (args.isl + (args.osl / 2))
                / decode_kv_cache_size[selected_decode_idx]
            )
            # set a +- 20% range for the kv cache utilization
            logger.info(
                f"Suggested planner upper/lower bound for decode kv cache utilization: {min(1, selected_decode_kv_cache_utilization + 0.2):.2f}/{max(0.1, selected_decode_kv_cache_utilization - 0.2):.2f}"
            )

        if args.dry_run:
            # use min value for prefill and decode GPU counts
            prefill_num_gpus = [args.min_num_gpus_per_engine]
            decode_num_gpus = [args.min_num_gpus_per_engine]
            selected_prefill_idx = 0
            selected_decode_idx = 0

        # interpolate ISL - TTFT with best prefill GPU count
        best_prefill_gpus = prefill_num_gpus[selected_prefill_idx]
        logger.info(
            f"Profiling prefill under best {best_prefill_gpus} GPU(s) with different ISL..."
        )
        prefill_config = config_modifier.convert_config(
            config, "prefill", is_moe_model=args.is_moe_model
        )
        if args.is_moe_model:
            prefill_config = config_modifier.set_config_tep_size(
                prefill_config, best_prefill_gpus, args.num_gpus_per_node
            )
        else:
            prefill_config = config_modifier.set_config_tp_size(
                prefill_config, best_prefill_gpus
            )
        logger.info(f"Dynamo config: {prefill_config}")

        work_dir = f"{args.output_dir}/selected_prefill_interpolation"
        os.makedirs(work_dir, exist_ok=True)

        prefill_config_fn = f"{work_dir}/config.yaml"
        with open(prefill_config_fn, "w") as f:
            yaml.dump(prefill_config, f)

        if args.dry_run:
            logger.info("Skipping deployment creation in dry run mode")
        elif args.use_ai_configurator:
            profile_prefill_aiconfigurator(
                work_dir,
                best_prefill_gpus,  # num_gpus
                args.max_context_length,
                args.prefill_interpolation_granularity,
                ai_configurator_perf_estimator,
                tp_size=best_prefill_gpus,
            )
        else:
            client = DynamoDeploymentClient(
                namespace=args.namespace,
                base_log_dir=work_dir,
                model_name=model_name,
                service_name=args.service_name,
                frontend_port=frontend_port,
                deployment_name=prefill_config["metadata"]["name"],
            )
            deployment_clients.append(client)  # Track for cleanup
            await client.create_deployment(prefill_config_fn)
            logger.info("Waiting for deployment to be ready...")
            try:
                await client.wait_for_deployment_ready()
                logger.info("Deployment is ready")

                skip_profile = False
            except TimeoutError:
                logger.error(
                    "Deployment or model failed to become ready within timeout, skipping profiling"
                )
                skip_profile = True

            if not skip_profile:
                logger.info("Getting deployment logs...")
                await client.get_deployment_logs()
                logger.info(
                    f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
                )

            base_url = client.get_service_url()

            profile_prefill(
                work_dir,
                model_name,
                model_name,
                base_url,
                best_prefill_gpus,
                args.max_context_length,
                args.prefill_interpolation_granularity,
            )

            logger.info("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            logger.info("Deployment deleted")

        # interpolate ITL - Active_KV_Cache - Decode_Context_Length with best decode GPU count
        best_decode_gpus = decode_num_gpus[selected_decode_idx]
        logger.info(f"Profiling decode with {best_decode_gpus} GPUs...")
        if args.is_moe_model:
            decode_config = config_modifier.set_config_dep_size(
                decode_config, best_decode_gpus, args.num_gpus_per_node
            )
        else:
            decode_config = config_modifier.set_config_tp_size(
                decode_config, best_decode_gpus
            )
        logger.info(f"Dynamo config: {decode_config}")

        work_dir = f"{args.output_dir}/selected_decode_interpolation"
        os.makedirs(work_dir, exist_ok=True)

        decode_config_fn = f"{work_dir}/config.yaml"
        with open(decode_config_fn, "w") as f:
            yaml.dump(decode_config, f)

        if args.dry_run:
            logger.info("Skipping deployment creation in dry run mode")
        elif args.use_ai_configurator:
            # For MoE models, attention_dp_size = DEP size (best_decode_gpus), for dense models = 1
            attention_dp_size = best_decode_gpus if args.is_moe_model else 1
            max_kv_tokens = ai_configurator_perf_estimator.get_max_kv_tokens(
                args.isl, args.osl, tp_size=best_decode_gpus
            )
            profile_decode_aiconfigurator(
                work_dir,
                best_decode_gpus,  # num_gpus
                max_kv_tokens,
                args.max_context_length,
                args.decode_interpolation_granularity,
                ai_configurator_perf_estimator,
                attention_dp_size,
                tp_size=best_decode_gpus,
            )
        else:
            client = DynamoDeploymentClient(
                namespace=args.namespace,
                base_log_dir=work_dir,
                model_name=model_name,
                service_name=args.service_name,
                frontend_port=frontend_port,
                deployment_name=decode_config["metadata"]["name"],
            )
            deployment_clients.append(client)  # Track for cleanup
            await client.create_deployment(decode_config_fn)
            logger.info("Waiting for deployment to be ready...")
            await client.wait_for_deployment_ready()
            logger.info("Deployment is ready")

            logger.info("Getting deployment logs...")
            await client.get_deployment_logs()
            logger.info(
                f"Logs have been saved to {client.base_log_dir / client.deployment_name}"
            )

            # For MoE models, attention_dp_size = DEP size (best_decode_gpus), for dense models = 1
            attention_dp_size = best_decode_gpus if args.is_moe_model else 1
            max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
                f"{work_dir}/{client.deployment_name}/{WORKER_COMPONENT_NAMES[args.backend].decode_worker_k8s_name.lower()}/0.log",
                attention_dp_size=attention_dp_size,
            )

            base_url = client.get_service_url()

            profile_decode(
                work_dir,
                model_name,
                model_name,
                base_url,
                best_decode_gpus,
                max_kv_tokens,
                args.max_context_length,
                args.decode_interpolation_granularity,
                attention_dp_size,
            )

            logger.info("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            logger.info("Deployment deleted")

        # generate DGD with planner based on profiling results
        config = generate_dgd_config_with_planner(
            config_path=args.config,
            config_modifier=config_modifier,
            best_prefill_gpus=best_prefill_gpus,
            best_decode_gpus=best_decode_gpus,
            output_dir=args.output_dir,
            args=args,
            is_moe_model=args.is_moe_model,
            num_gpus_per_node=args.num_gpus_per_node,
        )
        logger.info(f"Final DGD config with planner: {config}")

        # save DGD config with planner; support multi-document output when a ConfigMap is included
        with open(f"{args.output_dir}/config_with_planner.yaml", "w") as f:
            if isinstance(config, list):
                yaml.dump_all(config, f)
            else:
                yaml.dump(config, f)

    except Exception as e:
        logger.error(f"Profile job failed with error: {e}")
        raise
    finally:
        # Always clean up any remaining deployments, even if the job failed
        logger.info("Performing final cleanup of any remaining deployments...")
        await cleanup_remaining_deployments(deployment_clients, args.namespace)
        logger.info("Final cleanup completed.")


if __name__ == "__main__":
    args = create_profiler_parser()

    # setup file logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_handler = logging.FileHandler(f"{args.output_dir}/profile_sla.log")
    log_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

    asyncio.run(run_profile(args))
