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
from dataclasses import dataclass, field

import numpy as np
import yaml

from benchmarks.profiler.utils.aiperf import (
    get_decode_itl_and_thpt_per_gpu,
    get_prefill_ttft,
)
from benchmarks.profiler.utils.config_modifiers import CONFIG_MODIFIERS
from benchmarks.profiler.utils.config_modifiers.parallelization_mapping import (
    ParallelizationMapping,
    apply_parallel_mapping_to_config,
    get_candidate_parallel_mappings,
)
from benchmarks.profiler.utils.defaults import EngineType
from benchmarks.profiler.utils.dgd_generation import generate_dgd_config_with_planner
from benchmarks.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from benchmarks.profiler.utils.plot import (
    plot_decode_performance,
    plot_pd_joint_results,
    plot_prefill_performance,
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


@dataclass
class PrefillProfileData:
    """Container for prefill profiling results."""

    num_gpus: list[int] = field(default_factory=list)
    ttft: list[float] = field(default_factory=list)
    thpt_per_gpu: list[float] = field(default_factory=list)
    parallel_mapping_labels: list[str] = field(default_factory=list)
    parallel_mappings: list[ParallelizationMapping] = field(default_factory=list)

    def add_data(
        self,
        num_gpus: int,
        ttft: float,
        thpt_per_gpu: float,
        parallel_mapping_label: str,
        parallel_mapping: ParallelizationMapping,
    ) -> None:
        """Add a complete data point to the profile data."""
        self.num_gpus.append(num_gpus)
        self.ttft.append(ttft)
        self.thpt_per_gpu.append(thpt_per_gpu)
        self.parallel_mapping_labels.append(parallel_mapping_label)
        self.parallel_mappings.append(parallel_mapping)


@dataclass
class DecodeProfileData:
    """Container for decode profiling results."""

    num_gpus: list[int] = field(default_factory=list)
    itl: list[float] = field(default_factory=list)
    thpt_per_gpu: list[float] = field(default_factory=list)
    concurrency: list[int] = field(default_factory=list)
    kv_cache_size: list[int] = field(default_factory=list)
    parallel_mapping_labels: list[str] = field(default_factory=list)
    parallel_mappings: list[ParallelizationMapping] = field(default_factory=list)

    def add_data(
        self,
        num_gpus: int,
        itl: float,
        thpt_per_gpu: float,
        concurrency: int,
        kv_cache_size: int,
        parallel_mapping_label: str,
        parallel_mapping: ParallelizationMapping,
    ) -> None:
        """Add a complete data point to the profile data."""
        self.num_gpus.append(num_gpus)
        self.itl.append(itl)
        self.thpt_per_gpu.append(thpt_per_gpu)
        self.concurrency.append(concurrency)
        self.kv_cache_size.append(kv_cache_size)
        self.parallel_mapping_labels.append(parallel_mapping_label)
        self.parallel_mappings.append(parallel_mapping)


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
        config_modifier = CONFIG_MODIFIERS[args.backend]

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        if args.dgd_image:
            config = config_modifier.update_image(config, args.dgd_image)
            logger.debug(f"Using DGD image: {args.dgd_image}")

        profile_num_gpus = [
            2**i
            for i in range(int(math.log2(args.max_num_gpus_per_engine)) + 1)
            if args.min_num_gpus_per_engine <= 2**i <= args.max_num_gpus_per_engine
        ]
        logger.info(f"Profiling GPU counts: {profile_num_gpus}")
        os.makedirs(args.output_dir, exist_ok=True)

        model_name = config_modifier.get_model_name(config)

        # Determine sweep max context length: allow user-provided cap to override model's if smaller
        use_specified_max_context_len = getattr(args, "max_context_length", None)
        model_max_context_len = args.model_info.max_context_length
        if not use_specified_max_context_len and not model_max_context_len:
            raise ValueError(
                "No max_context_length available from args.max_context_length or model_info from HF config"
            )
        elif not use_specified_max_context_len:
            sweep_max_context_length = model_max_context_len
            logger.info(
                f"Using model's maximum context length: {model_max_context_len}"
            )
        elif not model_max_context_len:
            sweep_max_context_length = use_specified_max_context_len
            logger.info(
                f"Using user-provided max_context_length: {use_specified_max_context_len}"
            )
        else:
            sweep_max_context_length = min(
                use_specified_max_context_len, model_max_context_len
            )
            logger.info(
                f"Using minimum of user-provided and model's maximum context length: {sweep_max_context_length}"
            )

        if args.use_ai_configurator:
            if not args.aic_system:
                raise ValueError(
                    "Must provide --aic-system when using --use-ai-configurator."
                )

            # Fallback to args.model if aic_hf_id is not provided
            if not args.aic_hf_id:
                if args.model:
                    logger.info(
                        f"--aic-hf-id not provided, using --model ({args.model}) as HuggingFace ID for AI configurator"
                    )
                    args.aic_hf_id = args.model
                else:
                    raise ValueError(
                        "Must provide --aic-hf-id or --model when using --use-ai-configurator."
                    )

            logger.info("Using aiconfigurator to estimate performance...")
            ai_configurator_perf_estimator = AIConfiguratorPerfEstimator(
                args.aic_hf_id,
                args.aic_system.lower(),
                args.aic_backend,
                args.aic_backend_version,
            )
        else:
            if args.aic_system or args.aic_hf_id or args.aic_backend_version:
                logger.warning(
                    "Ignoring --aic-system, --aic-hf-id, and/or --backend-version "
                    "when not using --use-ai-configurator."
                )

        # first profile prefill
        prefill_data = PrefillProfileData()
        logger.info("Profiling prefill...")
        base_prefill_config = config_modifier.convert_config(
            config, EngineType.PREFILL, is_moe_model=args.model_info.is_moe
        )
        frontend_port = config_modifier.get_port(config)
        itl: float | None = None
        thpt_per_gpu: float | None = None
        for num_gpus in profile_num_gpus:
            logger.info(f"Profiling prefill with {num_gpus} GPUs...")
            candidate_mappings = get_candidate_parallel_mappings(
                num_gpus, args.model_info, EngineType.PREFILL
            )

            for mapping in candidate_mappings:
                # Apply parallel mapping to config
                prefill_config = apply_parallel_mapping_to_config(
                    base_prefill_config,
                    mapping,
                    EngineType.PREFILL,
                    config_modifier,
                    args.num_gpus_per_node,
                )
                logger.debug(f"Dynamo config: {prefill_config}")

                # Work dir includes mapping label (safe chars only)
                parallel_mapping_tag = (
                    mapping.label().replace("=", "").replace("/", "_")
                )
                work_dir = (
                    f"{args.output_dir}/prefill_{num_gpus}gpus_{parallel_mapping_tag}"
                )
                os.makedirs(work_dir, exist_ok=True)

                prefill_config_fn = f"{work_dir}/config.yaml"
                with open(prefill_config_fn, "w") as f:
                    yaml.dump(prefill_config, f)

                ttft = None
                if args.dry_run:
                    logger.info("Skipping deployment creation in dry run mode")
                elif args.use_ai_configurator:
                    logger.info("Using ai-configurator to estimate prefill latency")
                    perf_dict = ai_configurator_perf_estimator.estimate_prefill_perf(
                        args.isl,
                        tp_size=mapping.get_tp_size(),
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
                    logger.info(
                        f"Created client with service_name: {client.service_name}"
                    )
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
                    ttft = get_prefill_ttft(
                        args.isl,
                        ai_perf_artifact_dir,
                        model_name,
                        model_name,
                        base_url,
                        attention_dp_size=mapping.get_attn_dp_size(),
                    )

                    logger.info("Cleaning up deployment...")
                    await client.delete_deployment()
                    deployment_clients.remove(client)
                    logger.info("Deployment deleted")

                if ttft is not None:
                    prefill_data.add_data(
                        num_gpus=num_gpus,
                        ttft=ttft,
                        thpt_per_gpu=args.isl
                        / ttft
                        / num_gpus
                        * 1000
                        * mapping.get_attn_dp_size(),
                        parallel_mapping_label=mapping.label(),
                        parallel_mapping=mapping,
                    )

        # Plot the results as a 2D scatter plot
        if prefill_data.num_gpus and prefill_data.ttft and prefill_data.thpt_per_gpu:
            plot_prefill_performance(prefill_data, args.ttft, args.output_dir)

        # then profile decode
        decode_data = DecodeProfileData()
        logger.info("Profiling decode...")
        base_decode_config = config_modifier.convert_config(
            config, EngineType.DECODE, is_moe_model=args.model_info.is_moe
        )
        for num_gpus in profile_num_gpus:
            logger.info(f"Profiling decode with {num_gpus} GPUs...")
            candidate_mappings = get_candidate_parallel_mappings(
                num_gpus, args.model_info, EngineType.DECODE
            )

            for mapping in candidate_mappings:
                # Apply parallel mapping to config
                decode_config = apply_parallel_mapping_to_config(
                    base_decode_config,
                    mapping,
                    EngineType.DECODE,
                    config_modifier,
                    args.num_gpus_per_node,
                )
                logger.debug(f"Dynamo config: {decode_config}")

                parallel_mapping_tag = (
                    mapping.label()
                    .replace("=", "")
                    .replace("/", "_")  # safe chars for directory
                )
                work_dir = (
                    f"{args.output_dir}/decode_{num_gpus}gpus_{parallel_mapping_tag}"
                )
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
                        args.isl, args.osl, tp_size=mapping.get_tp_size()
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
                    attention_dp_size = mapping.get_attn_dp_size()
                    max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(
                        f"{work_dir}/{client.deployment_name}/{WORKER_COMPONENT_NAMES[args.backend].decode_worker_k8s_name.lower()}/0.log",
                        attention_dp_size=attention_dp_size,
                    )
                    max_concurrency = max_kv_tokens // (args.isl + args.osl)

                if not args.dry_run:
                    attention_dp_size = mapping.get_attn_dp_size()
                    sweep_num_request = get_num_request_range(
                        attention_dp_size,
                        max_concurrency,
                        args.decode_interpolation_granularity,
                    )
                    logger.info(
                        f"Sweeping num_request range based on maximum number of kv tokens: {sweep_num_request}"
                    )

                    for num_request in sweep_num_request:
                        itl = thpt_per_gpu = None
                        if args.use_ai_configurator:
                            logger.info(
                                "Using ai-configurator to estimate decode latency."
                            )
                            perf_dict = ai_configurator_perf_estimator.estimate_perf(
                                args.isl,
                                args.osl,
                                num_request,
                                mode=EngineType.DECODE,
                                tp_size=mapping.get_tp_size(),
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
                            itl, thpt_per_gpu = get_decode_itl_and_thpt_per_gpu(
                                args.isl,
                                args.osl,
                                num_request,
                                ai_perf_artifact_dir,
                                model_name,
                                model_name,
                                base_url=base_url,
                                num_gpus=num_gpus,
                                attention_dp_size=mapping.get_attn_dp_size(),
                            )

                        if itl is not None and thpt_per_gpu is not None:
                            decode_data.add_data(
                                num_gpus=num_gpus,
                                itl=itl,
                                thpt_per_gpu=thpt_per_gpu,
                                concurrency=num_request,
                                kv_cache_size=max_kv_tokens,
                                parallel_mapping_label=mapping.label(),
                                parallel_mapping=mapping,
                            )

                if not args.dry_run and not args.use_ai_configurator:
                    logger.info("Cleaning up deployment...")
                    await client.delete_deployment()
                    deployment_clients.remove(client)
                    logger.info("Deployment deleted")

        # Plot all decode results after profiling is complete
        if decode_data.num_gpus:
            plot_decode_performance(decode_data, args.itl, args.output_dir)

        if prefill_data.num_gpus and decode_data.num_gpus:
            plot_pd_joint_results(
                args.isl, args.osl, prefill_data, decode_data, args.output_dir
            )

        if args.dry_run:
            logger.info("Skipping recommendations in dry run mode")
        else:
            logger.info("Analyzing results and generate recommendations...")
            # Safety guards: no results → exit early with a clear message
            if not prefill_data.num_gpus:
                logger.error("No prefill results produced; skipping recommendations.")

            # select best parallel mapping for prefill
            if min(prefill_data.ttft) > args.ttft:
                logger.warning(
                    "No engine configuration satisfies the TTFT requirement, please try a smaller model or more powerful hardware"
                )
                selected_prefill_idx = int(np.argmin(np.array(prefill_data.ttft)))
            else:
                valid_indices = [
                    i for i, ttft in enumerate(prefill_data.ttft) if ttft <= args.ttft
                ]
                # Among valid TP sizes, select the one with highest throughput per GPU
                valid_thpts = [prefill_data.thpt_per_gpu[i] for i in valid_indices]
                max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
                selected_prefill_idx = max_thpt_idx
            logger.info(
                f"Suggested prefill parallel mapping: {prefill_data.parallel_mapping_labels[selected_prefill_idx]} on {prefill_data.num_gpus[selected_prefill_idx]} GPU(s) (TTFT {prefill_data.ttft[selected_prefill_idx]:.2f} ms, throughput {prefill_data.thpt_per_gpu[selected_prefill_idx]:.2f} tokens/s/GPU)"
            )

            # select best parallel mapping for decode
            if not decode_data.num_gpus:
                logger.error("No decode results produced; skipping recommendations.")
                return
            if min(decode_data.itl) > args.itl:
                logger.warning(
                    "No engine configuration satisfies the ITL requirement, please try a smaller model or more powerful hardware"
                )
                selected_decode_idx = int(np.argmin(np.array(decode_data.itl)))
            else:
                valid_indices = [
                    i for i, itl in enumerate(decode_data.itl) if itl <= args.itl
                ]
                # Among valid TP sizes, select the one with highest throughput per GPU
                valid_thpts = [decode_data.thpt_per_gpu[i] for i in valid_indices]
                max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
                selected_decode_idx = max_thpt_idx
            logger.info(
                f"Suggested decode parallel mapping: {decode_data.parallel_mapping_labels[selected_decode_idx]} on {decode_data.num_gpus[selected_decode_idx]} GPU(s) (ITL {decode_data.itl[selected_decode_idx]:.2f} ms, throughput {decode_data.thpt_per_gpu[selected_decode_idx]:.2f} tokens/s/GPU)"
            )

        if args.dry_run:
            # use min value for prefill and decode GPU counts
            prefill_data.num_gpus = [args.min_num_gpus_per_engine]
            decode_data.num_gpus = [args.min_num_gpus_per_engine]
            prefill_data.parallel_mappings = [
                ParallelizationMapping(tp=args.min_num_gpus_per_engine)
            ]
            decode_data.parallel_mappings = [
                ParallelizationMapping(tp=args.min_num_gpus_per_engine)
            ]
            selected_prefill_idx = 0
            selected_decode_idx = 0

        # interpolate ISL - TTFT with best prefill parallel mapping
        best_prefill_gpus = prefill_data.num_gpus[selected_prefill_idx]
        best_prefill_mapping = prefill_data.parallel_mappings[selected_prefill_idx]
        logger.info(
            f"Profiling prefill under best {best_prefill_gpus} GPU(s) with parallel mapping [{best_prefill_mapping.label()}] with different ISL..."
        )
        prefill_config = config_modifier.convert_config(
            config, EngineType.PREFILL, is_moe_model=args.model_info.is_moe
        )
        prefill_config = apply_parallel_mapping_to_config(
            prefill_config,
            best_prefill_mapping,
            EngineType.PREFILL,
            config_modifier,
            args.num_gpus_per_node,
        )
        logger.debug(f"Dynamo config: {prefill_config}")

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
                sweep_max_context_length,
                args.prefill_interpolation_granularity,
                ai_configurator_perf_estimator,
                tp_size=best_prefill_mapping.get_tp_size(),
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
                sweep_max_context_length,
                args.prefill_interpolation_granularity,
                attention_dp_size=best_prefill_mapping.get_attn_dp_size(),
            )

            logger.info("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            logger.info("Deployment deleted")

        # interpolate ITL - Active_KV_Cache - Decode_Context_Length with best decode parallel mapping
        best_decode_gpus = decode_data.num_gpus[selected_decode_idx]
        best_decode_mapping = decode_data.parallel_mappings[selected_decode_idx]
        logger.info(
            f"Profiling decode with {best_decode_gpus} GPUs with parallel mapping [{best_decode_mapping.label()}]..."
        )
        decode_config = config_modifier.convert_config(
            config, EngineType.DECODE, is_moe_model=args.model_info.is_moe
        )
        decode_config = apply_parallel_mapping_to_config(
            decode_config,
            best_decode_mapping,
            EngineType.DECODE,
            config_modifier,
            args.num_gpus_per_node,
        )
        logger.debug(f"Dynamo config: {decode_config}")

        work_dir = f"{args.output_dir}/selected_decode_interpolation"
        os.makedirs(work_dir, exist_ok=True)

        decode_config_fn = f"{work_dir}/config.yaml"
        with open(decode_config_fn, "w") as f:
            yaml.dump(decode_config, f)

        if args.dry_run:
            logger.info("Skipping deployment creation in dry run mode")
        elif args.use_ai_configurator:
            attention_dp_size = best_decode_mapping.get_attn_dp_size()
            max_kv_tokens = ai_configurator_perf_estimator.get_max_kv_tokens(
                args.isl, args.osl, tp_size=best_decode_mapping.get_tp_size()
            )
            profile_decode_aiconfigurator(
                work_dir,
                best_decode_gpus,  # num_gpus
                max_kv_tokens,
                sweep_max_context_length,
                args.decode_interpolation_granularity,
                ai_configurator_perf_estimator,
                attention_dp_size,
                tp_size=best_decode_mapping.get_tp_size(),
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

            attention_dp_size = best_decode_mapping.get_attn_dp_size()
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
                sweep_max_context_length,
                args.decode_interpolation_granularity,
                attention_dp_size,
            )

            logger.info("Cleaning up deployment...")
            await client.delete_deployment()
            deployment_clients.remove(client)
            logger.info("Deployment deleted")

        # generate DGD with planner based on profiling results
        config, mocker_config = generate_dgd_config_with_planner(
            config_path=args.config,
            config_modifier=config_modifier,
            output_dir=args.output_dir,
            args=args,
            best_prefill_mapping=best_prefill_mapping,
            best_decode_mapping=best_decode_mapping,
            num_gpus_per_node=args.num_gpus_per_node,
        )
        logger.debug(f"Final DGD config with planner: {config}")

        # save DGD config with planner; support multi-document output when a ConfigMap is included
        with open(f"{args.output_dir}/config_with_planner.yaml", "w") as f:
            if isinstance(config, list):
                yaml.dump_all(config, f)
            else:
                yaml.dump(config, f)

        # save mocker config with planner for testing purposes
        logger.debug(f"Mocker config with planner: {mocker_config}")
        with open(f"{args.output_dir}/mocker_config_with_planner.yaml", "w") as f:
            if isinstance(mocker_config, list):
                yaml.dump_all(mocker_config, f)
            else:
                yaml.dump(mocker_config, f)

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
