# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from benchmarks.utils.genai import run_concurrency_sweep
from benchmarks.utils.plot import generate_plots
from benchmarks.utils.vanilla_client import VanillaBackendClient
from deploy.utils.dynamo_deployment import DynamoDeploymentClient


@dataclass
class DeploymentConfig:
    """Configuration for a single deployment type"""

    name: str  # Human-readable name (e.g., "aggregated")
    manifest_path: str  # Path to deployment manifest
    output_subdir: str  # Subdirectory name for results (e.g., "agg")
    client_factory: Callable  # Function to create the client
    deploy_func: Callable  # Function to deploy the client


def create_dynamo_client(namespace: str, manifest_path: str) -> DynamoDeploymentClient:
    """Factory function for DynamoDeploymentClient"""
    deployment_name = Path(manifest_path).stem
    return DynamoDeploymentClient(namespace=namespace, deployment_name=deployment_name)


def create_vanilla_client(namespace: str, manifest_path: str) -> VanillaBackendClient:
    """Factory function for VanillaBackendClient"""
    return VanillaBackendClient(namespace=namespace)


async def deploy_dynamo_client(
    client: DynamoDeploymentClient, manifest_path: str
) -> None:
    """Deploy a DynamoDeploymentClient"""
    await client.create_deployment(manifest_path)
    await client.wait_for_deployment_ready(timeout=1800)


async def deploy_vanilla_client(
    client: VanillaBackendClient, manifest_path: str
) -> None:
    """Deploy a VanillaBackendClient"""
    await client.create_deployment(manifest_path)
    await client.wait_for_deployment_ready(timeout=1800)


async def teardown(client) -> None:
    """Clean up deployment and stop port forwarding"""
    try:
        if hasattr(client, "stop_port_forward"):
            client.stop_port_forward()
        await client.delete_deployment()
    except Exception:
        pass


def print_deployment_start(config: DeploymentConfig, output_dir: str) -> None:
    """Print deployment start messages"""
    print(f"🚀 Starting {config.name} deployment benchmark...")
    print(f"📄 Manifest: {config.manifest_path}")
    print(f"📁 Results will be saved to: {Path(output_dir) / config.output_subdir}")


def print_concurrency_start(
    deployment_name: str, model: str, isl: int, osl: int, std: int
) -> None:
    """Print concurrency sweep start messages"""
    print(f"⚙️  Starting {deployment_name} concurrency sweep!", flush=True)
    print(
        "⏱️  This may take several minutes - running through multiple concurrency levels...",
        flush=True,
    )
    print(f"🎯 Model: {model} | ISL: {isl} | OSL: {osl} | StdDev: {std}")


def print_deployment_complete(config: DeploymentConfig) -> None:
    """Print deployment completion message"""
    print(f"✅ {config.name.title()} deployment benchmark completed successfully!")


def print_deployment_skip(deployment_type: str) -> None:
    """Print deployment skip message"""
    print(f"⏭️  Skipping {deployment_type} deployment (not specified)")


async def run_single_deployment_benchmark(
    config: DeploymentConfig,
    namespace: str,
    output_dir: str,
    model: str,
    isl: int,
    osl: int,
    std: int,
) -> None:
    """Run benchmark for a single deployment type"""
    print_deployment_start(config, output_dir)

    # Create and deploy client
    client = config.client_factory(namespace, config.manifest_path)
    await config.deploy_func(client, config.manifest_path)

    try:
        print_concurrency_start(config.name, model, isl, osl, std)

        # Run concurrency sweep
        (Path(output_dir) / config.output_subdir).mkdir(parents=True, exist_ok=True)
        run_concurrency_sweep(
            service_url=client.port_forward_frontend(quiet=True),
            model_name=model,
            isl=isl,
            osl=osl,
            stddev=std,
            output_dir=Path(output_dir) / config.output_subdir,
        )

    finally:
        await teardown(client)

    print_deployment_complete(config)


async def run_endpoint_benchmark(
    endpoint: str,
    model: str,
    isl: int,
    osl: int,
    std: int,
    output_dir: str,
) -> None:
    """Run benchmark for an existing endpoint"""
    print(f"🚀 Starting benchmark of existing endpoint: {endpoint}")
    print(f"📁 Results will be saved to: {Path(output_dir) / 'benchmarking'}")
    print_concurrency_start("endpoint", model, isl, osl, std)

    run_concurrency_sweep(
        service_url=endpoint,
        model_name=model,
        isl=isl,
        osl=osl,
        stddev=std,
        output_dir=Path(output_dir) / "benchmarking",
    )
    print("✅ Endpoint benchmark completed successfully!")


def print_final_summary(output_dir: str, deployed_types: List[str]) -> None:
    """Print final benchmark summary"""
    print("📊 Generating performance plots...")
    generate_plots(base_output_dir=Path(output_dir))
    print(f"📈 Plots saved to: {Path(output_dir) / 'plots'}")
    print(f"📋 Summary saved to: {Path(output_dir) / 'SUMMARY.txt'}")

    print()
    print("🎉 Benchmark workflow completed successfully!")
    print(f"📁 All results available at: {output_dir}")

    if deployed_types:
        print(f"🚀 Benchmarked deployments: {', '.join(deployed_types)}")

    print(f"📊 View plots at: {Path(output_dir) / 'plots'}")


async def run_benchmark_workflow(
    namespace: str,
    agg_manifest: str = None,
    disagg_manifest: str = None,
    vanilla_manifest: str = None,
    endpoint: str = None,
    isl: int = 200,
    std: int = 10,
    osl: int = 200,
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    output_dir: str = "benchmarks/results",
) -> None:
    """Main benchmark workflow orchestrator"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Handle endpoint benchmarking
    if endpoint:
        await run_endpoint_benchmark(endpoint, model, isl, osl, std, output_dir)
        print_final_summary(output_dir, [])
        return

    # Define deployment configurations
    deployment_configs = []

    if agg_manifest:
        deployment_configs.append(
            DeploymentConfig(
                name="aggregated",
                manifest_path=agg_manifest,
                output_subdir="agg",
                client_factory=create_dynamo_client,
                deploy_func=deploy_dynamo_client,
            )
        )
    else:
        print_deployment_skip("aggregated")

    if disagg_manifest:
        deployment_configs.append(
            DeploymentConfig(
                name="disaggregated",
                manifest_path=disagg_manifest,
                output_subdir="disagg",
                client_factory=create_dynamo_client,
                deploy_func=deploy_dynamo_client,
            )
        )
    else:
        print_deployment_skip("disaggregated")

    if vanilla_manifest:
        deployment_configs.append(
            DeploymentConfig(
                name="vanilla backend",
                manifest_path=vanilla_manifest,
                output_subdir="vanilla",
                client_factory=create_vanilla_client,
                deploy_func=deploy_vanilla_client,
            )
        )
    else:
        print_deployment_skip("vanilla backend")

    # Run benchmarks for each deployment type
    deployed_types = []
    for config in deployment_configs:
        await run_single_deployment_benchmark(
            config, namespace, output_dir, model, isl, osl, std
        )
        deployed_types.append(config.name)

    # Generate final summary
    print_final_summary(output_dir, deployed_types)
