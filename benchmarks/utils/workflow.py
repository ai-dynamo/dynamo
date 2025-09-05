# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from benchmarks.utils.genai import run_concurrency_sweep
from benchmarks.utils.plot import generate_plots
from deploy.utils.dynamo_deployment import DynamoDeploymentClient


@dataclass
class DeploymentConfig:
    """Configuration for a single deployment type"""

    name: str  # Human-readable name (e.g., "aggregated")
    manifest_path: str  # Path to deployment manifest
    output_subdir: str  # Subdirectory name for results (e.g., "agg")
    client_factory: Callable  # Function to create the client
    deploy_func: Callable  # Function to deploy the client


def create_dynamo_client(
    namespace: str, deployment_name: str
) -> DynamoDeploymentClient:
    """Factory function for DynamoDeploymentClient"""
    return DynamoDeploymentClient(namespace=namespace, deployment_name=deployment_name)


async def deploy_dynamo_client(
    client: DynamoDeploymentClient, manifest_path: str
) -> None:
    """Deploy a DynamoDeploymentClient"""
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
    client = config.client_factory(namespace, config.output_subdir)
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
    label: str,
    endpoint: str,
    model: str,
    isl: int,
    osl: int,
    std: int,
    output_dir: str,
) -> None:
    """Run benchmark for an existing endpoint with custom label"""
    print(f"🚀 Starting benchmark of endpoint '{label}': {endpoint}")
    print(f"📁 Results will be saved to: {Path(output_dir) / label}")
    print_concurrency_start(f"endpoint ({label})", model, isl, osl, std)

    run_concurrency_sweep(
        service_url=endpoint,
        model_name=model,
        isl=isl,
        osl=osl,
        stddev=std,
        output_dir=Path(output_dir) / label,
    )
    print("✅ Endpoint benchmark completed successfully!")


def print_final_summary(output_dir: str, deployed_types: List[str]) -> None:
    """Print final benchmark summary"""
    print("📊 Generating performance plots...")
    generate_plots(
        base_output_dir=Path(output_dir), output_dir=Path(output_dir) / "plots"
    )
    print(f"📈 Plots saved to: {Path(output_dir) / 'plots'}")
    print(f"📋 Summary saved to: {Path(output_dir) / 'SUMMARY.txt'}")

    print()
    print("🎉 Benchmark workflow completed successfully!")
    print(f"📁 All results available at: {output_dir}")

    if deployed_types:
        print(f"🚀 Benchmarked deployments: {', '.join(deployed_types)}")

    print(f"📊 View plots at: {Path(output_dir) / 'plots'}")


def categorize_inputs(inputs: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Categorize inputs into endpoints and manifests"""
    endpoints = {}
    manifests = {}

    for label, value in inputs.items():
        # Validate reserved labels
        if label.lower() == "plots":
            raise ValueError(
                "Label 'plots' is reserved and cannot be used. Please choose a different label."
            )

        if value.startswith(("http://", "https://")):
            endpoints[label] = value
        else:
            # It should be a file path - validate it exists
            if not Path(value).is_file():
                raise FileNotFoundError(
                    f"Manifest file not found for input '{label}': {value}"
                )
            manifests[label] = value

    return endpoints, manifests


def validate_dynamo_manifest(manifest_path: str) -> None:
    """Validate that the manifest is a DynamoGraphDeployment"""
    try:
        with open(manifest_path, "r") as f:
            content = f.read()

        # Check for DynamoGraphDeployment
        if "kind: DynamoGraphDeployment" not in content:
            raise ValueError(
                f"Manifest {manifest_path} is not a DynamoGraphDeployment. Only DynamoGraphDeployments are supported for deployment benchmarking."
            )

    except FileNotFoundError:
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    except Exception as e:
        raise ValueError(f"Error reading manifest {manifest_path}: {e}")


async def run_benchmark_workflow(
    namespace: str,
    inputs: Dict[str, str],
    isl: int = 200,
    std: int = 10,
    osl: int = 200,
    model: str = "nvidia/Llama-3.1-8B-Instruct-FP8",
    output_dir: str = "benchmarks/results",
) -> None:
    """Main benchmark workflow orchestrator with dynamic inputs"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Categorize inputs into endpoints and manifests
    endpoints, manifests = categorize_inputs(inputs)

    # Run endpoint benchmarks
    for label, endpoint in endpoints.items():
        await run_endpoint_benchmark(label, endpoint, model, isl, osl, std, output_dir)

    # Create deployment configurations for manifests
    deployment_configs = []

    for label, manifest_path in manifests.items():
        # Validate that it's a DynamoGraphDeployment
        validate_dynamo_manifest(manifest_path)

        config = DeploymentConfig(
            name=label,
            manifest_path=manifest_path,
            output_subdir=label,
            client_factory=create_dynamo_client,
            deploy_func=deploy_dynamo_client,
        )

        deployment_configs.append(config)

    # Run benchmarks for each deployment type
    deployed_labels = list(endpoints.keys())
    for config in deployment_configs:
        await run_single_deployment_benchmark(
            config=config,
            namespace=namespace,
            output_dir=output_dir,
            model=model,
            isl=isl,
            osl=osl,
            std=std,
        )
        deployed_labels.append(config.name)

    # Generate final summary
    print_final_summary(output_dir, deployed_labels)
