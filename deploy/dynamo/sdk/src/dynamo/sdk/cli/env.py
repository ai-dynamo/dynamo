from __future__ import annotations

import pkg_resources
import subprocess
import sys

import click
import distro
import platform


def get_os_version() -> str:
    """Get OS version."""
    #TODO: Revisit once we need to support Windows based systems
    return f'{distro.name()} {distro.version()}'

def get_cpu_architecture() -> str:
    """Get CPU architecture."""
    return platform.machine()

def execute_subprocess_output(command: str) -> str:
    """Execute a subprocess command and return the output."""
    try:
        out = subprocess.check_output(command, shell=True, stderr=subprocess.DEVNULL)
        return out.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        return "N/A"

def get_glibc_version() -> str:
    """Get GLIBC version."""
    return execute_subprocess_output("ldd --version | head -n 1 | awk '{print $NF}'")

def get_gpu_architecture() -> str:
    """Get GPU architecture if available"""
    return execute_subprocess_output("nvidia-smi --query-gpu=gpu_name --format=csv,noheader")

def get_cuda_version() -> str:
    """Get CUDA version if available."""
    return execute_subprocess_output(r"nvcc --version | grep -Po 'release \K\d+\.\d+'")

def get_installed_packages() -> list[tuple[str, str]]:
    """Get list of installed Python packages and their versions."""
    return [(pkg.key, pkg.version) for pkg in pkg_resources.working_set]

def build_env_command() -> click.Group:
    @click.command(name="env")
    def env() -> None:
        """Display information about the current environment."""
        click.echo("\nSystem Information:")
        click.echo(f"   OS: {get_os_version()}")
        click.echo(f"   CPU Architecture: {get_cpu_architecture()}")
        click.echo(f"   GPU Architecture: {get_gpu_architecture()}")
        click.echo(f"   GLIBC Version: {get_glibc_version()}")
        click.echo(f"   CUDA Version: {get_cuda_version()}")
        # Get Python version
        py_version = sys.version.split()[0]
        click.echo(f"\nPython Version: {py_version}")

        click.echo("\nPython Packages:")
        packages = get_installed_packages()
        python_packages = [
            'ai-dynamo',
            'ai-dynamo-runtime',
            'ai-dynamo-vllm',
            'nixl',
        ]
        for pkg_name in python_packages:
            version = next((version for name, version in packages if name == pkg_name), None)
            if version:
                click.echo(f"  {pkg_name}: {version}")
            else:
                click.echo(f"  {pkg_name}: Not installed")
        click.echo("")
    return env


env_command = build_env_command()
