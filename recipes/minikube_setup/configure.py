#!/usr/bin/env python3
"""Configure a minikube cluster with Docker driver, GPU support, and NVIDIA GPU Operator."""

import logging
import subprocess
import sys
from subprocess import CalledProcessError, check_call, check_output, DEVNULL

import typer

logger = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_show_locals=False, no_args_is_help=True)

GPU_OPERATOR_VERSION = "v25.3.0"


def check_nvidia_driver() -> tuple[bool, str]:
    try:
        output = check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            stderr=DEVNULL,
            text=True,
        ).strip()

        if not output:
            return False, "NVIDIA driver loaded but no GPUs detected"

        lines = output.split("\n")
        gpu_count = len(lines)
        driver_version = lines[0].split(",")[0].strip()
        check_output(["nvidia-smi", "-L"], stderr=DEVNULL)

        return True, f"NVIDIA driver {driver_version} loaded, {gpu_count} GPU(s) found"

    except FileNotFoundError:
        return False, "nvidia-smi not found — NVIDIA driver may not be installed"
    except CalledProcessError as e:
        return False, f"nvidia-smi failed with exit code {e.returncode}"
    except Exception as e:
        return False, f"Unexpected error checking NVIDIA driver: {e!s}"


def _is_minikube_running() -> bool:
    result = subprocess.run(["minikube", "status"], capture_output=True, text=True)
    if result.returncode == 85:
        logger.info("minikube is not running")
        return False
    if result.returncode == 0 and "Running" in result.stdout:
        return True
    logger.error(f"minikube status: rc={result.returncode}, stdout={result.stdout!r}")
    return False


def _minikube_delete() -> None:
    if _is_minikube_running():
        logger.warning("Minikube is currently running!")
        if not typer.confirm("This will stop and delete the existing minikube cluster. Continue?"):
            logger.info("Cancelled by user")
            raise typer.Exit(0)
        logger.info("Stopping and deleting minikube")
        check_call(["minikube", "stop"])
        check_call(["minikube", "delete"])


def _start_minikube() -> None:
    """Start minikube with Docker driver and GPU passthrough."""
    cmd = ["minikube", "start", "--driver", "docker", "--container-runtime", "docker", "--gpus", "all"]
    check_call(cmd)
    logger.info("Minikube started")

    current_context = check_output(["kubectl", "config", "current-context"], text=True).strip()
    logger.info(f"Current context: {current_context}")

    if current_context != "minikube":
        logger.info("Switching context to minikube")
        check_call(["kubectl", "config", "use-context", "minikube"])

    current_context = check_output(["kubectl", "config", "current-context"], text=True).strip()
    assert current_context == "minikube", f"Expected minikube context, got: {current_context}"


def _install_gpu_operator(driver_installed: bool) -> None:
    """Install NVIDIA GPU Operator via Helm."""
    check_call(["helm", "repo", "add", "nvidia", "https://helm.ngc.nvidia.com/nvidia", "--force-update"])
    check_call(["helm", "repo", "update"])

    cmd = [
        "helm", "install", "--wait", "--generate-name",
        "-n", "gpu-operator", "--create-namespace",
        "nvidia/gpu-operator",
        f"--version={GPU_OPERATOR_VERSION}",
    ]
    if driver_installed:
        # Host driver is already loaded; tell the operator not to install its own
        cmd += ["--set", "driver.enabled=false"]

    check_call(cmd)
    logger.info("GPU Operator installed successfully")


@app.command()
def setup() -> None:
    """Set up or rebuild the minikube cluster from scratch with GPU support."""
    logger.info("Starting minikube full setup")

    driver_ok, driver_msg = check_nvidia_driver()
    logger.info(driver_msg)
    if not driver_ok:
        logger.error("NVIDIA driver is not working — aborting")
        raise typer.Exit(1)

    _minikube_delete()
    _start_minikube()
    _install_gpu_operator(driver_installed=driver_ok)

    logger.info("Minikube full setup completed successfully!")


@app.command()
def start() -> None:
    """Start minikube with Docker driver and GPU support (no teardown)."""
    _start_minikube()


@app.command()
def delete() -> None:
    """Stop and delete the existing minikube cluster (with confirmation)."""
    _minikube_delete()


@app.command()
def install_gpu_operator() -> None:
    """Install the NVIDIA GPU Operator into the running minikube cluster."""
    driver_ok, driver_msg = check_nvidia_driver()
    logger.info(driver_msg)
    _install_gpu_operator(driver_installed=driver_ok)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
