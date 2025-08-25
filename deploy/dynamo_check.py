#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
dynamo package checker, Python import tester, and usage guide.

Combines version checking, import testing, and usage examples into a single tool.
Features dynamic component discovery and comprehensive troubleshooting guidance.

Usage:
    dynamo_check.py                        # Run all checks
    dynamo_check.py --import-check-only    # Only test imports
    dynamo_check.py --examples             # Only show examples
    dynamo_check.py --try-pythonpath      # Test imports with workspace paths
    dynamo_check.py --help                 # Show help

Outputs:
System info (hostname: jensen-linux):
├─ OS: Ubuntu 24.04.1 LTS (Noble Numbat) (Linux 6.11.0-28-generic x86_64); Memory: 30.9/125.5 GiB; Cores: 32
├─ NVIDIA GPU: NVIDIA RTX 6000 Ada Generation (driver 570.133.07, CUDA 12.8); Power: 28.20/300.00 W; Memory: 2/49140 MiB
├─ Cargo (/usr/local/cargo/bin/cargo, cargo 1.87.0 (99624be96 2025-05-06))
   ├─ Cargo home directory: $HOME/dynamo/.build/.cargo (CARGO_HOME is set)
   └─ Cargo target directory: $HOME/dynamo/.build/target (CARGO_TARGET_DIR is set)
      ├─ Debug:   $HOME/dynamo/.build/target/debug (modified: 2025-08-14 16:47:13 PDT)
      ├─ Release: $HOME/dynamo/.build/target/release (modified: 2025-08-14 15:38:39 PDT)
      └─ Binary:  $HOME/dynamo/.build/target/debug/libdynamo_llm_capi.so (modified: 2025-08-14 16:45:31 PDT)
├─ Maturin (/opt/dynamo/venv/bin/maturin, maturin 1.9.3)
├─ Python: 3.12.3 (/opt/dynamo/venv/bin/python3)
   ├─ Torch: 2.7.1+cu126 (✅torch.cuda.is_available())
   └─ PYTHONPATH: /home/ubuntu/dynamo/components/planner/src
└─ Dynamo ($HOME/dynamo, SHA: b0d4499f2a8c, Date: 2025-08-18 11:55:00 PDT):
   └─ Runtime components (ai-dynamo-runtime 0.4.0):
      ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime-0.4.0.dist-info (created: 2025-08-14 16:47:15 PDT)
      ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime.pth (modified: 2025-08-14 16:47:15 PDT)
         └─ Points to: $HOME/dynamo/lib/bindings/python/src
      ├─ ✅ dynamo._core        $HOME/dynamo/lib/bindings/python/src/dynamo/_core.cpython-312-x86_64-linux-gnu.so (modified: 2025-08-14 16:47:15 PDT)
      ├─ ✅ dynamo.nixl_connect $HOME/dynamo/lib/bindings/python/src/dynamo/nixl_connect/__init__.py
      ├─ ✅ dynamo.llm          $HOME/dynamo/lib/bindings/python/src/dynamo/llm/__init__.py
      └─ ✅ dynamo.runtime      $HOME/dynamo/lib/bindings/python/src/dynamo/runtime/__init__.py
   └─ Framework components (ai-dynamo 0.4.0):
      ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo-0.4.0.dist-info (created: 2025-08-14 16:47:16 PDT)
      ├─ /opt/dynamo/venv/lib/python3.12/site-packages/_ai_dynamo.pth (modified: 2025-08-14 16:47:16 PDT)
         └─ Points to: $HOME/dynamo/components/backends/vllm/src
      ├─ ✅ dynamo.frontend     $HOME/dynamo/components/frontend/src/dynamo/frontend/__init__.py
      ├─ ✅ dynamo.planner      $HOME/dynamo/components/planner/src/dynamo/planner/__init__.py
      ├─ ✅ dynamo.mocker       $HOME/dynamo/components/backends/mocker/src/dynamo/mocker/__init__.py
      ├─ ✅ dynamo.trtllm       $HOME/dynamo/components/backends/trtllm/src/dynamo/trtllm/__init__.py
      ├─ ✅ dynamo.vllm         $HOME/dynamo/components/backends/vllm/src/dynamo/vllm/__init__.py
      ├─ ✅ dynamo.sglang       $HOME/dynamo/components/backends/sglang/src/dynamo/sglang/__init__.py
      └─ ✅ dynamo.llama_cpp    $HOME/dynamo/components/backends/llama_cpp/src/dynamo/llama_cpp/__init__.py
"""

import argparse
import datetime
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


class NVIDIAGPUDetector:
    """Handles NVIDIA GPU detection and information gathering."""

    def find_nvidia_smi(self) -> Optional[str]:
        """Find nvidia-smi executable."""
        nvsmi = shutil.which("nvidia-smi")
        if not nvsmi:
            for candidate in [
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
                "/usr/local/nvidia/bin/nvidia-smi",
            ]:
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        return nvsmi

    def get_nvidia_gpu_names(self, nvsmi: str) -> Tuple[List[str], bool]:
        """Get list of NVIDIA GPU names and whether nvidia-smi succeeded.

        Returns:
            Tuple of (gpu_names_list, nvidia_smi_succeeded)
        """
        try:
            proc = subprocess.run(
                [nvsmi, "-L"], capture_output=True, text=True, timeout=10
            )
            if proc.returncode == 0:
                names = []
                if proc.stdout:
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        # Example: "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)"
                        if ":" in line:
                            part = line.split(":", 1)[1].strip()
                            # Take up to first parenthesis for clean model name
                            name_only = part.split("(")[0].strip()
                            names.append(name_only)
                return names, True
            else:
                # Collect and surface error details (e.g. "Failed to initialize NVML: Unknown Error")
                errors: List[str] = []
                if proc.stderr:
                    for line in proc.stderr.splitlines():
                        line = line.strip()
                        if line:
                            errors.append(line)
                if not errors and proc.stdout:
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        if line:
                            errors.append(line)

                if errors:
                    # Return the first error line to display concisely upstream
                    return [errors[0]], False
                return [], False
        except Exception:
            return [], False

    def get_nvidia_driver_cuda_versions(self, nvsmi: str) -> Tuple[str, str]:
        """Get NVIDIA driver and CUDA versions.

        Returns:
            Tuple of (driver_version, cuda_version)
        """
        driver, cuda = "?", "?"
        try:
            # Try query method first
            proc = subprocess.run(
                [
                    nvsmi,
                    "--query-gpu=driver_version,cuda_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                parts = proc.stdout.strip().splitlines()[0].split(",")
                if len(parts) >= 1:
                    driver = parts[0].strip()
                if len(parts) >= 2:
                    cuda = parts[1].strip()
            else:
                # Fallback: parse banner using regex instead of structured query
                #
                # Why regex fallback instead of command line query:
                # 1. Compatibility: Some older nvidia-smi versions don't support
                #    --query-gpu with cuda_version field
                # 2. Robustness: The banner output is more stable across different
                #    nvidia-smi versions and driver releases
                # 3. Error handling: If the structured query fails (e.g., due to
                #    driver issues, permission problems, or unsupported fields),
                #    the banner parsing provides a reliable alternative
                # 4. Case variations: Different nvidia-smi versions may output
                #    "Driver Version" vs "driver version" vs "DRIVER VERSION"
                proc = subprocess.run(
                    [nvsmi], capture_output=True, text=True, timeout=10
                )
                if proc.returncode == 0 and proc.stdout:
                    import re

                    m = re.search(
                        r"Driver Version:\s*([0-9.]+)", proc.stdout, re.IGNORECASE
                    )
                    if m:
                        driver = m.group(1)
                    m = re.search(
                        r"CUDA Version:\s*([0-9.]+)", proc.stdout, re.IGNORECASE
                    )
                    if m:
                        cuda = m.group(1)
        except Exception:
            pass
        return driver, cuda

    def get_nvidia_power_memory_all(self, nvsmi: str, gpu_count: int) -> List[str]:
        """Get NVIDIA GPU power and memory info for all GPUs.

        Returns:
            List of formatted strings for each GPU
        """
        try:
            proc = subprocess.run(
                [
                    nvsmi,
                    "--query-gpu=power.draw,power.limit,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return [""] * gpu_count

            lines = proc.stdout.strip().splitlines()
            gpu_infos = []

            for i, line in enumerate(lines[:gpu_count]):  # Limit to expected GPU count
                parts = line.split(",")
                if len(parts) < 4:
                    gpu_infos.append("")
                    continue

                power_draw = parts[0].strip() if parts[0].strip() else "?"
                power_limit = parts[1].strip() if parts[1].strip() else "?"
                mem_used = parts[2].strip() if parts[2].strip() else "?"
                mem_total = parts[3].strip() if parts[3].strip() else "?"

                info_parts = []
                if power_draw != "?" or power_limit != "?":
                    info_parts.append(f"Power: {power_draw}/{power_limit} W")

                if mem_used != "?" and mem_total != "?":
                    # Add warning symbol if GPU memory usage is 90% or higher
                    warning_symbol = ""
                    try:
                        mem_usage_percent = (float(mem_used) / float(mem_total)) * 100
                        warning_symbol = " ⚠️" if mem_usage_percent >= 90 else ""
                    except (ValueError, ZeroDivisionError):
                        pass
                    info_parts.append(
                        f"Memory: {mem_used}/{mem_total} MiB{warning_symbol}"
                    )

                gpu_infos.append("; " + "; ".join(info_parts) if info_parts else "")

            # Fill remaining slots if we got fewer results than expected
            while len(gpu_infos) < gpu_count:
                gpu_infos.append("")

            return gpu_infos
        except Exception:
            return [""] * gpu_count

    def get_gpu_info(self) -> Tuple[List[str], Optional[str], Optional[str]]:
        """Get NVIDIA GPU information.

        Returns:
            Tuple of (gpu_lines_list, driver_version, cuda_version)
        """
        nvsmi = self.find_nvidia_smi()
        if not nvsmi:
            return ["❌ NVIDIA GPU: nvidia-smi not found"], None, None

        names_or_errors, nvsmi_succeeded = self.get_nvidia_gpu_names(nvsmi)
        if not nvsmi_succeeded:
            # If error details were captured, display them directly
            if names_or_errors:
                return [f"❌ NVIDIA GPU: {names_or_errors[0]}"], None, None
            return ["❌ NVIDIA GPU: nvidia-smi failed"], None, None

        driver, cuda = self.get_nvidia_driver_cuda_versions(nvsmi)

        # Format GPU lines
        names = names_or_errors
        if not names:
            # Treat zero GPUs as an error condition
            return (
                [f"❌ NVIDIA GPU: not detected (driver {driver}, CUDA {cuda})"],
                driver,
                cuda,
            )

        if len(names) == 1:
            # Single GPU - keep compact format
            power_mem_infos = self.get_nvidia_power_memory_all(nvsmi, 1)
            gpu_line = f"NVIDIA GPU: {names[0]} (driver {driver}, CUDA {cuda}){power_mem_infos[0]}"
            return [gpu_line], driver, cuda
        else:
            # Multiple GPUs - show each individually
            power_mem_infos = self.get_nvidia_power_memory_all(nvsmi, len(names))
            gpu_lines = []
            for i, name in enumerate(names):
                power_mem_info = power_mem_infos[i] if i < len(power_mem_infos) else ""
                gpu_line = f"NVIDIA GPU {i}: {name} (driver {driver}, CUDA {cuda}){power_mem_info}"
                gpu_lines.append(gpu_line)
            return gpu_lines, driver, cuda


class DynamoChecker:
    """Comprehensive dynamo package checker."""

    def __init__(self, workspace_dir: Optional[str] = None) -> None:
        # If a path is provided, use it directly; otherwise discover
        self.workspace_dir = (
            os.path.abspath(workspace_dir) if workspace_dir else self._find_workspace()
        )
        self.results: Dict[str, Any] = {}
        self._suppress_planner_warnings()
        # Collect warnings that should be printed later (after specific headers)
        self._deferred_messages: List[str] = []
        # Initialize NVIDIA GPU detector
        self.gpu_detector = NVIDIAGPUDetector()
        # Track whether GPU issues were detected (nvidia-smi failure or zero GPUs)
        self._gpu_error: bool = False

    def _suppress_planner_warnings(self) -> None:
        """Suppress Prometheus endpoint warnings from planner module during import testing."""
        # The planner module logs a warning about Prometheus endpoint when imported
        # outside of a Kubernetes cluster. Suppress this for cleaner output.
        planner_logger = logging.getLogger("dynamo.planner.defaults")
        planner_logger.setLevel(logging.ERROR)

    # ====================================================================
    # WORKSPACE AND COMPONENT DISCOVERY
    # ====================================================================

    def _find_workspace(self) -> str:
        """Find dynamo workspace directory.

        Returns:
            Path to workspace directory or empty string if not found
            Example: '.' (if current dir), '/home/ubuntu/dynamo', '/workspace', or ''

        Note: Checks local path first, then common locations. Validates by looking for README.md file.
        """
        candidates = [
            ".",  # Current directory (local path)
            os.path.expanduser("~/dynamo"),
            "/workspace",
            "/home/ubuntu/dynamo",
        ]

        for candidate in candidates:
            if self._is_dynamo_workspace(candidate):
                # Always return absolute path for consistent $HOME replacement
                return os.path.abspath(candidate)
        return ""

    def _is_dynamo_workspace(self, path: str) -> bool:
        """Check if a directory is a dynamo workspace by looking for characteristic files/directories.

        Args:
            path: Directory path to check

        Returns:
            True if directory appears to be a dynamo workspace

        Note: Checks for multiple indicators like README.md, components/, lib/bindings/, lib/runtime/, Cargo.toml, etc.
        """
        if not os.path.exists(path):
            return False

        # Check for characteristic dynamo workspace files and directories
        indicators = [
            "README.md",
            "components",
            "lib/bindings/python",
            "lib/runtime",
            "Cargo.toml",
        ]

        # Require at least 3 indicators to be confident it's a dynamo workspace
        found_indicators = 0
        for indicator in indicators:
            if os.path.exists(os.path.join(path, indicator)):
                found_indicators += 1

        return found_indicators >= 4

    def _discover_runtime_components(self) -> List[str]:
        """Discover ai-dynamo-runtime components from filesystem.

        Returns:
            List of runtime component module names
            Example: ['dynamo._core', 'dynamo.nixl_connect', 'dynamo.llm', 'dynamo.runtime']

        Note: Always includes 'dynamo._core' (compiled Rust module), then scans
              lib/bindings/python/src/dynamo/ for additional components.
        """
        components = ["dynamo._core"]  # Always include compiled Rust module

        if not self.workspace_dir:
            return components

        # Scan runtime components (llm, runtime, nixl_connect, etc.)
        # Examples: lib/bindings/python/src/dynamo/{llm,runtime,nixl_connect}/__init__.py
        runtime_path = f"{self.workspace_dir}/lib/bindings/python/src/dynamo"
        if not os.path.exists(runtime_path):
            print(
                f"⚠️  Warning: Runtime components directory not found: {runtime_path}"
            )
            return components

        for item in os.listdir(runtime_path):
            item_path = os.path.join(runtime_path, item)
            if os.path.isdir(item_path) and os.path.exists(f"{item_path}/__init__.py"):
                components.append(f"dynamo.{item}")

        return components

    def _discover_framework_components(self) -> List[str]:
        """Discover ai-dynamo framework components from filesystem.

        Returns:
            List of framework component module names
            Example: ['dynamo.frontend', 'dynamo.planner', 'dynamo.vllm', 'dynamo.sglang', 'dynamo.llama_cpp']

        Note: Scans components/ and components/backends/ directories for modules with __init__.py files.
        """
        components: List[str] = []

        if not self.workspace_dir:
            return components

        # Scan direct components (frontend, planner, etc.)
        # Examples: components/{frontend,planner}/src/dynamo/{frontend,planner}/__init__.py
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                item_path = os.path.join(comp_path, item)
                if os.path.isdir(item_path) and os.path.exists(
                    f"{item_path}/src/dynamo/{item}/__init__.py"
                ):
                    components.append(f"dynamo.{item}")
        else:
            # Defer this message to print under the Dynamo header for alignment
            self._deferred_messages.append(
                f"⚠️  Warning: Components directory not found: {self._replace_home_with_var(comp_path)}"
            )

        # Scan backend components (vllm, sglang, etc.)
        # Examples: components/backends/{vllm,sglang,llama_cpp}/src/dynamo/{vllm,sglang,llama_cpp}/__init__.py
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                item_path = os.path.join(backend_path, item)
                if os.path.isdir(item_path) and os.path.exists(
                    f"{item_path}/src/dynamo/{item}/__init__.py"
                ):
                    components.append(f"dynamo.{item}")
        else:
            # Defer this message to print under the Dynamo header for alignment
            self._deferred_messages.append(
                f"⚠️  Warning: Backend components directory not found: {self._replace_home_with_var(backend_path)}"
            )

        return components

    def _replace_home_with_var(self, path: str) -> str:
        """Replace user's home directory in path with $HOME.

        Args:
            path: File system path or colon-separated paths (for PYTHONPATH)

        Returns:
            Path with home directory replaced by $HOME if applicable
            Example: '/home/ubuntu/dynamo/...' -> '$HOME/dynamo/...'
            Example: '/home/ubuntu/dynamo/a:/home/ubuntu/dynamo/b' -> '$HOME/dynamo/a:$HOME/dynamo/b'
        """
        home_dir = os.path.expanduser("~")
        try:
            # Replace all occurrences for colon-separated paths like PYTHONPATH
            return path.replace(home_dir, "$HOME")
        except Exception:
            return path

    def _format_timestamp_pdt(self, timestamp: float) -> str:
        """Format a timestamp in PDT timezone.

        Args:
            timestamp: Unix timestamp

        Returns:
            Formatted timestamp string in PDT or local timezone
            Example: '2025-08-10 22:22:52 PDT'
        """
        try:
            # Use zoneinfo (standard library in Python 3.9+)
            pdt = ZoneInfo("America/Los_Angeles")
            dt = datetime.datetime.fromtimestamp(timestamp, tz=pdt)
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            # Fallback to manual PDT offset approximation
            # PDT is UTC-7, so subtract 7 hours from UTC
            dt_utc = datetime.datetime.utcfromtimestamp(timestamp)
            dt_pdt = dt_utc - datetime.timedelta(hours=7)
            return dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")

    def _get_cargo_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Get cargo target directory and cargo home directory.

        Returns:
            Tuple of (target_directory, cargo_home) or (None, None) if cargo not available
            Example: ('/home/ubuntu/dynamo/.build/target', '/home/ubuntu/.cargo')
        """
        # First check if cargo is available
        try:
            subprocess.run(
                ["cargo", "--version"], capture_output=True, text=True, timeout=5
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Do not print here; caller will render a nicely aligned warning
            return None, None

        # Get cargo home directory
        cargo_home = os.environ.get("CARGO_HOME")
        if not cargo_home:
            cargo_home = os.path.expanduser("~/.cargo")

        # Get cargo target directory
        target_directory = None
        try:
            # Run cargo metadata command to get target directory
            result = subprocess.run(
                ["cargo", "metadata", "--format-version=1", "--no-deps"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.workspace_dir
                if (self.workspace_dir and os.path.isdir(self.workspace_dir))
                else None,
            )

            if result.returncode == 0:
                # Parse JSON output to extract target_directory
                metadata = json.loads(result.stdout)
                target_directory = metadata.get("target_directory")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            # cargo metadata failed or JSON parsing failed
            pass

        return target_directory, cargo_home

    def _get_git_info(self, workspace_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Get git commit SHA and date for the workspace.

        Args:
            workspace_dir: Path to the workspace directory

        Returns:
            Tuple of (short_sha, commit_date) or (None, None) if not a git repo
            Example: ('a1b2c3d4e5f6', '2025-08-14 16:45:31 PDT')
        """
        if not workspace_dir or not os.path.exists(workspace_dir):
            return None, None

        try:
            # Get the longer SHA (12 characters)
            sha_result = subprocess.run(
                ["git", "rev-parse", "--short=12", "HEAD"],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if sha_result.returncode != 0:
                return None, None
            short_sha = sha_result.stdout.strip()

            # Get the commit timestamp
            date_result = subprocess.run(
                ["git", "show", "-s", "--format=%ct", "HEAD"],
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if date_result.returncode != 0:
                return None, None

            # Convert timestamp to PST/PDT
            timestamp = int(date_result.stdout.strip())
            commit_date = self._format_timestamp_pdt(timestamp)

            return short_sha, commit_date
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return None, None

    def _print_system_info(self) -> bool:
        """Print concise system information as a top-level section.

        Tree structure:
        System info (hostname: ...):
        ├─ OS: ...
        ├─ NVIDIA GPU: ...
        ├─ Cargo: ...
        ├─ Maturin: ...
        └─ Python: ...
           ├─ Torch: ...
           └─ PYTHONPATH: ...
        """
        # OS info
        distro = ""
        version = ""
        try:
            os_release_path = "/etc/os-release"
            if os.path.exists(os_release_path):
                with open(os_release_path, "r") as f:
                    data = f.read()
                name = ""
                ver = ""
                for line in data.splitlines():
                    if line.startswith("NAME=") and not name:
                        name = line.split("=", 1)[1].strip().strip('"')
                    elif line.startswith("VERSION=") and not ver:
                        ver = line.split("=", 1)[1].strip().strip('"')
                distro = name
                version = ver
        except Exception:
            pass

        uname = platform.uname()
        # Memory (used/total) and CPU cores
        mem_used_gib = None
        mem_total_gib = None
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meminfo[k.strip()] = v.strip()
            if "MemTotal" in meminfo and "MemAvailable" in meminfo:
                # Values are in kB
                total_kib = float(meminfo["MemTotal"].split()[0])
                avail_kib = float(meminfo["MemAvailable"].split()[0])
                used_kib = max(total_kib - avail_kib, 0.0)
                mem_total_gib = total_kib / (1024.0 * 1024.0)
                mem_used_gib = used_kib / (1024.0 * 1024.0)
        except Exception:
            pass

        cores = os.cpu_count() or 0

        if distro:
            base_linux = f"OS: {distro} {version} ({uname.system} {uname.release} {uname.machine})".strip()
        else:
            base_linux = (
                f"OS: {uname.system} {uname.release} {uname.version} ({uname.machine})"
            )

        extras = []
        if mem_used_gib is not None and mem_total_gib is not None:
            if mem_total_gib > 0:
                mem_usage_percent = (mem_used_gib / mem_total_gib) * 100
                warning_symbol = " ⚠️" if mem_usage_percent >= 90 else ""
            else:
                warning_symbol = ""
            extras.append(
                f"Memory: {mem_used_gib:.1f}/{mem_total_gib:.1f} GiB{warning_symbol}"
            )
        if cores:
            extras.append(f"Cores: {cores}")
        linux_line = base_linux if not extras else base_linux + "; " + "; ".join(extras)
        # Defer printing until we have all three lines; we print as a tree below

        # GPU info
        (
            gpu_lines,
            gpu_driver_version,
            gpu_cuda_version,
        ) = self.gpu_detector.get_gpu_info()
        # Python info
        py_ver = platform.python_version()
        py_exec = sys.executable or "python"
        py_path_env = os.environ.get("PYTHONPATH")
        py_path_str = py_path_env if py_path_env else "unset"
        python_line = f"Python: {py_ver} ({py_exec})"
        if not os.path.exists(py_exec):
            python_line = "❌ Python: not found"

        # PyTorch info
        torch_version: Optional[str] = None
        torch_cuda_available: Optional[bool] = None
        try:
            import importlib

            torch = importlib.import_module("torch")  # type: ignore
            try:
                torch_version = getattr(torch, "__version__", None)  # type: ignore[attr-defined]
                # Check CUDA availability through PyTorch
                if hasattr(torch, "cuda"):
                    torch_cuda_available = torch.cuda.is_available()  # type: ignore[attr-defined]
            except Exception:
                torch_version = None
                torch_cuda_available = None
        except Exception:
            # torch not installed
            pass

        # Extra lines for additional system info
        extra_lines: List[str] = []

        # Detect cargo binary path and version for heading
        cargo_path = shutil.which("cargo")
        cargo_version = None
        try:
            proc = subprocess.run(
                ["cargo", "--version"], capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0 and proc.stdout:
                cargo_version = proc.stdout.strip()
        except Exception:
            pass

        cargo_target, cargo_home = self._get_cargo_info()
        has_cargo = bool(cargo_path or cargo_home or cargo_target)

        # Build system info output
        hostname = platform.node()
        system_output = [f"System info (hostname: {hostname}):", f"├─ {linux_line}"]

        # Add GPU lines - handle single or multiple GPUs
        if len(gpu_lines) == 1:
            system_output.append(f"├─ {gpu_lines[0]}")
        else:
            for i, gpu_line in enumerate(gpu_lines):
                # All GPUs use ├─ (more system info follows)
                system_output.append(f"├─ {gpu_line}")

        print("\n".join(system_output))

        # CUDA line removed - driver/CUDA versions already shown in NVIDIA GPU line
        # Extra lines (e.g., CUDA memory clear status)
        for i, line in enumerate(extra_lines):
            # If cargo follows after extra lines, use mid symbol; else close on last
            is_last_extra = i == len(extra_lines) - 1
            symbol = "├─" if (has_cargo or not is_last_extra) else "└─"
            print(f"{symbol} {line}")

        # If no extra lines, and no cargo, close the system info section
        if not extra_lines and not has_cargo:
            # System info is complete, Dynamo Environment follows
            pass

        # Cargo Info block
        if has_cargo:
            cargo_heading = "Cargo ("
            if cargo_path:
                cargo_heading += f"{cargo_path}"
            else:
                cargo_heading += "cargo not found"
            if cargo_version:
                cargo_heading += f", {cargo_version}"
            cargo_heading += ")"

            # Cargo heading is not the last top-level child (Dynamo Environment follows)
            print(f"├─ {cargo_heading}")

            # Under cargo heading, indent nested details
            if cargo_home:
                cargo_home_env = os.environ.get("CARGO_HOME")
                display_cargo_home = self._replace_home_with_var(cargo_home)
                if cargo_home_env:
                    print(
                        f"   ├─ Cargo home directory: {display_cargo_home} (CARGO_HOME is set)"
                    )
                else:
                    # If there's also a target below, keep mid connector, else close
                    print(
                        f"   {'├─' if cargo_target else '└─'} Cargo home directory: {display_cargo_home}"
                    )

            if cargo_target:
                cargo_target_env = os.environ.get("CARGO_TARGET_DIR")
                display_cargo_target = self._replace_home_with_var(cargo_target)
                target_msg = (
                    f"   └─ Cargo target directory: {display_cargo_target} (CARGO_TARGET_DIR is set)"
                    if cargo_target_env
                    else f"   └─ Cargo target directory: {display_cargo_target}"
                )
                print(target_msg)

                # Nested details under Cargo target directory
                debug_dir = os.path.join(cargo_target, "debug")
                release_dir = os.path.join(cargo_target, "release")

                debug_exists = os.path.exists(debug_dir)
                release_exists = os.path.exists(release_dir)

                # Find *.so file
                so_file = self._find_so_file(cargo_target)
                has_so_file = so_file is not None

                if debug_exists:
                    symbol = "├─" if release_exists or has_so_file else "└─"
                    display_debug_dir = self._replace_home_with_var(debug_dir)
                    try:
                        debug_mtime = os.path.getmtime(debug_dir)
                        debug_time = self._format_timestamp_pdt(debug_mtime)
                        print(
                            f"      {symbol} Debug:   {display_debug_dir} (modified: {debug_time})"
                        )
                    except OSError:
                        print(
                            f"      {symbol} Debug:   {display_debug_dir} (unable to read timestamp)"
                        )

                if release_exists:
                    symbol = "├─" if has_so_file else "└─"
                    display_release_dir = self._replace_home_with_var(release_dir)
                    try:
                        release_mtime = os.path.getmtime(release_dir)
                        release_time = self._format_timestamp_pdt(release_mtime)
                        print(
                            f"      {symbol} Release: {display_release_dir} (modified: {release_time})"
                        )
                    except OSError:
                        print(
                            f"      {symbol} Release: {display_release_dir} (unable to read timestamp)"
                        )

                if has_so_file and so_file is not None:
                    display_so_file = self._replace_home_with_var(so_file)
                    try:
                        so_mtime = os.path.getmtime(so_file)
                        so_time = self._format_timestamp_pdt(so_mtime)
                        print(
                            f"      └─ Binary:  {display_so_file} (modified: {so_time})"
                        )
                    except OSError:
                        print(
                            f"      └─ Binary:  {display_so_file} (unable to read timestamp)"
                        )
        else:
            # Cargo not found: show as a top-level sibling; Dynamo follows, so use mid connector
            print(
                "├─ ❌ Cargo: not found (install Rust toolchain to see cargo target directory)"
            )

        # Maturin check (Python-Rust build tool)
        maturin_path = shutil.which("maturin")
        maturin_version = None
        try:
            proc = subprocess.run(
                ["maturin", "--version"], capture_output=True, text=True, timeout=5
            )
            if proc.returncode == 0 and proc.stdout:
                maturin_version = proc.stdout.strip()
        except Exception:
            pass

        has_maturin = bool(maturin_path or maturin_version)

        if has_maturin:
            maturin_heading = "Maturin ("
            if maturin_path:
                maturin_heading += f"{maturin_path}"
            else:
                maturin_heading += "maturin not found"
            if maturin_version:
                maturin_heading += f", {maturin_version}"
            maturin_heading += ")"
            print(f"├─ {maturin_heading}")
        else:
            print("├─ ❌ Maturin: not found")
            print("   Install with: uv pip install maturin[patchelf]")

        # Python line (moved here to appear after Maturin, before Dynamo)
        # Determine if more top-level entries come after Python
        more_after_python = bool(has_cargo)
        print(f"{'├─' if more_after_python else '└─'} {python_line}")

        # Torch version as a child under Python (before PYTHONPATH)
        if torch_version:
            cuda_status = ""
            if torch_cuda_available is not None:
                cuda_status = (
                    " (✅torch.cuda.is_available())"
                    if torch_cuda_available
                    else " (❌torch.cuda.is_available())"
                )
            print("   ├─ Torch: " + str(torch_version) + cuda_status)
        else:
            # Show as a child under Python
            print("   ├─ ❌ Torch: not installed")

        # PYTHONPATH as the last child under Python
        print(f"   └─ PYTHONPATH: {py_path_str}")
        # Determine if any errors were printed in system info
        system_errors_found = False
        if isinstance(python_line, str) and python_line.startswith("❌"):
            system_errors_found = True
        if not has_cargo:
            system_errors_found = True
        # Mark GPU error based on lines printed; treat as error for overall status as well
        try:
            self._gpu_error = any(
                isinstance(line, str) and line.startswith("❌") for line in gpu_lines
            )
            if self._gpu_error:
                system_errors_found = True
        except Exception:
            pass
        return system_errors_found

    def _find_so_file(self, target_directory: str) -> Optional[str]:
        """Find the compiled *.so file in target directory or Python bindings.

        Args:
            target_directory: Path to cargo target directory

        Returns:
            Path to *.so file or None if not found
            Example: '/home/ubuntu/dynamo/target/debug/libdynamo_core.so'
        """
        if not target_directory or not os.path.exists(target_directory):
            return None

        # Look for *.so files in debug and release directories
        for profile in ["debug", "release"]:
            profile_dir = os.path.join(target_directory, profile)
            if os.path.exists(profile_dir):
                try:
                    for root, dirs, files in os.walk(profile_dir):
                        for file in files:
                            if file.endswith(".so"):
                                return os.path.join(root, file)
                except OSError:
                    continue

        # Also check Python bindings directory for installed *.so
        if self.workspace_dir:
            bindings_dir = f"{self.workspace_dir}/lib/bindings/python/src/dynamo"
            if os.path.exists(bindings_dir):
                try:
                    for root, dirs, files in os.walk(bindings_dir):
                        for file in files:
                            if file.endswith(".so") and "_core" in file:
                                return os.path.join(root, file)
                except OSError:
                    pass

        return None

    def _get_cargo_build_profile(self, target_directory: str) -> Optional[str]:
        """Determine which cargo build profile (debug/release) was used most recently.

        Args:
            target_directory: Path to cargo target directory

        Returns:
            'debug', 'release', 'debug/release', or None if cannot determine
            Example: 'debug'
        """
        # First check environment variables that indicate current build profile
        profile_env = os.environ.get("PROFILE")
        if profile_env:
            if profile_env == "dev":
                return "debug"
            elif profile_env == "release":
                return "release"

        # Check OPT_LEVEL as secondary indicator
        opt_level = os.environ.get("OPT_LEVEL")
        if opt_level:
            if opt_level == "0":
                return "debug"
            elif opt_level in ["2", "3"]:
                return "release"

        # Fall back to filesystem inspection
        if not target_directory or not os.path.exists(target_directory):
            return None

        debug_dir = os.path.join(target_directory, "debug")
        release_dir = os.path.join(target_directory, "release")

        debug_exists = os.path.exists(debug_dir)
        release_exists = os.path.exists(release_dir)

        if not debug_exists and not release_exists:
            return None
        elif debug_exists and not release_exists:
            return "debug"
        elif release_exists and not debug_exists:
            return "release"
        else:
            # Both exist, check which was modified more recently
            try:
                debug_mtime = os.path.getmtime(debug_dir)
                release_mtime = os.path.getmtime(release_dir)

                if (
                    abs(debug_mtime - release_mtime) < 1.0
                ):  # Same timestamp (within 1 second)
                    return "debug/release"  # Both available, runtime choice depends on invocation
                else:
                    return "release" if release_mtime > debug_mtime else "debug"
            except OSError:
                return None

    def _setup_pythonpath(self) -> None:
        """Set up PYTHONPATH for component imports."""
        if not self.workspace_dir:
            return

        paths = []

        # Collect component source paths
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                src_path = f"{comp_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)
        else:
            print(
                f"⚠️  Warning: Components directory not found for PYTHONPATH setup: {comp_path}"
            )

        # Collect backend source paths
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                src_path = f"{backend_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)
        else:
            print(
                f"⚠️  Warning: Backend components directory not found for PYTHONPATH setup: {backend_path}"
            )

        # Update sys.path for current process
        if paths:
            # Add paths to sys.path for immediate effect on imports
            for path in paths:
                if path not in sys.path:
                    sys.path.insert(0, path)  # Insert at beginning for priority

            # Show what PYTHONPATH would be (for manual shell setup)
            pythonpath_value = ":".join(paths)
            current_path = os.environ.get("PYTHONPATH", "")
            if current_path:
                pythonpath_value = f"{pythonpath_value}:{current_path}"

            print(
                f"""Below are the results if you export PYTHONPATH="{pythonpath_value}":
   ({len(paths)} workspace component paths found)"""
            )
            for path in paths:
                print(f"   • {path}")
            print()
        else:
            print("⚠️  Warning: No component source paths found for PYTHONPATH setup")

    # ====================================================================
    # IMPORT TESTING
    # ====================================================================

    def _test_component_group(
        self,
        components: List[str],
        package_name: str,
        group_name: str,
        max_width: int,
        site_packages: str,
        collect_failures: bool = False,
        package_info: Optional[Dict[str, Any]] = None,
        sub_indent: str = "   ",
    ) -> Tuple[Dict[str, str], List[str]]:
        """Test a group of components for a given package.

        Args:
            components: List of component names to test
                Example: ['dynamo._core', 'dynamo.llm', 'dynamo.runtime']
            package_name: Name of the package to get version from
                Example: 'ai-dynamo-runtime'
            group_name: Display name for the group
                Example: 'Runtime components'
            max_width: Maximum width for component name alignment
                Example: 20
            site_packages: Path to site-packages directory
                Example: '/opt/dynamo/venv/lib/python3.12/site-packages'
            collect_failures: Whether to collect failed component names
                Example: True (for framework components), False (for runtime)

        Returns:
            Tuple of (results dict, list of failed components)
            Example: ({'dynamo._core': '✅ Success', 'dynamo.llm': '❌ Failed: No module named dynamo.llm'},
                     ['dynamo.llm'])

        Output printed to console:
            Dynamo Environment ($HOME/dynamo):
            └─ Runtime components (ai-dynamo-runtime 0.4.0):
               ├─ /opt/dynamo/venv/lib/.../ai_dynamo_runtime-0.4.0.dist-info (created: 2025-08-12 14:17:34 PDT)
               ├─ ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               └─ ❌ dynamo.llm          No module named 'dynamo.llm'
        """
        results = {}
        failures = []

        # Print header with version info
        try:
            version = importlib.metadata.version(package_name)
            header = f"{group_name} ({package_name} {version}):"
        except importlib.metadata.PackageNotFoundError:
            header = f"{group_name} ({package_name} - Not installed):"
        except Exception:
            header = f"{group_name} ({package_name}):"

        print(header)

        # Determine if package info should use ├─ or └─ based on whether there are components
        has_components = len(components) > 0
        package_symbol = "├─" if has_components else "└─"

        # Print package info as subitem of component group (only if found)
        if package_info:
            package_path = package_info.get("path", "")
            package_created = package_info.get("created", "")
            display_path = self._replace_home_with_var(package_path)
            if package_created:
                print(
                    f"{sub_indent}{package_symbol} {display_path} (created: {package_created})"
                )
            else:
                print(f"{sub_indent}{package_symbol} {display_path}")

            # Show .pth files if they exist (editable installs) - at same level as package info
            pth_files = package_info.get("pth_files", [])
            for i, pth_file in enumerate(pth_files):
                is_last_pth = i == len(pth_files) - 1
                pth_symbol = "└─" if (is_last_pth and not has_components) else "├─"
                display_pth_path = self._replace_home_with_var(pth_file["path"])
                display_points_to = self._replace_home_with_var(pth_file["points_to"])
                print(
                    f"{sub_indent}{pth_symbol} {display_pth_path} (modified: {pth_file['modified']})"
                )
                print(f"{sub_indent}   └─ Points to: {display_points_to}")
        # Don't print anything for "Not found" - just skip it

        # Test each component as subitems of the package
        for i, component in enumerate(components):
            # Determine tree symbol - last component gets └─, others get ├─, with proper indentation (deeper nesting)
            is_last = i == len(components) - 1
            tree_symbol = f"{sub_indent}{'└─' if is_last else '├─'}"

            try:
                module = __import__(component, fromlist=[""])
                results[component] = "✅ Success"
                # Get module path for location info
                module_path = getattr(module, "__file__", "built-in")
                if module_path and module_path != "built-in":
                    # Only show timestamps for generated files (*.so, *.pth, etc.), not __init__.py
                    timestamp_str = ""
                    show_timestamp = False

                    # Check if this is a generated file we want to show timestamps for
                    if any(
                        module_path.endswith(ext)
                        for ext in [".so", ".pth", ".dll", ".dylib"]
                    ):
                        show_timestamp = True

                    if show_timestamp:
                        try:
                            if os.path.exists(module_path):
                                mtime = os.path.getmtime(module_path)
                                timestamp_str = (
                                    f" (modified: {self._format_timestamp_pdt(mtime)})"
                                )
                        except OSError:
                            pass

                    if self.workspace_dir and module_path.startswith(
                        self.workspace_dir
                    ):
                        # From workspace source - show absolute path with $HOME replacement
                        display_path = self._replace_home_with_var(module_path)
                        if show_timestamp:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}{timestamp_str}"
                            )
                        else:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}"
                            )
                    elif site_packages and module_path.startswith(site_packages):
                        # From installed package - show path with $HOME replacement
                        display_path = self._replace_home_with_var(module_path)
                        if show_timestamp:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}{timestamp_str}"
                            )
                        else:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}"
                            )
                    else:
                        # Other location - show path with $HOME replacement
                        display_path = self._replace_home_with_var(module_path)
                        if show_timestamp:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}{timestamp_str}"
                            )
                        else:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {display_path}"
                            )
                else:
                    built_in_suffix = (
                        " (built-in)"
                        if group_name.lower().startswith("framework")
                        else " built-in"
                    )
                    print(f"{tree_symbol} ✅ {component:<{max_width}}{built_in_suffix}")
            except ImportError as e:
                results[component] = f"❌ Failed: {e}"
                print(f"{tree_symbol} ❌ {component:<{max_width}} {e}")
                if collect_failures:
                    failures.append(component)

        return results, failures

    def _get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get package installation information including .pth files.

        Args:
            package_name: Name of the package (e.g., 'ai-dynamo-runtime')

        Returns:
            Dict with 'path', 'created', and optionally 'pth_files' keys
        """
        import site

        site_packages_dirs = site.getsitepackages()
        if hasattr(site, "getusersitepackages"):
            site_packages_dirs.append(site.getusersitepackages())

        result: Dict[str, Any] = {}
        pth_files: List[Dict[str, str]] = []

        for site_dir in site_packages_dirs:
            if not os.path.exists(site_dir):
                continue

            try:
                for file in os.listdir(site_dir):
                    # Look for .dist-info directories that exactly match the package name
                    if file.endswith(".dist-info"):
                        # Extract package name from .dist-info directory name
                        dist_name = file.replace(".dist-info", "")
                        # Handle version suffixes (e.g., ai_dynamo_runtime-0.4.0 -> ai_dynamo_runtime)
                        base_name = (
                            dist_name.split("-")[0] if "-" in dist_name else dist_name
                        )
                        expected_name = package_name.replace("-", "_")

                        if base_name == expected_name:
                            dist_info_path = os.path.join(site_dir, file)
                            if os.path.isdir(dist_info_path):
                                try:
                                    ctime = os.path.getctime(dist_info_path)
                                    created_time = self._format_timestamp_pdt(ctime)
                                    result.update(
                                        {
                                            "path": dist_info_path,
                                            "created": created_time,
                                        }
                                    )
                                except OSError:
                                    result.update({"path": dist_info_path})

                    # Look for .pth files that match this specific package
                    if file.endswith(".pth"):
                        # Match .pth files to specific packages
                        pth_matches_package = False
                        if package_name == "ai-dynamo-runtime":
                            # Look for ai_dynamo_runtime.pth or similar
                            if (
                                "ai_dynamo_runtime" in file.lower()
                                or file.lower().startswith("ai_dynamo_runtime")
                            ):
                                pth_matches_package = True
                        elif package_name == "ai-dynamo":
                            # Look for _ai_dynamo.pth or ai_dynamo.pth (but not ai_dynamo_runtime.pth)
                            if (
                                "ai_dynamo" in file.lower()
                                or "_ai_dynamo" in file.lower()
                            ) and "runtime" not in file.lower():
                                pth_matches_package = True

                        if pth_matches_package:
                            pth_path = os.path.join(site_dir, file)
                            try:
                                mtime = os.path.getmtime(pth_path)
                                # Read the content to see what path it adds
                                with open(pth_path, "r") as f:
                                    content = f.read().strip()
                                pth_files.append(
                                    {
                                        "path": pth_path,
                                        "modified": self._format_timestamp_pdt(mtime),
                                        "points_to": content,
                                    }
                                )
                            except OSError:
                                pass
            except OSError:
                continue

        if pth_files:
            result["pth_files"] = pth_files

        return result

    def test_imports(self) -> Dict[str, str]:
        """Test imports for all discovered components.

        Returns:
            Dictionary mapping component names to their import status
            Example: {
                'dynamo._core': '✅ Success',
                'dynamo.llm': '✅ Success',
                'dynamo.runtime': '✅ Success',
                'dynamo.frontend': '❌ Failed: No module named dynamo.frontend',
                'dynamo.planner': '✅ Success'
            }

        Console output example:
            Dynamo Environment ($HOME/dynamo):
            └─ Runtime components (ai-dynamo-runtime 0.4.0):
               ├─ /opt/dynamo/venv/lib/.../ai_dynamo_runtime-0.4.0.dist-info (created: 2025-08-12 14:17:34 PDT)
               ├─ /opt/dynamo/venv/lib/.../ai_dynamo_runtime.pth (modified: 2025-08-12 14:17:34 PDT)
                  └─ Points to: $HOME/dynamo/lib/bindings/python/src
               ├─ ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               └─ ✅ dynamo.llm          /opt/dynamo/venv/lib/.../dynamo/llm/__init__.py

            └─ Framework components (ai-dynamo - Not installed):
               ├─ ✅ dynamo.frontend     /opt/dynamo/venv/lib/.../dynamo/frontend/__init__.py
               └─ ❌ dynamo.missing      No module named 'dynamo.missing'
        """
        results = {}

        # Print system info at top-level, before Dynamo Environment
        system_errors = self._print_system_info()

        # Then print main environment header as a subtree under System info
        if (
            self.workspace_dir
            and os.path.exists(self.workspace_dir)
            and self._is_dynamo_workspace(self.workspace_dir)
        ):
            workspace_path = os.path.abspath(self.workspace_dir)
            display_workspace = self._replace_home_with_var(workspace_path)

            # Get git info
            sha, date = self._get_git_info(self.workspace_dir)
            if sha and date:
                print(f"└─ Dynamo ({display_workspace}, SHA: {sha}, Date: {date}):")
            else:
                print(f"└─ Dynamo ({display_workspace}):")
            # Backend components directory warning after the Dynamo line
            backend_path = f"{self.workspace_dir}/components/backends"
            if not os.path.exists(backend_path):
                print(
                    f"   ⚠️  Warning: Backend components directory not found: {self._replace_home_with_var(backend_path)}"
                )
            # If there were deferred messages (e.g., invalid --path), show them here under Dynamo
            for message in self._deferred_messages:
                print(f"   {message}")
        else:
            # If a user provided an invalid --path, reflect that, otherwise generic not found
            if self.workspace_dir and not os.path.exists(self.workspace_dir):
                print(f"└─ Dynamo ({self._replace_home_with_var(self.workspace_dir)}):")
                print("   ❌ Workspace path does not exist")
            elif self.workspace_dir and not self._is_dynamo_workspace(
                self.workspace_dir
            ):
                # Still try to get git info even if it's not a valid workspace
                sha, date = self._get_git_info(self.workspace_dir)
                if sha and date:
                    print(
                        f"└─ Dynamo ({self._replace_home_with_var(self.workspace_dir)}, SHA: {sha}, Date: {date}):"
                    )
                else:
                    print(
                        f"└─ Dynamo ({self._replace_home_with_var(self.workspace_dir)}):"
                    )
                print("   ❌ Invalid dynamo workspace (missing expected files)")
            else:
                print("└─ Dynamo (workspace not found):")

        # Discover all components
        runtime_components = self._discover_runtime_components()
        framework_components = self._discover_framework_components()

        # Calculate max width for alignment across ALL components
        all_components = runtime_components + framework_components
        max_width = max(len(comp) for comp in all_components) if all_components else 0

        # Get site-packages path for comparison
        import site

        site_packages = site.getsitepackages()[0] if site.getsitepackages() else ""

        # Get package information for headers
        runtime_package_info = self._get_package_info("ai-dynamo-runtime")
        framework_package_info = self._get_package_info("ai-dynamo")

        # Test runtime components (as subitem of Dynamo Environment, indented; components printed below group header)
        runtime_results, _ = self._test_component_group(
            runtime_components,
            "ai-dynamo-runtime",
            "   └─ Runtime components",
            max_width,
            site_packages,
            collect_failures=False,
            package_info=runtime_package_info,
            sub_indent="      ",
        )
        results.update(runtime_results)

        # Test framework components (as subitem of Dynamo Environment, indented; components printed below group header)
        framework_results, framework_failures = self._test_component_group(
            framework_components,
            "ai-dynamo",
            "   └─ Framework components",
            max_width,
            site_packages,
            collect_failures=True,
            package_info=framework_package_info,
            sub_indent="      ",
        )
        results.update(framework_results)

        # Cargo information is printed under System info

        # Show PYTHONPATH recommendation if any framework components failed (moved to end)
        if framework_failures and self.workspace_dir:
            pythonpath = self._get_pythonpath()
            if pythonpath:
                # Apply $HOME replacement to PYTHONPATH for consistency
                display_pythonpath = self._replace_home_with_var(pythonpath)
                self._show_build_options(display_pythonpath)

        # Exit with non-zero status if any errors detected
        # Treat Python or Cargo failures from system info, and invalid path, as failures.
        any_failures = (
            system_errors
            or any(msg.startswith("❌ Error:") for msg in self._deferred_messages)
            or bool(framework_failures)
        )
        # Store whether errors occurred for overall run
        self.results["had_errors"] = any_failures

        return results

    def _show_build_options(self, display_pythonpath: Optional[str] = None) -> None:
        """Show usage/build guidance including PYTHONPATH export.

        Args:
            display_pythonpath: Optional precomputed PYTHONPATH string with $HOME replacement
        """
        # Compute display_pythonpath if not provided
        if not display_pythonpath:
            if self.workspace_dir:
                pythonpath = self._get_pythonpath()
                display_pythonpath = (
                    self._replace_home_with_var(pythonpath)
                    if pythonpath
                    else "$HOME/dynamo/components/*/src"
                )
            else:
                display_pythonpath = "$HOME/dynamo/components/*/src"

        # Single source of truth for the export command
        print(
            f'\nSet PYTHONPATH for development:\nexport PYTHONPATH="{display_pythonpath}"\n'
        )

    # ====================================================================
    # USAGE EXAMPLES AND GUIDANCE
    # ====================================================================

    def _get_pythonpath(self) -> str:
        """Generate PYTHONPATH recommendation string.

        Returns:
            Colon-separated string of component source paths
            Example: '/home/ubuntu/dynamo/components/frontend/src:/home/ubuntu/dynamo/components/planner/src:/home/ubuntu/dynamo/components/backends/vllm/src'

        Note: Scans workspace for all component src directories and joins them for PYTHONPATH usage.
        """
        paths = []
        if not self.workspace_dir:
            return ""

        # Collect all component source paths
        comp_path = f"{self.workspace_dir}/components"
        if os.path.exists(comp_path):
            for item in os.listdir(comp_path):
                src_path = f"{comp_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)

        # Collect all backend source paths
        backend_path = f"{self.workspace_dir}/components/backends"
        if os.path.exists(backend_path):
            for item in os.listdir(backend_path):
                src_path = f"{backend_path}/{item}/src"
                if os.path.exists(src_path):
                    paths.append(src_path)

        return ":".join(paths)

    # ====================================================================
    # MAIN ORCHESTRATION
    # ====================================================================

    def run_all(self):
        """Run comprehensive check with all functionality.

        Performs complete dynamo package validation including:
        - Component discovery and import testing
        - Usage examples and troubleshooting guidance
        - Summary of results

        Console output: terse, tree-formatted sections
        """
        # Terse mode: no banner or separators

        # Execute all checks (package versions now shown in import testing headers)
        self.results["imports"] = self.test_imports()

        # Check if there were any import failures
        import_results = self.results.get("imports", {})
        has_failures = any(result.startswith("❌") for result in import_results.values())

        # Provide guidance (show only if all checks succeed and no errors flagged)
        had_errors_flag = bool(self.results.get("had_errors"))
        if not has_failures and not had_errors_flag:
            self._show_build_options()
        # If any errors found, exit with status 1
        had_errors = bool(self.results.get("had_errors"))
        if had_errors:
            sys.exit(1)


def main() -> None:
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive dynamo package checker")
    parser.add_argument(
        "--import-check-only", action="store_true", help="Only test imports"
    )
    parser.add_argument("--examples", action="store_true", help="Only show examples")
    parser.add_argument(
        "--build-options",
        action="store_true",
        help="Show build options for missing framework components",
    )
    parser.add_argument(
        "--try-pythonpath",
        action="store_true",
        help="Test imports with workspace component source directories in sys.path",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Explicit path to dynamo workspace; if set, bypass workspace auto-discovery",
    )

    args = parser.parse_args()
    checker = DynamoChecker(workspace_dir=args.path)
    # If --path is provided, validate it; do not exit early, but record error to display and for exit code
    if args.path:
        abs_path = os.path.abspath(args.path)
        if (not os.path.exists(abs_path)) or (
            not checker._is_dynamo_workspace(abs_path)
        ):
            checker._deferred_messages.append(
                f"❌ Error: invalid workspace path: {abs_path}"
            )

    # Set up sys.path if requested
    if args.try_pythonpath:
        checker._setup_pythonpath()

    if args.import_check_only:
        checker.test_imports()
        # Exit code handled inside run; reflect errors if set
        had_errors = bool(checker.results.get("had_errors"))
        if had_errors:
            sys.exit(1)
        # If examples are also requested and imports succeeded, show them
        if args.examples:
            checker._show_build_options()
        # If build options are also requested, show them
        if args.build_options:
            if checker.workspace_dir:
                pythonpath = checker._get_pythonpath()
                if pythonpath:
                    display_pythonpath = checker._replace_home_with_var(pythonpath)
                    checker._show_build_options(display_pythonpath)
                else:
                    print("❌ Error: Could not determine PYTHONPATH for build options")
            else:
                print("❌ Error: No dynamo workspace found for build options")
    elif args.build_options:
        # Show build options directly
        if checker.workspace_dir:
            pythonpath = checker._get_pythonpath()
            if pythonpath:
                display_pythonpath = checker._replace_home_with_var(pythonpath)
                checker._show_build_options(display_pythonpath)
            else:
                print("❌ Error: Could not determine PYTHONPATH for build options")
        else:
            print("❌ Error: No dynamo workspace found for build options")
    elif args.examples:
        # Only show examples, no system info or environment header
        checker._show_build_options()
    else:
        checker.run_all()


if __name__ == "__main__":
    main()
