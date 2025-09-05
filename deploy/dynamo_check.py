#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo System Information Checker

A comprehensive diagnostic tool that displays system configuration and Dynamo project status
in a hierarchical tree format. This script checks for:

- System resources (OS, CPU, memory, GPU)
- Development tools (Cargo/Rust, Maturin, Python)
- LLM frameworks (vllm, sglang, tensorrt_llm)
- Dynamo runtime and framework components
- Installation status and component availability

The output uses status indicators:
- ✅ Component found and working
- ❌ Component missing or error
- ⚠️ Warning condition
- ❓ Component not found (for optional items)

Exit codes:
- 0: All critical components are present
- 1: One or more errors detected (❌ status)

Example output:

System info (hostname=jensen-linux, IP=10.111.122.133)
├─ OS Ubuntu 24.04.1 LTS (Noble Numbat) (Linux 6.11.0-28-generic x86_64), Memory=26.7/125.5 GiB, Cores=32
├─ ✅ NVIDIA GPU NVIDIA RTX 6000 Ada Generation, driver 570.133.07, CUDA 12.8, Power=26.14/300.00 W, Memory=289/49140 MiB
├─ ✅ Cargo /usr/local/cargo/bin/cargo, cargo 1.89.0 (c24e10642 2025-06-23)
│  ├─ cargo home directory $HOME/dynamo/.build/.cargo (CARGO_HOME is set)
│  └─ cargo target directory $HOME/dynamo/.build/target (CARGO_TARGET_DIR is set)
│     ├─ Debug $HOME/dynamo/.build/target/debug, modified=2025-08-30 16:26:49 PDT
│     ├─ Release $HOME/dynamo/.build/target/release, modified=2025-08-30 18:21:12 PDT
│     └─ Binary $HOME/dynamo/.build/target/debug/libdynamo_llm_capi.so, modified=2025-08-30 16:25:37 PDT
├─ ✅ Maturin /opt/dynamo/venv/bin/maturin, maturin 1.9.3
├─ ✅ Python 3.12.3, /opt/dynamo/venv/bin/python
│  ├─ ✅ PyTorch 2.7.1+cu128, ✅torch.cuda.is_available
│  └─ PYTHONPATH $HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/llama_cpp/src:$HOME/dynamo/components/backends/mocker/src
├─ 🤖Framework
│  ├─ ✅ vllm 0.10.1.1, module=/opt/vllm/vllm/__init__.py, exec=/opt/dynamo/venv/bin/vllm
│  ├─ ❓ sglang -
│  └─ ❓ tensorrt_llm -
└─ Dynamo $HOME/dynamo, SHA: a03d29066, Date: 2025-08-30 16:22:29 PDT
   ├─ ✅ Runtime components ai-dynamo-runtime 0.4.1
   │  ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime-0.4.1.dist-info created=2025-08-30 19:14:29 PDT
   │  ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime.pth modified=2025-08-30 19:14:29 PDT
   │  │  └─ → $HOME/dynamo/lib/bindings/python/src
   │  ├─ ✅ dynamo._core             $HOME/dynamo/lib/bindings/python/src/dynamo/_core.cpython-312-x86_64-linux-gnu.so, modified=2025-08-30 19:14:29 PDT
   │  ├─ ✅ dynamo.logits_processing $HOME/dynamo/lib/bindings/python/src/dynamo/logits_processing/__init__.py
   │  ├─ ✅ dynamo.nixl_connect      $HOME/dynamo/lib/bindings/python/src/dynamo/nixl_connect/__init__.py
   │  ├─ ✅ dynamo.llm               $HOME/dynamo/lib/bindings/python/src/dynamo/llm/__init__.py
   │  └─ ✅ dynamo.runtime           $HOME/dynamo/lib/bindings/python/src/dynamo/runtime/__init__.py
   └─ ✅ Framework components ai-dynamo (via PYTHONPATH)
      ├─ ✅ dynamo.frontend  $HOME/dynamo/components/frontend/src/dynamo/frontend/__init__.py
      ├─ ✅ dynamo.llama_cpp $HOME/dynamo/components/backends/llama_cpp/src/dynamo/llama_cpp/__init__.py
      ├─ ✅ dynamo.mocker    $HOME/dynamo/components/backends/mocker/src/dynamo/mocker/__init__.py
      ├─ ✅ dynamo.planner   $HOME/dynamo/components/planner/src/dynamo/planner/__init__.py
      ├─ ✅ dynamo.sglang    $HOME/dynamo/components/backends/sglang/src/dynamo/sglang/__init__.py
      ├─ ✅ dynamo.trtllm    $HOME/dynamo/components/backends/trtllm/src/dynamo/trtllm/__init__.py
      └─ ✅ dynamo.vllm      $HOME/dynamo/components/backends/vllm/src/dynamo/vllm/__init__.py

Usage:
    python dynamo_check.py [--fast]

Options:
    --fast  Skip directory size calculations for faster output
"""

import datetime
import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class NodeStatus(Enum):
    """Status of a tree node"""

    OK = "ok"  # ✅ Success/available
    ERROR = "error"  # ❌ Error/not found
    WARNING = "warn"  # ⚠️ Warning
    INFO = "info"  # No symbol, just information
    NONE = "none"  # No status indicator
    UNKNOWN = "unknown"  # ❓ Unknown/not found


@dataclass
class NodeInfo:
    """Base class for all information nodes in the tree structure"""

    # Core properties
    label: str  # Main text/description
    desc: Optional[str] = None  # Primary value/description
    status: NodeStatus = NodeStatus.NONE  # Status indicator

    # Additional metadata as key-value pairs
    metadata: Dict[str, str] = field(default_factory=dict)

    # Tree structure
    children: List["NodeInfo"] = field(default_factory=list)

    # Display control
    show_symbol: bool = True  # Whether to show status symbol

    def add_child(self, child: "NodeInfo") -> "NodeInfo":
        """Add a child node and return it for chaining"""
        self.children.append(child)
        return child

    def add_metadata(self, key: str, value: str) -> "NodeInfo":
        """Add metadata key-value pair"""
        self.metadata[key] = value
        return self

    def render(
        self, prefix: str = "", is_last: bool = True, is_root: bool = True
    ) -> List[str]:
        """Render the tree node and its children as a list of strings"""
        lines = []

        # Determine the connector
        if not is_root:
            connector = "└─" if is_last else "├─"
            current_prefix = prefix + connector + " "
        else:
            current_prefix = ""

        # Build the line content
        line_parts = []

        # Add status symbol
        if self.show_symbol and self.status != NodeStatus.NONE:
            if self.status == NodeStatus.OK:
                line_parts.append("✅")
            elif self.status == NodeStatus.ERROR:
                line_parts.append("❌")
            elif self.status == NodeStatus.WARNING:
                line_parts.append("⚠️")
            elif self.status == NodeStatus.UNKNOWN:
                line_parts.append("❓")

        # Add label and value
        if self.desc:
            line_parts.append(f"{self.label} {self.desc}")
        else:
            line_parts.append(self.label)

        # Add metadata inline - consistent format for all
        if self.metadata:
            metadata_items = []
            for k, v in self.metadata.items():
                # Format all metadata consistently as "key=value"
                metadata_items.append(f"{k}={v}")

            if metadata_items:
                # Use consistent separator (comma) for all metadata
                metadata_str = ", ".join(metadata_items)
                line_parts[-1] += f", {metadata_str}"

        # Construct the full line
        line_content = " ".join(line_parts)
        if current_prefix or line_content:
            lines.append(current_prefix + line_content)

        # Render children
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            if is_root:
                child_prefix = ""
            else:
                child_prefix = prefix + ("   " if is_last else "│  ")
            lines.extend(child.render(child_prefix, is_last_child, False))

        return lines

    def print_tree(self) -> None:
        """Print the tree to console"""
        for line in self.render():
            print(line)

    def has_errors(self) -> bool:
        """Check if this node or any of its children have errors"""
        # Check if this node has an error
        if self.status == NodeStatus.ERROR:
            return True

        # Recursively check all children
        for child in self.children:
            if child.has_errors():
                return True

        return False

    def _replace_home_with_var(self, path: str) -> str:
        """Replace home directory with $HOME in path."""
        home = os.path.expanduser("~")
        if path.startswith(home):
            return path.replace(home, "$HOME", 1)
        return path

    def _format_timestamp_pdt(self, timestamp: float) -> str:
        """Format timestamp as PDT time string."""
        dt_utc = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        # Convert to PDT (UTC-7)
        dt_pdt = dt_utc - datetime.timedelta(hours=7)
        return dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")


class SystemInfo(NodeInfo):
    """Root node for system information"""

    def __init__(self, hostname: Optional[str] = None, fast_mode: bool = False):
        self.fast_mode = fast_mode
        if hostname is None:
            hostname = platform.node()

        # Get IP address
        ip_address = self._get_ip_address()

        # Format label with hostname and IP
        if ip_address:
            label = f"System info (hostname={hostname}, IP={ip_address})"
        else:
            label = f"System info (hostname={hostname})"

        super().__init__(label=label, status=NodeStatus.INFO)

        # Suppress Prometheus endpoint warnings from planner module
        self._suppress_planner_warnings()

        # Collect and add all system information
        # Add OS info
        self.add_child(OSInfo())

        # Add GPU info
        gpu_info = GPUInfo()
        # Always add GPU info so we can see errors like "nvidia-smi not found"
        self.add_child(gpu_info)

        # Add Cargo (always show, even if not found)
        self.add_child(CargoInfo(fast_mode=self.fast_mode))

        # Add Maturin (Python-Rust build tool)
        self.add_child(MaturinInfo())

        # Add Python info
        self.add_child(PythonInfo())

        # Add Framework info (vllm, sglang, tensorrt_llm)
        self.add_child(FrameworkInfo())

        # Add Dynamo workspace info (always show, even if not found)
        self.add_child(DynamoInfo(fast_mode=self.fast_mode))

    def _get_ip_address(self) -> Optional[str]:
        """Get the primary IP address of the system."""
        try:
            import socket

            # Get hostname
            hostname = socket.gethostname()
            # Get IP address
            ip_address = socket.gethostbyname(hostname)
            # Filter out localhost
            if ip_address.startswith("127."):
                # Try to get external IP by connecting to a public DNS
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # Connect to Google DNS (doesn't actually send data)
                    s.connect(("8.8.8.8", 80))
                    ip_address = s.getsockname()[0]
                finally:
                    s.close()
            return ip_address
        except Exception:
            return None

    def _suppress_planner_warnings(self) -> None:
        """Suppress Prometheus endpoint warnings from planner module during import testing."""
        # The planner module logs a warning about Prometheus endpoint when imported
        # outside of a Kubernetes cluster. Suppress this for cleaner output.
        planner_logger = logging.getLogger("dynamo.planner.defaults")
        planner_logger.setLevel(logging.ERROR)
        # Also suppress the defaults._get_default_prometheus_endpoint logger
        defaults_logger = logging.getLogger("defaults._get_default_prometheus_endpoint")
        defaults_logger.setLevel(logging.ERROR)


class OSInfo(NodeInfo):
    """Operating system information"""

    def __init__(self):
        # Collect OS information
        uname = platform.uname()

        # Try to get distribution info
        distro = ""
        version = ""
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("NAME="):
                            distro = line.split("=", 1)[1].strip().strip('"')
                        elif line.startswith("VERSION="):
                            version = line.split("=", 1)[1].strip().strip('"')
        except Exception:
            pass

        # Get memory info
        mem_used_gb = None
        mem_total_gb = None
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meminfo[k.strip()] = v.strip()

                if "MemTotal" in meminfo and "MemAvailable" in meminfo:
                    total_kb = float(meminfo["MemTotal"].split()[0])
                    avail_kb = float(meminfo["MemAvailable"].split()[0])
                    mem_used_gb = (total_kb - avail_kb) / (1024 * 1024)
                    mem_total_gb = total_kb / (1024 * 1024)
        except Exception:
            pass

        # Get CPU cores
        cores = os.cpu_count()

        # Build the value string
        if distro:
            value = f"{distro} {version} ({uname.system} {uname.release} {uname.machine})".strip()
        else:
            value = f"{uname.system} {uname.release} {uname.machine}"

        super().__init__(label="OS", desc=value, status=NodeStatus.INFO)

        # Add memory and cores as metadata
        if mem_used_gb is not None and mem_total_gb is not None:
            self.add_metadata("Memory", f"{mem_used_gb:.1f}/{mem_total_gb:.1f} GiB")
            if mem_total_gb > 0 and (mem_used_gb / mem_total_gb) >= 0.9:
                self.status = NodeStatus.WARNING
        if cores:
            self.add_metadata("Cores", str(cores))


class GPUInfo(NodeInfo):
    """NVIDIA GPU information"""

    def __init__(self):
        # Find nvidia-smi executable (check multiple paths)
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            # Check common paths if `which` fails
            for candidate in [
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
                "/usr/local/nvidia/bin/nvidia-smi",
            ]:
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    nvidia_smi = candidate
                    break

        if not nvidia_smi:
            super().__init__(
                label="NVIDIA GPU", desc="nvidia-smi not found", status=NodeStatus.ERROR
            )
            return

        try:
            # Get GPU list
            result = subprocess.run(
                [nvidia_smi, "-L"], capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                # Capture error details from stderr or stdout
                error_msg = "nvidia-smi failed"
                if result.stderr and result.stderr.strip():
                    # Get first line of error for concise display
                    error_lines = result.stderr.strip().splitlines()
                    if error_lines:
                        error_msg = error_lines[0].strip()
                        # Make NVML error more user-friendly
                        if "Failed to initialize NVML" in error_msg:
                            error_msg = (
                                "No NVIDIA GPU detected (NVML initialization failed)"
                            )
                elif result.stdout and result.stdout.strip():
                    error_lines = result.stdout.strip().splitlines()
                    if error_lines:
                        error_msg = error_lines[0].strip()
                        # Make NVML error more user-friendly
                        if "Failed to initialize NVML" in error_msg:
                            error_msg = (
                                "No NVIDIA GPU detected (NVML initialization failed)"
                            )
                super().__init__(
                    label="NVIDIA GPU", desc=error_msg, status=NodeStatus.ERROR
                )
                return

            # Parse GPU names
            gpu_names = []
            lines = result.stdout.strip().splitlines()
            for line in lines:
                # Example: "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)"
                if ":" in line:
                    gpu_name = line.split(":", 1)[1].split("(")[0].strip()
                    gpu_names.append(gpu_name)

            # Check for zero GPUs
            if not gpu_names:
                # Get driver and CUDA even for zero GPUs
                driver, cuda = self._get_driver_cuda_versions(nvidia_smi)
                driver_cuda_str = ""
                if driver or cuda:
                    parts = []
                    if driver:
                        parts.append(f"driver {driver}")
                    if cuda:
                        parts.append(f"CUDA {cuda}")
                    driver_cuda_str = f", {', '.join(parts)}"
                super().__init__(
                    label="NVIDIA GPU",
                    desc=f"not detected{driver_cuda_str}",
                    status=NodeStatus.ERROR,
                )
                return

            # Get driver and CUDA versions
            driver, cuda = self._get_driver_cuda_versions(nvidia_smi)

            # Handle single vs multiple GPUs
            if len(gpu_names) == 1:
                # Single GPU - compact format
                value = gpu_names[0]
                if driver or cuda:
                    driver_cuda = []
                    if driver:
                        driver_cuda.append(f"driver {driver}")
                    if cuda:
                        driver_cuda.append(f"CUDA {cuda}")
                    value += f", {', '.join(driver_cuda)}"

                super().__init__(label="NVIDIA GPU", desc=value, status=NodeStatus.OK)

                # Add power and memory metadata for single GPU
                self._add_power_memory_info(nvidia_smi, 0)
            else:
                # Multiple GPUs - show count in main label
                value = f"{len(gpu_names)} GPUs"
                if driver or cuda:
                    driver_cuda = []
                    if driver:
                        driver_cuda.append(f"driver {driver}")
                    if cuda:
                        driver_cuda.append(f"CUDA {cuda}")
                    value += f", {', '.join(driver_cuda)}"

                super().__init__(label="NVIDIA GPU", desc=value, status=NodeStatus.OK)

                # Add each GPU as a child node
                for i, name in enumerate(gpu_names):
                    gpu_child = NodeInfo(
                        label=f"GPU {i}", desc=name, status=NodeStatus.OK
                    )
                    # Add power and memory for this specific GPU
                    power_mem = self._get_power_memory_string(nvidia_smi, i)
                    if power_mem:
                        gpu_child.add_metadata("Stats", power_mem)
                    self.add_child(gpu_child)

        except Exception:
            super().__init__(
                label="NVIDIA GPU", desc="detection failed", status=NodeStatus.ERROR
            )

    def _get_driver_cuda_versions(
        self, nvidia_smi: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get NVIDIA driver and CUDA versions using query method."""
        driver, cuda = None, None
        try:
            # Use query method for more reliable detection
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                driver = result.stdout.strip().splitlines()[0].strip()

            # Try to get CUDA version from nvidia-smi output
            result = subprocess.run(
                [nvidia_smi], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                import re

                m = re.search(r"CUDA Version:\s*([0-9.]+)", result.stdout)
                if m:
                    cuda = m.group(1)
        except Exception:
            pass
        return driver, cuda

    def _add_power_memory_info(self, nvidia_smi: str, gpu_index: int = 0):
        """Add power and memory metadata for a specific GPU."""
        power_mem = self._get_power_memory_string(nvidia_smi, gpu_index)
        if power_mem:
            # Split into Power and Memory parts
            if "; " in power_mem:
                parts = power_mem.split("; ")
                for part in parts:
                    if part.startswith("Power:"):
                        self.add_metadata("Power", part.replace("Power: ", ""))
                    elif part.startswith("Memory:"):
                        self.add_metadata("Memory", part.replace("Memory: ", ""))

    def _get_power_memory_string(
        self, nvidia_smi: str, gpu_index: int = 0
    ) -> Optional[str]:
        """Get power and memory info string for a specific GPU."""
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=power.draw,power.limit,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().splitlines()
                if gpu_index < len(lines):
                    parts = lines[gpu_index].split(",")
                    if len(parts) >= 4:
                        power_draw = parts[0].strip()
                        power_limit = parts[1].strip()
                        mem_used = parts[2].strip()
                        mem_total = parts[3].strip()

                        info_parts = []
                        if power_draw and power_limit:
                            info_parts.append(f"Power: {power_draw}/{power_limit} W")

                        if mem_used and mem_total:
                            # Add warning if memory usage is 90% or higher
                            warning = ""
                            try:
                                if float(mem_used) / float(mem_total) >= 0.9:
                                    warning = " ⚠️"
                            except Exception:
                                pass
                            info_parts.append(
                                f"Memory: {mem_used}/{mem_total} MiB{warning}"
                            )

                        if info_parts:
                            return "; ".join(info_parts)
        except Exception:
            pass
        return None


class CargoInfo(NodeInfo):
    """Cargo tool information"""

    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode
        cargo_path = shutil.which("cargo")
        cargo_version = None

        # Get cargo version
        if cargo_path:
            try:
                result = subprocess.run(
                    ["cargo", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    cargo_version = result.stdout.strip()
            except Exception:
                pass

        if not cargo_path and not cargo_version:
            super().__init__(
                label="Cargo",
                desc="not found, install Rust toolchain to see cargo target directory",
                status=NodeStatus.ERROR,
            )
            return

        # Initialize with cargo path and version
        value = ""
        if cargo_path:
            value = cargo_path
        if cargo_version:
            value += f", {cargo_version}" if value else cargo_version

        super().__init__(label="Cargo", desc=value, status=NodeStatus.OK)

        # Get cargo home directory
        cargo_home = os.environ.get("CARGO_HOME")
        if not cargo_home:
            cargo_home = os.path.expanduser("~/.cargo")

        if cargo_home and os.path.exists(cargo_home):
            cargo_home_env = os.environ.get("CARGO_HOME")
            display_cargo_home = self._replace_home_with_var(cargo_home)
            home_value = display_cargo_home
            if cargo_home_env:
                home_value += " (CARGO_HOME is set)"

            home_node = NodeInfo(
                label="cargo home directory", desc=home_value, status=NodeStatus.INFO
            )
            self.add_child(home_node)

        # Get cargo target directory
        cargo_target = self._get_cargo_target_directory()
        if cargo_target and os.path.exists(cargo_target):
            cargo_target_env = os.environ.get("CARGO_TARGET_DIR")
            display_cargo_target = self._replace_home_with_var(cargo_target)

            # Calculate total directory size (skip if fast mode)
            size_str = ""
            if not self.fast_mode:
                total_size_gb = self._get_directory_size_gb(cargo_target)
                size_str = (
                    f", {total_size_gb:.1f} GB" if total_size_gb is not None else ""
                )

            target_value = display_cargo_target + size_str
            if cargo_target_env:
                target_value += " (CARGO_TARGET_DIR is set)"

            target_node = NodeInfo(
                label="cargo target directory",
                desc=target_value,
                status=NodeStatus.INFO,
            )
            self.add_child(target_node)

            # Add debug/release/binary info as children of target directory
            self._add_build_info(target_node, cargo_target)

    def _get_directory_size_gb(self, directory: str) -> Optional[float]:
        """Get the size of a directory in GB."""
        try:
            # Use du command to get directory size in bytes
            result = subprocess.run(
                ["du", "-sb", directory], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                # Parse output: "size_in_bytes\tdirectory_path"
                size_bytes = int(result.stdout.split()[0])
                # Convert to GB
                size_gb = size_bytes / (1024**3)
                return size_gb
        except Exception:
            pass
        return None

    def _get_cargo_target_directory(self) -> Optional[str]:
        """Get cargo target directory using cargo metadata."""
        try:
            # Use DynamoInfo's static method to find workspace
            workspace_dir = DynamoInfo.find_workspace()

            # Run cargo metadata command to get target directory
            cmd_args = ["cargo", "metadata", "--format-version=1", "--no-deps"]
            kwargs: Dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": 10,
            }

            # Add cwd if workspace_dir was found
            if workspace_dir and os.path.isdir(workspace_dir):
                kwargs["cwd"] = workspace_dir

            result = subprocess.run(cmd_args, **kwargs)

            if result.returncode == 0:
                # Parse JSON output to extract target_directory
                metadata = json.loads(result.stdout)
                return metadata.get("target_directory")
        except Exception:
            pass
        return None

    def _add_build_info(self, parent_node: NodeInfo, cargo_target: str):
        """Add debug/release/binary information as children of target directory."""
        debug_dir = os.path.join(cargo_target, "debug")
        release_dir = os.path.join(cargo_target, "release")

        # Check debug directory
        if os.path.exists(debug_dir):
            display_debug = self._replace_home_with_var(debug_dir)
            debug_value = display_debug

            # Add size (skip if fast mode)
            if not self.fast_mode:
                debug_size_gb = self._get_directory_size_gb(debug_dir)
                if debug_size_gb is not None:
                    debug_value += f", {debug_size_gb:.1f} GB"

            try:
                debug_mtime = os.path.getmtime(debug_dir)
                debug_time = self._format_timestamp_pdt(debug_mtime)
                debug_value += f", modified={debug_time}"
            except Exception:
                debug_value += " (unable to read timestamp)"

            debug_node = NodeInfo(
                label="Debug", desc=debug_value, status=NodeStatus.INFO
            )
            parent_node.add_child(debug_node)

        # Check release directory
        if os.path.exists(release_dir):
            display_release = self._replace_home_with_var(release_dir)
            release_value = display_release

            # Add size (skip if fast mode)
            if not self.fast_mode:
                release_size_gb = self._get_directory_size_gb(release_dir)
                if release_size_gb is not None:
                    release_value += f", {release_size_gb:.1f} GB"

            try:
                release_mtime = os.path.getmtime(release_dir)
                release_time = self._format_timestamp_pdt(release_mtime)
                release_value += f", modified={release_time}"
            except Exception:
                release_value += " (unable to read timestamp)"

            release_node = NodeInfo(
                label="Release", desc=release_value, status=NodeStatus.INFO
            )
            parent_node.add_child(release_node)

        # Find *.so file
        so_file = self._find_so_file(cargo_target)
        if so_file:
            display_so = self._replace_home_with_var(so_file)
            so_value = display_so

            # Add file size (skip if fast mode)
            if not self.fast_mode:
                try:
                    file_size_bytes = os.path.getsize(so_file)
                    file_size_mb = file_size_bytes / (1024**2)
                    so_value += f", {file_size_mb:.1f} MB"
                except Exception:
                    pass

            try:
                so_mtime = os.path.getmtime(so_file)
                so_time = self._format_timestamp_pdt(so_mtime)
                so_value += f", modified={so_time}"
            except Exception:
                so_value += " (unable to read timestamp)"

            binary_node = NodeInfo(
                label="Binary", desc=so_value, status=NodeStatus.INFO
            )
            parent_node.add_child(binary_node)

    def _find_so_file(self, target_directory: str) -> Optional[str]:
        """Find the compiled *.so file in target directory."""
        # Check common locations for .so files
        search_dirs = [
            os.path.join(target_directory, "debug"),
            os.path.join(target_directory, "release"),
            target_directory,
        ]

        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            # Walk through directory looking for .so files
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(".so"):
                            return os.path.join(root, file)
                    # Don't recurse too deep
                    if root.count(os.sep) - search_dir.count(os.sep) > 2:
                        dirs[:] = []  # Stop recursion
            except Exception:
                pass

        return None


class MaturinInfo(NodeInfo):
    """Maturin tool information (Python-Rust build tool)"""

    def __init__(self):
        maturin_path = shutil.which("maturin")
        if not maturin_path:
            super().__init__(label="Maturin", desc="not found", status=NodeStatus.ERROR)
            # Add installation hint as a child node
            install_hint = NodeInfo(
                label="Install with",
                desc="uv pip install maturin[patchelf]",
                status=NodeStatus.INFO,
            )
            self.add_child(install_hint)
            return

        try:
            result = subprocess.run(
                ["maturin", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                # Include the maturin binary path like Cargo and Git do
                display_maturin_path = self._replace_home_with_var(maturin_path)
                super().__init__(
                    label="Maturin",
                    desc=f"{display_maturin_path}, {version}",
                    status=NodeStatus.OK,
                )
                return
        except Exception:
            pass

        super().__init__(label="Maturin", desc="not found", status=NodeStatus.ERROR)


class PythonInfo(NodeInfo):
    """Python installation information"""

    def __init__(self):
        py_version = platform.python_version()
        py_exec = sys.executable or "python"
        display_py_exec = self._replace_home_with_var(py_exec)

        super().__init__(
            label="Python",
            desc=f"{py_version}, {display_py_exec}",
            status=NodeStatus.OK if os.path.exists(py_exec) else NodeStatus.ERROR,
        )

        # Check for PyTorch (optional)
        try:
            torch = __import__("torch")
            version = getattr(torch, "__version__", "installed")

            # Check CUDA availability
            cuda_status = None
            if hasattr(torch, "cuda"):
                try:
                    cuda_available = torch.cuda.is_available()
                    cuda_status = (
                        "✅torch.cuda.is_available"
                        if cuda_available
                        else "❌torch.cuda.is_available"
                    )
                except Exception:
                    pass

            # Get installation path
            install_path = None
            if hasattr(torch, "__file__") and torch.__file__:
                file_path = torch.__file__
                if "site-packages" in file_path:
                    parts = file_path.split(os.sep)
                    for i, part in enumerate(parts):
                        if part == "site-packages":
                            install_path = os.sep.join(parts[: i + 1])
                            break
                elif file_path:
                    install_path = os.path.dirname(file_path)

                if install_path:
                    install_path = self._replace_home_with_var(install_path)

            package_info = PythonPackageInfo(
                package_name="PyTorch",
                version=version,
                cuda_status=cuda_status,
                install_path=install_path,
                is_framework=False,
            )
            self.add_child(package_info)
        except ImportError:
            pass  # PyTorch is optional, don't show if not installed

        # Add PYTHONPATH
        pythonpath = os.environ.get("PYTHONPATH", "")
        self.add_child(PythonPathInfo(pythonpath))


class FrameworkInfo(NodeInfo):
    """LLM Framework information"""

    def __init__(self):
        super().__init__(label="🤖Framework", status=NodeStatus.INFO)

        # Check for framework packages (mandatory to show)
        frameworks_to_check = [
            ("vllm", "vLLM"),
            ("sglang", "Sglang"),
            ("tensorrt_llm", "tensorRT LLM"),
        ]

        for module_name, display_name in frameworks_to_check:
            # Special handling for TensorRT-LLM to avoid NVML crashes
            if module_name == "tensorrt_llm":
                # Check if it's installed in system packages first
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                system_packages = [
                    f"/usr/local/lib/python{python_version}/dist-packages",
                    f"/usr/lib/python{python_version}/dist-packages",
                ]

                found_in_system = False
                for pkg_path in system_packages:
                    if os.path.exists(pkg_path):
                        tensorrt_dirs = [
                            d for d in os.listdir(pkg_path) if "tensorrt_llm" in d
                        ]
                        if tensorrt_dirs:
                            found_in_system = True
                            # Try to get version safely
                            try:
                                result = subprocess.run(
                                    [
                                        sys.executable,
                                        "-c",
                                        "import tensorrt_llm; print(tensorrt_llm.__version__)",
                                    ],
                                    capture_output=True,
                                    text=True,
                                    timeout=10,
                                )
                                if result.returncode == 0:
                                    version = result.stdout.strip()
                                    package_info = PythonPackageInfo(
                                        package_name=display_name,
                                        version=version,
                                        module_path=f"{pkg_path}/tensorrt_llm/__init__.py",
                                        is_framework=True,
                                        is_installed=True,
                                    )
                                else:
                                    package_info = PythonPackageInfo(
                                        package_name=display_name,
                                        version=f"Found in {pkg_path} but not importable",
                                        is_framework=True,
                                        is_installed=True,
                                    )
                                self.add_child(package_info)
                                break
                            except (
                                subprocess.TimeoutExpired,
                                subprocess.CalledProcessError,
                            ):
                                package_info = PythonPackageInfo(
                                    package_name=display_name,
                                    version=f"Found in {pkg_path} but not importable",
                                    is_framework=True,
                                    is_installed=True,
                                )
                                self.add_child(package_info)
                                break

                if not found_in_system:
                    package_info = PythonPackageInfo(
                        package_name=display_name,
                        version="-",
                        is_framework=True,
                        is_installed=False,
                    )
                    self.add_child(package_info)
                continue

            # Regular import for other frameworks
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "installed")

                # Get module path
                module_path = None
                if hasattr(module, "__file__") and module.__file__:
                    module_path = self._replace_home_with_var(module.__file__)

                # Get executable path
                exec_path = None
                exec_path_raw = shutil.which(module_name)
                if exec_path_raw:
                    exec_path = self._replace_home_with_var(exec_path_raw)

                package_info = PythonPackageInfo(
                    package_name=display_name,
                    version=version,
                    module_path=module_path,
                    exec_path=exec_path,
                    is_framework=True,
                    is_installed=True,
                )
                self.add_child(package_info)
            except (ImportError, Exception):
                # Framework not installed - show with "-"
                package_info = PythonPackageInfo(
                    package_name=display_name,
                    version="-",
                    is_framework=True,
                    is_installed=False,
                )
                self.add_child(package_info)


class PythonPackageInfo(NodeInfo):
    """Python package information"""

    def __init__(
        self,
        package_name: str,
        version: str,
        cuda_status: Optional[str] = None,
        module_path: Optional[str] = None,
        exec_path: Optional[str] = None,
        install_path: Optional[str] = None,
        is_framework: bool = False,
        is_installed: bool = True,
    ):
        # Build display value
        display_value = version

        # Determine status based on whether package is installed
        if not is_installed or version == "-":
            # Framework not found - show with "-" and use UNKNOWN status for ❓ symbol
            display_value = "-"
            status = NodeStatus.UNKNOWN  # Show ❓ for not found frameworks
        else:
            status = NodeStatus.OK

            # Add CUDA status for PyTorch
            if cuda_status:
                display_value = f"{version}, {cuda_status}"
                # Don't add install path for PyTorch with CUDA status
            # For frameworks, add module and exec paths
            elif is_framework and (module_path or exec_path):
                parts = [version]
                if module_path:
                    parts.append(f"module={module_path}")
                if exec_path:
                    parts.append(f"exec={exec_path}")
                display_value = ", ".join(parts)
            # For regular packages, add install path
            elif install_path:
                display_value = f"{version} ({install_path})"

        super().__init__(label=package_name, desc=display_value, status=status)


class PythonPathInfo(NodeInfo):
    """PYTHONPATH environment variable information"""

    def __init__(self, pythonpath: str):
        if pythonpath:
            # Split by colon and replace home in each path
            paths = pythonpath.split(":")
            display_paths = [self._replace_home_with_var(p) for p in paths]
            display_pythonpath = ":".join(display_paths)
            status = NodeStatus.INFO
        else:
            display_pythonpath = "not set"
            status = NodeStatus.WARNING  # Show warning when PYTHONPATH is not set

        super().__init__(label="PYTHONPATH", desc=display_pythonpath, status=status)


class DynamoRuntimeInfo(NodeInfo):
    """Dynamo runtime components information"""

    def __init__(self, workspace_dir: str, fast_mode: bool = False):
        self.fast_mode = fast_mode
        # Try to get package version
        import importlib.metadata

        try:
            version = importlib.metadata.version("ai-dynamo-runtime")
            runtime_value = f"ai-dynamo-runtime {version}"
            is_installed = True
        except Exception:
            runtime_value = "ai-dynamo-runtime - Not installed"
            is_installed = False

        super().__init__(
            label="Runtime components",
            desc=runtime_value,
            status=NodeStatus.INFO,  # Will update based on components found
        )

        # Add package info if installed
        if is_installed:
            # Add dist-info directory
            dist_info = self._find_dist_info()
            if dist_info:
                self.add_child(dist_info)

            # Add .pth file
            pth_file = self._find_pth_file()
            if pth_file:
                self.add_child(pth_file)

        # Discover runtime components from source
        components = self._discover_runtime_components(workspace_dir)

        # Find where each component actually is and add them
        if components:
            # Calculate max width for alignment
            max_len = max(len(comp) for comp in components)

            components_found = False
            for component in components:
                try:
                    # Try to import to find actual location
                    module = __import__(component, fromlist=[""])
                    module_path = getattr(module, "__file__", None)

                    if module_path:
                        # Add timestamp for .so files
                        timestamp_str = ""
                        if module_path.endswith(".so"):
                            try:
                                stat = os.stat(module_path)
                                timestamp = self._format_timestamp_pdt(stat.st_mtime)
                                timestamp_str = f", modified={timestamp}"
                            except Exception:
                                pass

                        display_path = self._replace_home_with_var(module_path)
                        padded_name = f"{component:<{max_len}}"
                        module_node = NodeInfo(
                            label=f"✅ {padded_name}",
                            desc=f"{display_path}{timestamp_str}",
                            status=NodeStatus.NONE,
                        )
                        self.add_child(module_node)
                        components_found = True
                except ImportError as e:
                    # Module not importable - show as error
                    padded_name = f"{component:<{max_len}}"
                    error_msg = str(e) if str(e) else "Import failed"
                    module_node = NodeInfo(
                        label=padded_name, desc=error_msg, status=NodeStatus.ERROR
                    )
                    self.add_child(module_node)
                    # Don't set components_found to True for failed imports

            # Update status and value based on whether we found components
            if components_found:
                self.status = NodeStatus.OK
                # If not installed but components work via PYTHONPATH, update the message
                if not is_installed:
                    self.desc = "ai-dynamo-runtime (via PYTHONPATH)"
            else:
                self.status = NodeStatus.ERROR
        else:
            # No components discovered at all
            self.status = NodeStatus.ERROR

        # Final check: if no children at all (no components found), ensure it's an error
        if not self.children:
            self.status = NodeStatus.ERROR

    def _discover_runtime_components(self, workspace_dir: str) -> list:
        """Discover ai-dynamo-runtime components from filesystem.

        Returns:
            List of runtime component module names
            Example: ['dynamo._core', 'dynamo.nixl_connect', 'dynamo.llm', 'dynamo.runtime']

        Note: Always includes 'dynamo._core' (compiled Rust module), then scans
              lib/bindings/python/src/dynamo/ for additional components.
        """
        components = ["dynamo._core"]  # Always include compiled Rust module

        if not workspace_dir:
            return components

        # Scan runtime components (llm, runtime, nixl_connect, etc.)
        runtime_path = os.path.join(workspace_dir, "lib/bindings/python/src/dynamo")
        if not os.path.exists(runtime_path):
            return components

        for item in os.listdir(runtime_path):
            item_path = os.path.join(runtime_path, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "__init__.py")
            ):
                components.append(f"dynamo.{item}")

        return components

    def _find_dist_info(self) -> Optional[NodeInfo]:
        """Find the dist-info directory for ai-dynamo-runtime."""
        import site

        for site_dir in site.getsitepackages():
            pattern = os.path.join(site_dir, "ai_dynamo_runtime*.dist-info")
            matches = glob.glob(pattern)
            if matches:
                path = matches[0]
                display_path = self._replace_home_with_var(path)
                try:
                    stat = os.stat(path)
                    timestamp = self._format_timestamp_pdt(stat.st_ctime)
                    return NodeInfo(
                        label=display_path,
                        desc=f"created={timestamp}",
                        status=NodeStatus.INFO,
                    )
                except Exception:
                    return NodeInfo(label=display_path, status=NodeStatus.INFO)
        return None

    def _find_pth_file(self) -> Optional[NodeInfo]:
        """Find the .pth file for ai-dynamo-runtime."""
        import site

        for site_dir in site.getsitepackages():
            pth_path = os.path.join(site_dir, "ai_dynamo_runtime.pth")
            if os.path.exists(pth_path):
                display_path = self._replace_home_with_var(pth_path)
                try:
                    stat = os.stat(pth_path)
                    timestamp = self._format_timestamp_pdt(stat.st_mtime)
                    node = NodeInfo(
                        label=display_path,
                        desc=f"modified={timestamp}",
                        status=NodeStatus.INFO,
                    )

                    # Read where it points to
                    with open(pth_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            display_content = self._replace_home_with_var(content)
                            points_to = NodeInfo(
                                label="→", desc=display_content, status=NodeStatus.INFO
                            )
                            node.add_child(points_to)

                    return node
                except Exception:
                    return NodeInfo(label=display_path, status=NodeStatus.INFO)
        return None


class DynamoFrameworkInfo(NodeInfo):
    """Dynamo framework components information"""

    def __init__(self, workspace_dir: str, fast_mode: bool = False):
        self.fast_mode = fast_mode
        # Try to get package version
        import importlib.metadata

        try:
            version = importlib.metadata.version("ai-dynamo")
            framework_value = f"ai-dynamo {version}"
            is_installed = True
        except Exception:
            framework_value = "ai-dynamo - Not installed"
            is_installed = False

        super().__init__(
            label="Framework components",
            desc=framework_value,
            status=NodeStatus.INFO,  # Will update based on components found
        )

        # Add package info if installed
        if is_installed:
            import glob
            import site

            for site_dir in site.getsitepackages():
                # Look specifically for ai_dynamo (not ai_dynamo_runtime)
                dist_pattern = os.path.join(site_dir, "ai_dynamo-*.dist-info")
                matches = glob.glob(dist_pattern)
                if matches:
                    path = matches[0]
                    display_path = self._replace_home_with_var(path)
                    try:
                        stat = os.stat(path)
                        timestamp = self._format_timestamp_pdt(stat.st_ctime)
                        dist_node = NodeInfo(
                            label=display_path,
                            desc=f"created={timestamp}",
                            status=NodeStatus.INFO,
                        )
                        self.add_child(dist_node)
                    except Exception:
                        dist_node = NodeInfo(label=display_path, status=NodeStatus.INFO)
                        self.add_child(dist_node)
                    break

        # Discover framework components from source
        components = self._discover_framework_components(workspace_dir)

        # Find where each component actually is and add them
        if components:
            # Sort components for consistent output
            components.sort()

            # Calculate max width for alignment
            max_len = max(len(comp) for comp in components)

            components_found = False
            for component in components:
                try:
                    # Try to import to find actual location
                    module = __import__(component, fromlist=[""])
                    module_path = getattr(module, "__file__", None)

                    if module_path:
                        display_path = self._replace_home_with_var(module_path)
                        padded_name = f"{component:<{max_len}}"
                        component_node = NodeInfo(
                            label=f"✅ {padded_name}",
                            desc=display_path,
                            status=NodeStatus.NONE,
                        )
                        self.add_child(component_node)
                        components_found = True
                except ImportError as e:
                    # Module not importable - show as error
                    padded_name = f"{component:<{max_len}}"
                    error_msg = str(e) if str(e) else "Import failed"
                    component_node = NodeInfo(
                        label=padded_name, desc=error_msg, status=NodeStatus.ERROR
                    )
                    self.add_child(component_node)
                    # Don't set components_found to True for failed imports

            # Update status and value based on whether we found components
            if components_found:
                self.status = NodeStatus.OK
                # If not installed but components work via PYTHONPATH, update the message
                if not is_installed:
                    self.desc = "ai-dynamo (via PYTHONPATH)"
            else:
                self.status = NodeStatus.ERROR
        else:
            # No components discovered at all
            self.status = NodeStatus.ERROR

    def _discover_framework_components(self, workspace_dir: str) -> list:
        """Discover ai-dynamo framework components from filesystem.

        Returns:
            List of framework component module names
            Example: ['dynamo.frontend', 'dynamo.planner', 'dynamo.vllm', 'dynamo.sglang', 'dynamo.llama_cpp']

        Note: Scans components/ and components/backends/ directories for modules with __init__.py files.
        """
        components: List[str] = []

        if not workspace_dir:
            return components

        # Scan components directory (frontend, planner, etc.)
        components_path = os.path.join(workspace_dir, "components")
        if os.path.exists(components_path):
            for item in os.listdir(components_path):
                item_path = os.path.join(components_path, item)
                if os.path.isdir(item_path):
                    # Check for dynamo module in src
                    module_path = os.path.join(
                        item_path, "src", "dynamo", item, "__init__.py"
                    )
                    if os.path.exists(module_path):
                        components.append(f"dynamo.{item}")

        # Scan backends directory (vllm, sglang, trtllm, etc.)
        backends_path = os.path.join(workspace_dir, "components", "backends")
        if os.path.exists(backends_path):
            for item in os.listdir(backends_path):
                item_path = os.path.join(backends_path, item)
                if os.path.isdir(item_path):
                    # Check for dynamo module in src
                    module_path = os.path.join(
                        item_path, "src", "dynamo", item, "__init__.py"
                    )
                    if os.path.exists(module_path):
                        components.append(f"dynamo.{item}")

        return components


class DynamoInfo(NodeInfo):
    """Dynamo workspace information"""

    def __init__(self, fast_mode: bool = False):
        self.fast_mode = fast_mode

        # Find workspace directory
        workspace_dir = DynamoInfo.find_workspace()

        if not workspace_dir:
            # Show error when workspace is not found
            super().__init__(
                label="Dynamo",
                desc="workspace not found - cannot detect Runtime and Framework components",
                status=NodeStatus.ERROR,
            )
            # Add helpful information about where we looked
            search_paths = NodeInfo(
                label="Searched in",
                desc="current dir, ~/dynamo, DYNAMO_HOME, /workspace",
                status=NodeStatus.INFO,
            )
            self.add_child(search_paths)
            hint = NodeInfo(
                label="Hint",
                desc="Run from a Dynamo workspace directory or set DYNAMO_HOME",
                status=NodeStatus.INFO,
            )
            self.add_child(hint)
            return

        # Get git info
        sha, date = self._get_git_info(workspace_dir)

        # Build main label
        display_workspace = self._replace_home_with_var(workspace_dir)
        if sha and date:
            value = f"{display_workspace}, SHA: {sha}, Date: {date}"
        else:
            value = display_workspace

        super().__init__(label="Dynamo", desc=value, status=NodeStatus.INFO)

        # Always add runtime components
        runtime_info = DynamoRuntimeInfo(workspace_dir, fast_mode=self.fast_mode)
        self.add_child(runtime_info)

        # Always add framework components
        framework_info = DynamoFrameworkInfo(workspace_dir, fast_mode=self.fast_mode)
        self.add_child(framework_info)

    def _get_git_info(self, workspace_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """Get git SHA and date for the workspace."""
        try:
            # Get short SHA
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                timeout=5,
            )
            sha = result.stdout.strip() if result.returncode == 0 else None

            # Get commit date
            result = subprocess.run(
                ["git", "show", "-s", "--format=%ci", "HEAD"],
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Convert to PDT format
                date_str = result.stdout.strip()
                # Parse and format as PDT
                try:
                    # Parse the git date (format: 2025-08-30 23:22:29 +0000)
                    import datetime as dt_module

                    # Split off timezone info
                    date_part = date_str.rsplit(" ", 1)[0]
                    dt = dt_module.datetime.strptime(date_part, "%Y-%m-%d %H:%M:%S")
                    # Convert to PDT (UTC-7)
                    dt_pdt = dt - dt_module.timedelta(hours=7)
                    date = dt_pdt.strftime("%Y-%m-%d %H:%M:%S PDT")
                except Exception:
                    date = date_str
            else:
                date = None

            return sha, date
        except Exception:
            return None, None

    @staticmethod
    def find_workspace() -> Optional[str]:
        """Find dynamo workspace directory."""
        candidates = []

        # Check DYNAMO_HOME environment variable first
        dynamo_home = os.environ.get("DYNAMO_HOME")
        if dynamo_home:
            candidates.append(dynamo_home)

        # Then check common locations
        candidates.extend(
            [
                ".",  # Current directory
                os.path.expanduser("~/dynamo"),
                "/workspace",
            ]
        )

        for candidate in candidates:
            if DynamoInfo.is_dynamo_workspace(candidate):
                return os.path.abspath(candidate)
        return None

    @staticmethod
    def is_dynamo_workspace(path: str) -> bool:
        """Check if directory is a dynamo workspace."""
        if not os.path.exists(path):
            return False

        # Check for indicators of a dynamo workspace
        indicators = [
            "README.md",
            "components",
            "lib/bindings/python",
            "lib/runtime",
            "Cargo.toml",
        ]

        # Require at least 3 indicators to be confident
        found = 0
        for indicator in indicators:
            check_path = os.path.join(path, indicator)
            if os.path.exists(check_path):
                found += 1

        return found >= 3


def has_framework_errors(tree: NodeInfo) -> bool:
    """Check if there are framework component errors in the tree"""
    # Find the Dynamo node
    for child in tree.children:
        if child.label and "Dynamo" in child.label:
            # Find the Framework components node
            for dynamo_child in child.children:
                if dynamo_child.label and "Framework components" in dynamo_child.label:
                    # Use the has_errors() method to check the entire subtree
                    return dynamo_child.has_errors()
    return False


def show_pythonpath_recommendation():
    """Show PYTHONPATH recommendation for fixing import errors.

    Generates and displays the recommended PYTHONPATH based on discovered
    component source paths in the workspace.
    """
    paths = []

    # Try to find workspace directory
    workspace_dir = None
    candidates = [
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.environ.get("DYNAMO_HOME", ""),
        os.path.expanduser("~/dynamo"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "lib/bindings/python/src/dynamo")):
            workspace_dir = os.path.abspath(candidate)
            break

    if not workspace_dir:
        return

    # Collect all component source paths
    comp_path = os.path.join(workspace_dir, "components")
    if os.path.exists(comp_path):
        for item in os.listdir(comp_path):
            if item == "backends":
                continue  # Handle backends separately
            src_path = os.path.join(comp_path, item, "src")
            if os.path.exists(src_path):
                paths.append(src_path)

    # Collect all backend source paths
    backend_path = os.path.join(workspace_dir, "components", "backends")
    if os.path.exists(backend_path):
        for item in os.listdir(backend_path):
            src_path = os.path.join(backend_path, item, "src")
            if os.path.exists(src_path):
                paths.append(src_path)

    # Also add runtime path
    runtime_path = os.path.join(workspace_dir, "lib/bindings/python/src")
    if os.path.exists(runtime_path):
        paths.insert(0, runtime_path)  # Add at beginning for priority

    if paths:
        pythonpath = ":".join(paths)
        # Replace home directory with $HOME
        home = os.path.expanduser("~")
        if home in pythonpath:
            pythonpath = pythonpath.replace(home, "$HOME")

        print(f'\nSet PYTHONPATH for development:\nexport PYTHONPATH="{pythonpath}"\n')


def main():
    """Main function - collect and display system information"""
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Display system information for Dynamo project"
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Skip size calculations for faster output",
    )
    args = parser.parse_args()

    # Simply create a SystemInfo instance - it collects everything in its constructor
    tree = SystemInfo(fast_mode=args.fast)
    tree.print_tree()

    # Check if there are framework component errors and show PYTHONPATH recommendation
    if has_framework_errors(tree):
        show_pythonpath_recommendation()

    # Exit with non-zero status if there are any errors
    if tree.has_errors():
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
