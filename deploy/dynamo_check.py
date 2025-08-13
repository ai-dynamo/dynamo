#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
dynamo package checker, Python import tester, and usage guide.

Combines version checking, import testing, and usage examples into a single tool.
Features dynamic component discovery and comprehensive troubleshooting guidance.

Usage:
    dynamo_check.py                        # Run all checks
    dynamo_check.py --imports              # Only test imports
    dynamo_check.py --examples             # Only show examples
    dynamo_check.py --try-pythonpath      # Test imports with workspace paths
    dynamo_check.py --help                 # Show help

Outputs:
Dynamo Environment ($HOME/dynamo):
└─ Runtime components (ai-dynamo-runtime 0.4.0):
   ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime-0.4.0.dist-info (created: 2025-08-12 15:10:05 PDT)
   ├─ /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo_runtime.pth (modified: 2025-08-12 15:10:05 PDT)
      └─ Points to: $HOME/dynamo/lib/bindings/python/src
   ├─ ✅ dynamo._core        $HOME/dynamo/lib/bindings/python/src/dynamo/_core.cpython-312-x86_64-linux-gnu.so (modified: 2025-08-12 15:10:05 PDT)
   ├─ ✅ dynamo.nixl_connect $HOME/dynamo/lib/bindings/python/src/dynamo/nixl_connect/__init__.py
   ├─ ✅ dynamo.llm          $HOME/dynamo/lib/bindings/python/src/dynamo/llm/__init__.py
   └─ ✅ dynamo.runtime      $HOME/dynamo/lib/bindings/python/src/dynamo/runtime/__init__.py
└─ Framework components (ai-dynamo - Not installed):
   ├─ ❌ dynamo.frontend     No module named 'dynamo.frontend'
   ├─ ✅ dynamo.planner      $HOME/dynamo/components/planner/src/dynamo/planner/__init__.py
   ├─ ❌ dynamo.mocker       No module named 'dynamo.mocker'
   ├─ ❌ dynamo.trtllm       No module named 'dynamo.trtllm'
   ├─ ❌ dynamo.vllm         No module named 'dynamo.vllm'
   ├─ ❌ dynamo.sglang       No module named 'dynamo.sglang'
   └─ ❌ dynamo.llama_cpp    No module named 'dynamo.llama_cpp'
└─ Cargo home directory: $HOME/dynamo/.build/.cargo (CARGO_HOME is set)
└─ Cargo target directory: $HOME/dynamo/.build/target (CARGO_TARGET_DIR is set)
   ├─ Debug:   $HOME/dynamo/.build/target/debug (modified: 2025-08-12 15:10:02 PDT)
   └─ Binary:  $HOME/dynamo/.build/target/debug/libdynamo_llm_capi.so (modified: 2025-08-12 15:08:33 PDT)

Missing framework components. You can choose one of the following options:
1. For local development, set the PYTHONPATH environment variable:
   dynamo_check.py --try-pythonpath --imports
   export PYTHONPATH="$HOME/dynamo/components/router/src:$HOME/dynamo/components/metrics/src:$HOME/dynamo/components/frontend/src:$HOME/dynamo/components/planner/src:$HOME/dynamo/components/backends/mocker/src:$HOME/dynamo/components/backends/trtllm/src:$HOME/dynamo/components/backends/vllm/src:$HOME/dynamo/components/backends/sglang/src:$HOME/dynamo/components/backends/llama_cpp/src"
2. For a production-release (slower build time), build the packages with:
   dynamo_build.sh --release
"""

import argparse
import datetime
import importlib.metadata
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


class DynamoChecker:
    """Comprehensive dynamo package checker."""

    def __init__(self):
        self.workspace_dir = self._find_workspace()
        self.results = {}
        self._suppress_planner_warnings()

    def _suppress_planner_warnings(self):
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
            print(f"⚠️  Warning: Components directory not found: {comp_path}")

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
            print(
                f"⚠️  Warning: Backend components directory not found: {backend_path}"
            )

        return components

    def _is_dynamo_build_available(self) -> bool:
        """Check if dynamo_build.sh is available in the same directory as this script.

        Returns:
            True if dynamo_build.sh exists in the same directory as dynamo_check.py
        """
        script_dir = Path(__file__).parent
        dynamo_build_path = script_dir / "dynamo_build.sh"
        return dynamo_build_path.exists()

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
        # Replace all occurrences for colon-separated paths like PYTHONPATH
        return path.replace(home_dir, "$HOME")

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
        except FileNotFoundError:
            print(
                "⚠️  Warning: cargo command not found. Install Rust toolchain to see cargo target directory."
            )
            return None, None
        except subprocess.TimeoutExpired:
            print("⚠️  Warning: cargo command timed out")
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
                cwd=self.workspace_dir if self.workspace_dir else None,
            )

            if result.returncode == 0:
                # Parse JSON output to extract target_directory
                import json

                metadata = json.loads(result.stdout)
                target_directory = metadata.get("target_directory")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ):
            # cargo metadata failed or JSON parsing failed
            pass

        return target_directory, cargo_home

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

    def _setup_pythonpath(self):
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
                f'Below are the results if you export PYTHONPATH="{pythonpath_value}":'
            )
            print(f"   ({len(paths)} workspace component paths found)")
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
                    f"   {package_symbol} {display_path} (created: {package_created})"
                )
            else:
                print(f"   {package_symbol} {display_path}")

            # Show .pth files if they exist (editable installs) - at same level as package info
            pth_files = package_info.get("pth_files", [])
            for i, pth_file in enumerate(pth_files):
                is_last_pth = i == len(pth_files) - 1
                pth_symbol = "└─" if (is_last_pth and not has_components) else "├─"
                display_pth_path = self._replace_home_with_var(pth_file["path"])
                display_points_to = self._replace_home_with_var(pth_file["points_to"])
                print(
                    f"   {pth_symbol} {display_pth_path} (modified: {pth_file['modified']})"
                )
                print(f"      └─ Points to: {display_points_to}")
        # Don't print anything for "Not found" - just skip it

        # Test each component as subitems of the package
        for i, component in enumerate(components):
            # Determine tree symbol - last component gets └─, others get ├─, with proper indentation (deeper nesting)
            is_last = i == len(components) - 1
            tree_symbol = "   └─" if is_last else "   ├─"

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
                        # From workspace source
                        rel_path = os.path.relpath(module_path, self.workspace_dir)
                        if show_timestamp:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {rel_path}{timestamp_str}"
                            )
                        else:
                            print(
                                f"{tree_symbol} ✅ {component:<{max_width}} {rel_path}"
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

        # Print main environment header with workspace path
        if self.workspace_dir:
            workspace_path = os.path.abspath(self.workspace_dir)
            display_workspace = self._replace_home_with_var(workspace_path)
            print(f"Dynamo Environment ({display_workspace}):")
        else:
            print("Dynamo Environment (workspace not found):")

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

        # Test runtime components (as subitem of Dynamo Environment)
        runtime_results, _ = self._test_component_group(
            runtime_components,
            "ai-dynamo-runtime",
            "└─ Runtime components",
            max_width,
            site_packages,
            collect_failures=False,
            package_info=runtime_package_info,
        )
        results.update(runtime_results)

        # Test framework components (as subitem of Dynamo Environment)
        framework_results, framework_failures = self._test_component_group(
            framework_components,
            "ai-dynamo",
            "└─ Framework components",
            max_width,
            site_packages,
            collect_failures=True,
            package_info=framework_package_info,
        )
        results.update(framework_results)

        # Show Rust cargo information as subitem of Dynamo Environment
        cargo_target, cargo_home = self._get_cargo_info()
        if cargo_target or cargo_home:
            if cargo_home:
                cargo_home_env = os.environ.get("CARGO_HOME")
                display_cargo_home = self._replace_home_with_var(cargo_home)
                if cargo_home_env:
                    print(
                        f"└─ Cargo home directory: {display_cargo_home} (CARGO_HOME is set)"
                    )
                else:
                    print(f"└─ Cargo home directory: {display_cargo_home}")

            if cargo_target:
                cargo_target_env = os.environ.get("CARGO_TARGET_DIR")

                # Build the target directory message
                display_cargo_target = self._replace_home_with_var(cargo_target)
                if cargo_target_env:
                    target_msg = f"└─ Cargo target directory: {display_cargo_target} (CARGO_TARGET_DIR is set)"
                else:
                    target_msg = f"└─ Cargo target directory: {display_cargo_target}"

                print(target_msg)

                # Show debug and release directories on separate lines
                debug_dir = os.path.join(cargo_target, "debug")
                release_dir = os.path.join(cargo_target, "release")

                debug_exists = os.path.exists(debug_dir)
                release_exists = os.path.exists(release_dir)

                # Find *.so file
                so_file = self._find_so_file(cargo_target)
                has_so_file = so_file is not None

                if debug_exists:
                    # Use ├─ if there are more items below
                    symbol = "├─" if release_exists or has_so_file else "└─"
                    display_debug_dir = self._replace_home_with_var(debug_dir)
                    try:
                        debug_mtime = os.path.getmtime(debug_dir)
                        debug_time = self._format_timestamp_pdt(debug_mtime)
                        print(
                            f"   {symbol} Debug:   {display_debug_dir} (modified: {debug_time})"
                        )
                    except OSError:
                        print(
                            f"   {symbol} Debug:   {display_debug_dir} (unable to read timestamp)"
                        )

                if release_exists:
                    # Use ├─ if there's a *.so file below
                    symbol = "├─" if has_so_file else "└─"
                    display_release_dir = self._replace_home_with_var(release_dir)
                    try:
                        release_mtime = os.path.getmtime(release_dir)
                        release_time = self._format_timestamp_pdt(release_mtime)
                        print(
                            f"   {symbol} Release: {display_release_dir} (modified: {release_time})"
                        )
                    except OSError:
                        print(
                            f"   {symbol} Release: {display_release_dir} (unable to read timestamp)"
                        )

                # Show *.so file if found
                if has_so_file and so_file is not None:
                    display_so_file = self._replace_home_with_var(so_file)
                    try:
                        so_mtime = os.path.getmtime(so_file)
                        so_time = self._format_timestamp_pdt(so_mtime)
                        print(f"   └─ Binary:  {display_so_file} (modified: {so_time})")
                    except OSError:
                        print(
                            f"   └─ Binary:  {display_so_file} (unable to read timestamp)"
                        )

        # Show PYTHONPATH recommendation if any framework components failed (moved to end)
        if framework_failures and self.workspace_dir:
            pythonpath = self._get_pythonpath()
            if pythonpath:
                # Apply $HOME replacement to PYTHONPATH for consistency
                display_pythonpath = self._replace_home_with_var(pythonpath)
                print(
                    "\nMissing framework components. You can choose one of the following options:"
                )
                print(
                    "1. For local development, set the PYTHONPATH environment variable:"
                )
                print(
                    f'   dynamo_check.py --try-pythonpath --imports\n   export PYTHONPATH="{display_pythonpath}"'
                )
                not_found_suffix = (
                    ""
                    if self._is_dynamo_build_available()
                    else "  # (dynamo_build.sh not found)"
                )
                print(
                    "2. For a production-release (slower build time), build the packages with:"
                )
                print(f"   dynamo_build.sh --release{not_found_suffix}")

        return results

    # ====================================================================
    # USAGE EXAMPLES AND GUIDANCE
    # ====================================================================

    def show_usage_examples(self):
        """Show practical usage examples.

        Prints formatted examples of common dynamo operations including:
        - Starting frontend server
        - Starting vLLM backend
        - Making inference requests
        - Setting up development environment
        - Building packages

        Console output example:
            Usage Examples
            ========================================

            1. Start Frontend Server:
               python -m dynamo.frontend --http-port 8000

            2. Start vLLM Backend:
               python -m dynamo.vllm --model Qwen/Qwen2.5-0.5B
               ...
        """
        print(
            """
Usage Examples
========================================

1. Start Frontend Server:
   python -m dynamo.frontend --http-port 8000

2. Start vLLM Backend:
   python -m dynamo.vllm --model Qwen/Qwen2.5-0.5B

3. Send Inference Request:
   curl -X POST http://localhost:8000/v1/completions \\
        -H 'Content-Type: application/json' \\
        -d '{"model": "Qwen/Qwen2.5-0.5B", "prompt": "Hello", "max_tokens": 50}'

4. For local development: Set PYTHONPATH to use workspace sources without rebuilding:
   • Discover what PYTHONPATH to set: dynamo_check.py --try-pythonpath --imports"""
        )
        if self.workspace_dir:
            pythonpath = self._get_pythonpath()
            display_pythonpath = self._replace_home_with_var(pythonpath)
            print(
                f'   • Then set in your shell: export PYTHONPATH="{display_pythonpath}"'
            )
        else:
            print(
                '   • Then set in your shell: export PYTHONPATH="$HOME/dynamo/components/*/src"'
            )

        not_found_suffix = (
            "" if self._is_dynamo_build_available() else " (dynamo_build.sh not found)"
        )
        print(
            f"""
5. Build Packages:
   dynamo_build.sh --dev              # Development mode{not_found_suffix}
   dynamo_build.sh --release          # Production wheels{not_found_suffix}"""
        )

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
    # TROUBLESHOOTING AND SUMMARY
    # ====================================================================

    def show_troubleshooting(self):
        """Show troubleshooting guidance only if there were import failures."""
        # Check if any imports failed
        import_results = self.results.get("imports", {})
        failed_imports = [
            component
            for component, result in import_results.items()
            if result.startswith("❌")
        ]

        if not failed_imports:
            return  # No failures, skip troubleshooting section

        not_found_suffix = (
            ""
            if self._is_dynamo_build_available()
            else "  # (dynamo_build.sh not found)"
        )
        troubleshooting_msg = f"""
Troubleshooting
========================================

Found {len(failed_imports)} failed import(s). Common Issues:
1. ImportError for framework components:
   $ export PYTHONPATH=...

2. Package not found:
   $ dynamo_build.sh --release{not_found_suffix}

3. Check current status:
   $ dynamo_build.sh --check{not_found_suffix}"""

        print(troubleshooting_msg)

        if not self.workspace_dir:
            print(
                """
⚠️  Workspace not found!
   → Ensure you're running from a dynamo workspace
   → Expected locations: ~/dynamo, /workspace, /home/ubuntu/dynamo"""
            )

    def show_summary(self):
        """Show comprehensive summary."""
        print("\nSummary")
        print("=" * 40)

        # Import status
        import_results = self.results.get("imports", {})
        if import_results:
            total = len(import_results)
            passed = sum(1 for r in import_results.values() if r.startswith("✅"))
            if passed == total:
                print(f"✅ Import tests: {passed}/{total} passed")
            else:
                print(f"❌ Import tests: {passed}/{total} passed")

    # ====================================================================
    # MAIN ORCHESTRATION
    # ====================================================================

    def run_all(self):
        """Run comprehensive check with all functionality.

        Performs complete dynamo package validation including:
        - Component discovery and import testing
        - Usage examples and troubleshooting guidance
        - Summary of results

        Console output example:
            Dynamo Comprehensive Check
            ============================================================
            Runtime components (ai-dynamo-runtime 0.4.0):
               ✅ dynamo._core        /opt/dynamo/venv/lib/.../dynamo/_core.cpython-312-x86_64-linux-gnu.so
               ✅ dynamo.llm          /opt/dynamo/venv/lib/.../dynamo/llm/__init__.py

            Framework components (ai-dynamo 0.4.0):
               ✅ dynamo.frontend     /opt/dynamo/venv/lib/.../dynamo/frontend/__init__.py

            Usage Examples
            ========================================
            1. Start Frontend Server:
               python -m dynamo.frontend --http-port 8000
               ...

            Summary
            ========================================
            ✅ Import tests: 5/5 passed
        """
        print("Dynamo Comprehensive Check")
        print("=" * 60)

        # Execute all checks (package versions now shown in import testing headers)
        self.results["imports"] = self.test_imports()

        # Check if there were any import failures
        import_results = self.results.get("imports", {})
        has_failures = any(result.startswith("❌") for result in import_results.values())

        # Provide guidance
        if not has_failures:
            self.show_usage_examples()
        self.show_troubleshooting()
        self.show_summary()


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive dynamo package checker")
    parser.add_argument("--imports", action="store_true", help="Only test imports")
    parser.add_argument("--examples", action="store_true", help="Only show examples")
    parser.add_argument(
        "--try-pythonpath",
        action="store_true",
        help="Test imports with workspace component source directories in sys.path",
    )

    args = parser.parse_args()
    checker = DynamoChecker()

    # Set up sys.path if requested
    if args.try_pythonpath:
        checker._setup_pythonpath()

    if args.imports:
        checker.test_imports()
    elif args.examples:
        checker.show_usage_examples()
    else:
        checker.run_all()


if __name__ == "__main__":
    main()
