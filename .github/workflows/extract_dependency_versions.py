#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Extract all dependency versions from Dockerfiles and requirements files.
Generates a CSV file with all dependencies across trtllm, vllm, sglang, and operator components.

Usage:
    python scripts/extract_dependency_versions.py [--output OUTPUT_PATH]

Output:
    dependency_versions.csv (or specified output path)
"""

import argparse
import csv
import glob as glob_module
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class DependencyExtractor:
    def __init__(self, repo_root: Path, github_repo: str = "ai-dynamo/dynamo", github_branch: str = "main", config_path: Optional[Path] = None, previous_latest_csv: Optional[Path] = None, previous_release_csv: Optional[Path] = None):
        self.repo_root = repo_root
        self.dependencies: List[Dict[str, str]] = []
        self.github_repo = github_repo
        self.github_branch = github_branch
        self.baseline_count = 251  # Baseline dependency count for warnings
        
        # Error tracking
        self.missing_files: List[Dict[str, str]] = []
        self.processed_files: Set[str] = set()
        self.failed_files: List[Dict[str, str]] = []
        self.warnings: List[str] = []
        
        # Previous dependencies for comparison (latest nightly and release)
        self.previous_latest_dependencies: Dict[str, Dict[str, str]] = {}
        self.previous_release_dependencies: Dict[str, Dict[str, str]] = {}
        
        if previous_latest_csv:
            self.load_previous_csv(previous_latest_csv, "latest")
        if previous_release_csv:
            self.load_previous_csv(previous_release_csv, "release")
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Load critical dependencies list
        self.critical_dependencies = self._load_critical_dependencies()

    def _load_critical_dependencies(self) -> List[Dict[str, str]]:
        """Load critical dependencies list from configuration."""
        critical_deps = self.config.get('critical_dependencies', [])
        if not critical_deps:
            # Default critical dependencies if not in config
            return [
                {'name': 'CUDA', 'reason': 'Core compute platform'},
                {'name': 'PyTorch', 'reason': 'Primary ML framework'},
                {'name': 'Python', 'reason': 'Runtime language'},
                {'name': 'Kubernetes', 'reason': 'Orchestration platform'},
            ]
        return critical_deps
    
    def _get_package_source_url(self, dep_name: str, category: str, version: str, source_file: str) -> str:
        """Generate source URL for package/dependency based on type and name."""
        dep_lower = dep_name.lower()
        
        # Docker images from NVIDIA NGC Catalog
        if category == "Base Image" or category == "Docker Compose Service":
            if "nvcr.io" in source_file or "nvidia" in dep_lower:
                # Extract image name for NGC
                image_slug = dep_name.split('/')[-1].lower()
                return f"https://catalog.ngc.nvidia.com/orgs/nvidia/containers/{image_slug}"
            elif "/" in dep_name:
                # Docker Hub
                return f"https://hub.docker.com/r/{dep_name}"
        
        # Helm Charts
        if "Helm Chart" in category:
            chart_slug = dep_name.lower().replace(' ', '-')
            return f"https://artifacthub.io/packages/search?ts_query_web={chart_slug}"
        
        # Python packages
        if "Python" in category:
            # Remove version constraints and extras
            pkg_name = dep_name.split('[')[0].strip().lower()
            pkg_name = pkg_name.replace(' ', '-')
            return f"https://pypi.org/project/{pkg_name}/"
        
        # Go modules
        if category == "Go Module":
            return f"https://pkg.go.dev/{dep_name}"
        
        # Rust crates
        if category == "Rust Crate":
            return f"https://crates.io/crates/{dep_name}"
        
        # Git dependencies already have repo URLs - extract repo URL
        if "Git" in category and "github.com" in source_file:
            # Try to extract from notes or return GitHub search
            return f"https://github.com/search?q={dep_name}&type=repositories"
        
        # Framework/System packages
        if dep_name.lower() in ["rust", "python", "go", "cmake"]:
            if "rust" in dep_lower:
                return "https://www.rust-lang.org/tools/install"
            elif "python" in dep_lower:
                return "https://www.python.org/downloads/"
            elif "go" in dep_lower:
                return "https://go.dev/dl/"
            elif "cmake" in dep_lower:
                return "https://cmake.org/download/"
        
        # CUDA
        if "cuda" in dep_lower:
            return "https://developer.nvidia.com/cuda-downloads"
        
        # Default: return N/A
        return "N/A"
    
    def _is_nvidia_product(self, dep_name: str, category: str, source_file: str, notes: str) -> bool:
        """Determine if a dependency is an NVIDIA product."""
        # Combine all text for checking
        all_text = f"{dep_name} {category} {source_file} {notes}".lower()
        
        # Direct NVIDIA indicators
        nvidia_indicators = [
            "nvidia", "nvcr.io", "cuda", "tensorrt", "triton",
            "nccl", "nvshmem", "dcgm", "cutlass", "cudf",
            "rapids", "dali", "tao", "nvtabular", "merlin",
            "trt", "nemo"
        ]
        
        for indicator in nvidia_indicators:
            if indicator in all_text:
                return True
        
        # NGC catalog images
        if "nvcr.io" in source_file:
            return True
        
        # Base images from NVIDIA
        if category == "Base Image" and ("pytorch" in dep_name.lower() or "cuda" in dep_name.lower()):
            return True
        
        return False
    
    def _is_critical_dependency(self, dependency_name: str) -> tuple[bool, str]:
        """
        Check if a dependency is marked as critical.
        Returns (is_critical, reason).
        Uses case-insensitive partial matching.
        """
        dep_lower = dependency_name.lower()
        
        for critical in self.critical_dependencies:
            critical_name = critical.get('name', '').lower()
            if not critical_name:
                continue
            
            # Check for exact match or partial match (critical name in dependency name)
            if critical_name == dep_lower or critical_name in dep_lower:
                reason = critical.get('reason', 'Critical dependency')
                return (True, reason)
        
        return (False, '')
    
    def load_previous_csv(self, csv_path: Path, csv_type: str = "latest") -> None:
        """Load previous CSV for comparison.
        
        Args:
            csv_path: Path to the CSV file
            csv_type: Either "latest" (for nightly) or "release" (for release snapshot)
        """
        if not csv_path.exists():
            self.warnings.append(f"Previous {csv_type} CSV not found: {csv_path}")
            return
        
        # Select the appropriate storage dict
        target_dict = self.previous_latest_dependencies if csv_type == "latest" else self.previous_release_dependencies
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Create unique key for each dependency
                    key = f"{row.get('Component', '')}:{row.get('Category', '')}:{row.get('Dependency Name', '')}"
                    target_dict[key] = row
            
            # Use the count from previous CSV as the baseline for warnings
            # (only if this is the latest CSV, not release)
            if csv_type == "latest" and len(target_dict) > 0:
                self.baseline_count = len(target_dict)
                print(f"Loaded {len(target_dict)} dependencies from previous {csv_type} CSV: {csv_path.name}")
                print(f"Set baseline count to {self.baseline_count} (from previous extraction)")
            else:
                print(f"Loaded {len(target_dict)} dependencies from previous {csv_type} CSV: {csv_path.name}")
        except Exception as e:
            self.warnings.append(f"Error loading previous {csv_type} CSV: {e}")
    
    def load_config(self, config_path: Optional[Path] = None) -> dict:
        """Load configuration from YAML or JSON file."""
        if config_path is None:
            # Default to extract_dependency_versions_config.yaml in same directory as script
            script_dir = Path(__file__).parent
            config_path = script_dir / "extract_dependency_versions_config.yaml"
        
        if not config_path.exists():
            self.warnings.append(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(config_path) as f:
                if HAS_YAML and (config_path.suffix in ['.yaml', '.yml']):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Update settings from config
            if 'github' in config:
                self.github_repo = config['github'].get('repo', self.github_repo)
                self.github_branch = config['github'].get('branch', self.github_branch)
            
            if 'baseline' in config:
                self.baseline_count = config['baseline'].get('dependency_count', self.baseline_count)
            
            return config
        except Exception as e:
            self.warnings.append(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration if config file is not available."""
        return {
            'components': {
                'trtllm': {
                    'dockerfiles': ['container/Dockerfile.trtllm'],
                    'scripts': [],
                    'required': True
                },
                'vllm': {
                    'dockerfiles': ['container/Dockerfile.vllm'],
                    'scripts': ['container/deps/vllm/install_vllm.sh'],
                    'required': True
                },
                'sglang': {
                    'dockerfiles': ['container/Dockerfile.sglang'],
                    'scripts': [],
                    'required': True
                },
                'operator': {
                    'dockerfiles': ['deploy/cloud/operator/Dockerfile'],
                    'go_modules': ['deploy/cloud/operator/go.mod'],
                    'required': True
                },
                'shared': {
                    'dockerfiles': ['container/Dockerfile'],
                    'requirements': [{'pattern': 'container/deps/requirements*.txt', 'exclude': []}],
                    'pyproject': ['pyproject.toml', 'benchmarks/pyproject.toml'],
                    'required': True
                }
            }
        }
    
    def discover_files(self, patterns: List[str]) -> List[Path]:
        """Find files matching patterns with fallback locations."""
        found_files = []
        
        for pattern in patterns:
            # Try direct path first
            file_path = self.repo_root / pattern
            if file_path.exists() and file_path.is_file():
                found_files.append(file_path)
                continue
            
            # Try glob pattern
            glob_results = list(self.repo_root.glob(pattern))
            if glob_results:
                found_files.extend([p for p in glob_results if p.is_file()])
        
        return found_files
    
    def discover_requirements_files(self, req_config: List) -> List[Path]:
        """Discover requirements files using patterns and exclusions."""
        found_files = []
        
        for item in req_config:
            if isinstance(item, dict):
                pattern = item.get('pattern', '')
                exclude = item.get('exclude', [])
            else:
                pattern = item
                exclude = []
            
            # Find files matching pattern
            matches = list(self.repo_root.glob(pattern))
            
            # Filter out exclusions
            for match in matches:
                if match.is_file():
                    excluded = False
                    for exc_pattern in exclude:
                        if match.match(exc_pattern):
                            excluded = True
                            break
                    if not excluded:
                        found_files.append(match)
        
        return found_files
    
    def validate_critical_files(self, strict_mode: bool = False) -> bool:
        """Validate that critical files exist."""
        all_valid = True
        
        if 'components' not in self.config:
            return True
        
        for component_name, component_config in self.config['components'].items():
            is_required = component_config.get('required', False)
            
            # Check dockerfiles
            dockerfiles = component_config.get('dockerfiles', [])
            if dockerfiles:
                found = self.discover_files(dockerfiles)
                if not found and is_required:
                    self.missing_files.append({
                        'component': component_name,
                        'type': 'dockerfile',
                        'patterns': dockerfiles,
                        'required': is_required
                    })
                    if strict_mode:
                        all_valid = False
        
        return all_valid

    def _make_github_url(self, file_path: str, line_number: str) -> str:
        """Generate GitHub URL for a specific file and line number."""
        if file_path == "N/A" or line_number == "N/A":
            return "N/A"
        
        # Clean the file path
        file_path = file_path.replace("\\", "/")
        
        # Create GitHub URL
        url = f"https://github.com/{self.github_repo}/blob/{self.github_branch}/{file_path}"
        
        # Add line number if available
        if line_number and line_number.isdigit():
            url += f"#L{line_number}"
        
        return url
    
    def _format_dependency_name(self, name: str, category: str, version: str) -> str:
        """Format dependency name to be human-readable and well-formatted."""
        # Handle URLs and Git repositories
        if 'git+' in name or name.startswith('http://') or name.startswith('https://'):
            # Extract repository name from URL
            parts = name.rstrip('/').split('/')
            if len(parts) >= 2:
                repo_name = parts[-1].replace('.git', '')
                # Convert kebab-case or snake_case to Title Case
                formatted = ' '.join(word.capitalize() for word in re.split(r'[-_]', repo_name))
                return self._strip_version_suffixes(formatted)
            return name
        
        # Handle package names with extras (e.g., "package[extra]")
        if '[' in name and ']' in name:
            base_name = name.split('[')[0]
            extras = name[name.find('['):name.find(']')+1]
            formatted_base = self._format_package_name(base_name, category)
            return f"{self._strip_version_suffixes(formatted_base)} {extras}"
        
        # Handle Go modules
        if category == "Go Module":
            # Extract the last meaningful part of the module path
            parts = name.split('/')
            if len(parts) > 1:
                # Get the package name (last part)
                pkg_name = parts[-1]
                # If it's a versioned path, use the second-to-last
                if pkg_name.startswith('v') and pkg_name[1:].replace('.', '').isdigit():
                    pkg_name = parts[-2] if len(parts) > 2 else pkg_name
                return self._strip_version_suffixes(self._format_package_name(pkg_name, category))
        
        # Handle Docker base images
        if category == "Base Image":
            # Format: "nvcr.io/nvidia/pytorch" -> "NVIDIA PyTorch"
            if '/' in name and 'nvidia' in name.lower():
                parts = name.split('/')
                image_name = parts[-1]
                return f"NVIDIA {self._strip_version_suffixes(self._format_package_name(image_name, category))}"
            elif '/' in name:
                # Generic format: use last part
                parts = name.split('/')
                return self._strip_version_suffixes(self._format_package_name(parts[-1], category))
        
        # Handle ARG/ENV variable names that are already formatted (e.g., "Base Image Tag")
        if ' ' in name and name[0].isupper():
            return self._strip_version_suffixes(name)
        
        # Default: format as a package name
        return self._strip_version_suffixes(self._format_package_name(name, category))
    
    def _strip_version_suffixes(self, name: str) -> str:
        """Remove common version-related suffixes from dependency names."""
        # Common suffixes that don't add value (version info is in separate column)
        suffixes = [' Ver', ' Version', ' Ref', ' Tag']
        
        for suffix in suffixes:
            if name.endswith(suffix):
                return name[:-len(suffix)].strip()
        
        return name
    
    def _format_notes(self, notes: str, category: str, source_file: str) -> str:
        """Format notes to be more user-friendly and concise."""
        if not notes:
            return ""
        
        # Handle "ARG: VARIABLE_NAME" format
        if notes.startswith("ARG: "):
            return f"Dockerfile build argument"
        
        # Handle "From install script: VARIABLE_NAME" format
        if notes.startswith("From install script:"):
            return "From installation script"
        
        # Handle "ENV: VARIABLE_NAME" format
        if notes.startswith("ENV: "):
            return "Dockerfile environment variable"
        
        # Handle Git dependency notes
        if notes.startswith("Git dependency:"):
            return f"Git repository dependency"
        
        # Handle "Git-based pip install from ..."
        if notes.startswith("Git-based pip install from"):
            org_repo = notes.replace("Git-based pip install from ", "")
            return f"Installed from Git ({org_repo})"
        
        # Helm dependencies
        if "Helm dependency from" in notes:
            # Extract just the source type
            if "oci://" in notes:
                return "Helm chart from OCI registry"
            elif "file://" in notes:
                return "Local Helm chart"
            elif "https://" in notes:
                # Extract domain
                import re
                match = re.search(r'https://([^/]+)', notes)
                if match:
                    domain = match.group(1)
                    return f"Helm chart from {domain}"
                return "Helm chart from registry"
            else:
                return "Helm chart dependency"
        
        # Service-related notes
        if notes.startswith("Service:"):
            return f"Docker Compose service"
        
        # Keep certain notes as-is if they're already readable
        readable_patterns = [
            "Build/Runtime base image",
            "Rust toolchain version",
            "Go version",
            "Go toolchain version",
            "Project version",
            "Helm chart version",
            "Direct dependency",
            "Indirect dependency",
            "Python package",
            "From pyproject.toml",
            "From requirements.txt",
        ]
        
        for pattern in readable_patterns:
            if pattern in notes:
                return notes
        
        # Default: return as-is but clean up
        return notes.strip()
    
    def _format_package_name(self, name: str, category: str) -> str:
        """Format a package/module name to be human-readable."""
        # Handle special cases and well-known packages
        special_cases = {
            'fastapi': 'FastAPI',
            'numpy': 'NumPy',
            'pytorch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'kubernetes': 'Kubernetes',
            'pydantic': 'Pydantic',
            'openai': 'OpenAI',
            'httpx': 'HTTPX',
            'uvicorn': 'Uvicorn',
            'pytest': 'pytest',
            'mypy': 'mypy',
            'pyright': 'Pyright',
            'golang': 'Go',
            'grpc': 'gRPC',
            'protobuf': 'Protocol Buffers',
            'yaml': 'YAML',
            'toml': 'TOML',
            'json': 'JSON',
            'jwt': 'JWT',
            'oauth': 'OAuth',
            'redis': 'Redis',
            'postgres': 'PostgreSQL',
            'postgresql': 'PostgreSQL',
            'mysql': 'MySQL',
            'mongodb': 'MongoDB',
            'etcd': 'etcd',
            'nats': 'NATS',
            'cuda': 'CUDA',
            'nvidia': 'NVIDIA',
            'asyncio': 'asyncio',
            'aiohttp': 'aiohttp',
            'sqlalchemy': 'SQLAlchemy',
            'alembic': 'Alembic',
            'celery': 'Celery',
            'flask': 'Flask',
            'django': 'Django',
            'jinja2': 'Jinja2',
        }
        
        name_lower = name.lower()
        if name_lower in special_cases:
            return special_cases[name_lower]
        
        # Check for partial matches in the name
        for key, value in special_cases.items():
            if key in name_lower:
                return name.replace(key, value).replace(key.upper(), value).replace(key.capitalize(), value)
        
        # Handle hyphen-separated or underscore-separated names
        if '-' in name or '_' in name:
            words = re.split(r'[-_]', name)
            formatted_words = []
            for word in words:
                # Keep acronyms uppercase (short all-caps words)
                if word.isupper() and len(word) <= 4:
                    formatted_words.append(word)
                # Make 1-2 letter words uppercase (likely acronyms like "io", "db")
                elif len(word) <= 2:
                    formatted_words.append(word.upper())
                else:
                    formatted_words.append(word.capitalize())
            return ' '.join(formatted_words)
        
        # Handle camelCase by inserting spaces
        if any(c.isupper() for c in name[1:]) and not name.isupper():
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
            return spaced
        
        # Default: capitalize first letter
        return name.capitalize() if name else name

    def add_dependency(self, component: str, category: str, name: str, 
                      version: str, source_file: str, line_ref: str, notes: str = ""):
        """Add a dependency entry to the list."""
        github_url = self._make_github_url(source_file, line_ref)
        
        # Format the dependency name for human readability
        formatted_name = self._format_dependency_name(name, category, version)
        
        # Check if this is a critical dependency (check both original and formatted names)
        is_critical_orig, _ = self._is_critical_dependency(name)
        is_critical_formatted, _ = self._is_critical_dependency(formatted_name)
        is_critical = is_critical_orig or is_critical_formatted
        
        # Determine if this is new or changed (use FORMATTED name for key since CSV stores formatted names)
        key = f"{component}:{category}:{formatted_name}"
        
        # Compare with latest nightly
        diff_from_latest = ""
        if self.previous_latest_dependencies:
            if key in self.previous_latest_dependencies:
                prev_version = self.previous_latest_dependencies[key].get('Version', '')
                if prev_version != version:
                    diff_from_latest = f"{prev_version} → {version}"
                else:
                    diff_from_latest = "Unchanged"
            else:
                diff_from_latest = "New"
        else:
            diff_from_latest = "N/A"
        
        # Compare with latest release
        diff_from_release = ""
        if self.previous_release_dependencies:
            if key in self.previous_release_dependencies:
                prev_version = self.previous_release_dependencies[key].get('Version', '')
                if prev_version != version:
                    diff_from_release = f"{prev_version} → {version}"
                else:
                    diff_from_release = "Unchanged"
            else:
                diff_from_release = "New"
        else:
            diff_from_release = "N/A"
        
        # Legacy status field (for backwards compatibility, based on latest)
        status = "New" if diff_from_latest == "New" else ("Changed" if "→" in diff_from_latest else "Unchanged")
        
        # Generate package source URL
        package_source_url = self._get_package_source_url(formatted_name, category, version, source_file)
        
        # Determine if this is an NVIDIA product
        is_nvidia = self._is_nvidia_product(formatted_name, category, source_file, notes)
        
        # Format notes to be more user-friendly
        formatted_notes = self._format_notes(notes, category, source_file)
        
        self.dependencies.append({
            "Component": component,
            "Category": category,
            "Dependency Name": formatted_name,
            "Version": version,
            "Source File": source_file,
            "GitHub URL": github_url,
            "Package Source URL": package_source_url,
            "Status": status,
            "Diff from Latest": diff_from_latest,
            "Diff from Release": diff_from_release,
            "Critical": "Yes" if is_critical else "No",
            "NVIDIA Product": "Yes" if is_nvidia else "No",
            "Notes": formatted_notes
        })

    def extract_dockerfile_args(self, dockerfile_path: Path, component: str) -> None:
        """Extract ARG and ENV declarations from Dockerfile."""
        if not dockerfile_path.exists():
            self.failed_files.append({
                'file': str(dockerfile_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(dockerfile_path.relative_to(self.repo_root)))
            
            with open(dockerfile_path) as f:
                lines = f.readlines()
            
            # Build a dictionary of ARG values for variable substitution
            arg_values = {}
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Collect ARG values
                if line.startswith("ARG ") and "=" in line:
                    arg_line = line[4:].strip()
                    if "=" in arg_line:
                        key, value = arg_line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        arg_values[key] = value
                        
                        # Extract version-related ARGs
                        version_keywords = ["VERSION", "REF", "TAG", "_VER"]
                        if any(kw in key for kw in version_keywords):
                            category = "System" if key.startswith(("NATS", "ETCD", "NIXL", "UCX", "RUST")) else "Framework"
                            self.add_dependency(
                                component, category, key.replace("_", " ").title(), value,
                                str(dockerfile_path.relative_to(self.repo_root)),
                                str(i), f"ARG: {key}"
                            )
                
                # Extract base images with variable resolution
                if line.startswith("FROM ") and "AS" in line:
                    parts = line.split()
                    image = parts[1]
                    if ":" in image:
                        img_name, tag = image.rsplit(":", 1)
                        
                        # Resolve variables in image name and tag
                        img_name = self._resolve_dockerfile_vars(img_name, arg_values)
                        tag = self._resolve_dockerfile_vars(tag, arg_values)
                        
                        # Only add if not just variable names
                        if not (img_name.startswith('${') or tag.startswith('${')):
                            self.add_dependency(
                                component, "Base Image", img_name, tag,
                                str(dockerfile_path.relative_to(self.repo_root)),
                                str(i), "Build/Runtime base image"
                            )
        except Exception as e:
            self.failed_files.append({
                'file': str(dockerfile_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def _resolve_dockerfile_vars(self, text: str, arg_values: dict) -> str:
        """Resolve Dockerfile variables like ${VAR} or $VAR to their values."""
        if not text or '$' not in text:
            return text
        
        # Handle ${VAR} syntax
        import re
        def replace_var(match):
            var_name = match.group(1)
            return arg_values.get(var_name, match.group(0))
        
        text = re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replace_var, text)
        
        # Handle $VAR syntax (without braces)
        def replace_simple_var(match):
            var_name = match.group(1)
            return arg_values.get(var_name, match.group(0))
        
        text = re.sub(r'\$([A-Z_][A-Z0-9_]*)', replace_simple_var, text)
        
        return text

    def extract_requirements_file(self, req_file: Path, component: str, category: str) -> None:
        """Extract dependencies from requirements.txt style files."""
        if not req_file.exists():
            self.failed_files.append({
                'file': str(req_file.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(req_file.relative_to(self.repo_root)))
            
            with open(req_file) as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                
                # Skip lines with just flags/options
                if line.startswith(('-', '--')):
                    continue
                
                # Enhanced parsing for multiple version specifier formats
                # Supports: ==, >=, <=, >, <, ~=, !=, @, [extras]
                # Examples: package==1.0, package>=1.0,<2.0, package[extra]==1.0, package @ url
                match = re.match(r'^([a-zA-Z0-9_\-]+)(\[[\w,\-]+\])?([=<>!~@]+)?(.*)$', line)
                if match:
                    package_name = match.group(1)
                    extras = match.group(2) or ""
                    operator = match.group(3) or ""
                    version_part = match.group(4).strip() if match.group(4) else ""
                    
                    # Build full package name with extras
                    full_package_name = package_name + extras if extras else package_name
                    
                    # Determine version
                    if operator and version_part:
                        # Handle special cases
                        if operator == '@':
                            # URL or git reference
                            if 'git+' in version_part or 'http' in version_part:
                                version = "from URL"
                            else:
                                version = f"@{version_part[:50]}"  # Truncate long URLs
                        else:
                            # Clean up version part (remove trailing commas, semicolons)
                            version_part = version_part.split(';')[0].strip()  # Remove markers
                            version = f"{operator}{version_part}"
                    else:
                        version = "unspecified"
                    
                    self.add_dependency(
                        component, category, full_package_name, version,
                        str(req_file.relative_to(self.repo_root)),
                        str(i), f"Python package from {req_file.name}"
                    )
        except Exception as e:
            self.failed_files.append({
                'file': str(req_file.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })

    def extract_pyproject_toml(self, pyproject_path: Path, component: str) -> None:
        """Extract dependencies from pyproject.toml."""
        if not pyproject_path.exists():
            self.failed_files.append({
                'file': str(pyproject_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(pyproject_path.relative_to(self.repo_root)))
            
            with open(pyproject_path) as f:
                content = f.read()
                lines = content.split('\n')
            
            in_dependencies = False
            in_optional = False
            current_optional = None
            in_tool_section = False  # Track if we're in a [tool.*] section
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Track if we enter a [tool.*] section (like [tool.pytest.ini_options])
                if stripped.startswith('[tool.'):
                    in_tool_section = True
                    in_dependencies = False
                    in_optional = False
                    current_optional = None
                    continue
                # Exit tool section when we hit another top-level section
                elif stripped.startswith('[') and not stripped.startswith('[tool.'):
                    in_tool_section = False
                
                # Skip everything in tool sections
                if in_tool_section:
                    continue
                
                # Extract project version
                if stripped.startswith('version = '):
                    version = stripped.split('=', 1)[1].strip().strip('"')
                    # Get project name from earlier in file
                    for j in range(max(0, i-20), i):
                        if lines[j].strip().startswith('name = '):
                            name = lines[j].strip().split('=', 1)[1].strip().strip('"')
                            self.add_dependency(
                                component, "Project", name, version,
                                str(pyproject_path.relative_to(self.repo_root)),
                                str(i), "Project version"
                            )
                            break
                
                # Track sections
                if stripped == 'dependencies = [':
                    in_dependencies = True
                    continue
                elif stripped.startswith('[project.optional-dependencies]'):
                    in_optional = True
                    continue
                elif stripped.startswith('[') and in_dependencies:
                    in_dependencies = False
                elif stripped == ']' and in_dependencies:
                    in_dependencies = False
                
                # Extract optional dependency group names
                if in_optional and '= [' in stripped:
                    current_optional = stripped.split('=')[0].strip()
                elif stripped == ']' and in_optional and current_optional:
                    current_optional = None
                
                # Extract dependency specs - enhanced version detection
                if (in_dependencies or current_optional) and stripped.startswith('"'):
                    # Parse "package==version" or "package>=version"
                    dep_spec = stripped.strip('",')
                    # Enhanced regex to handle extras, multiple operators, URLs
                    match = re.match(r'^([a-zA-Z0-9_\-]+)(\[[\w,\-]+\])?([=<>!~@]+)?(.*)$', dep_spec)
                    if match:
                        package_name = match.group(1)
                        extras = match.group(2) or ""
                        operator = match.group(3) or ""
                        version_part = match.group(4) if match.group(4) else ""
                        
                        # Build full package name with extras
                        full_package_name = package_name + extras if extras else package_name
                        
                        # Determine version with enhanced handling
                        if operator and version_part:
                            if operator == '@':
                                version = "from URL" if ('git+' in version_part or 'http' in version_part) else f"@{version_part[:30]}"
                            else:
                                version = f"{operator}{version_part}"
                        else:
                            version = "unspecified"
                        
                        category = f"Python Package ({current_optional})" if current_optional else "Python Package"
                        self.add_dependency(
                            component, category, full_package_name, version,
                            str(pyproject_path.relative_to(self.repo_root)),
                            str(i), "From pyproject.toml"
                        )
        except Exception as e:
            self.failed_files.append({
                'file': str(pyproject_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })

    def extract_docker_compose(self, compose_path: Path, component: str) -> None:
        """Extract service versions from docker-compose.yml."""
        if not compose_path.exists():
            self.failed_files.append({
                'file': str(compose_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(compose_path.relative_to(self.repo_root)))
            
            with open(compose_path) as f:
                if HAS_YAML:
                    compose_data = yaml.safe_load(f)
                else:
                    # Skip if no YAML support
                    self.warnings.append(f"Skipping {compose_path}: PyYAML not available")
                    return
            
            services = compose_data.get('services', {})
            for service_name, service_config in services.items():
                if isinstance(service_config, dict) and 'image' in service_config:
                    image = service_config['image']
                    if ':' in image:
                        image_name, tag = image.rsplit(':', 1)
                        self.add_dependency(
                            component, "Docker Compose Service", image_name, tag,
                            str(compose_path.relative_to(self.repo_root)),
                            "N/A", f"Service: {service_name}"
                        )
        except Exception as e:
            self.failed_files.append({
                'file': str(compose_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def extract_helm_chart(self, chart_path: Path, component: str) -> None:
        """Extract dependency versions from Helm Chart.yaml."""
        if not chart_path.exists():
            self.failed_files.append({
                'file': str(chart_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(chart_path.relative_to(self.repo_root)))
            
            with open(chart_path) as f:
                if HAS_YAML:
                    chart_data = yaml.safe_load(f)
                else:
                    # Skip if no YAML support
                    self.warnings.append(f"Skipping {chart_path}: PyYAML not available")
                    return
            
            # Extract chart version
            if 'version' in chart_data:
                chart_name = chart_data.get('name', 'Unknown Chart')
                self.add_dependency(
                    component, "Helm Chart", chart_name, chart_data['version'],
                    str(chart_path.relative_to(self.repo_root)),
                    "N/A", "Helm chart version"
                )
            
            # Extract dependencies
            dependencies = chart_data.get('dependencies', [])
            for dep in dependencies:
                if isinstance(dep, dict):
                    dep_name = dep.get('name', 'Unknown')
                    dep_version = dep.get('version', 'unspecified')
                    repository = dep.get('repository', '')
                    notes = f"Helm dependency"
                    if repository:
                        notes += f" from {repository}"
                    
                    self.add_dependency(
                        component, "Helm Chart Dependency", dep_name, dep_version,
                        str(chart_path.relative_to(self.repo_root)),
                        "N/A", notes
                    )
        except Exception as e:
            self.failed_files.append({
                'file': str(chart_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def extract_rust_toolchain(self, toolchain_path: Path, component: str) -> None:
        """Extract Rust version from rust-toolchain.toml."""
        if not toolchain_path.exists():
            self.failed_files.append({
                'file': str(toolchain_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(toolchain_path.relative_to(self.repo_root)))
            
            with open(toolchain_path) as f:
                content = f.read()
            
            # Parse TOML manually (simple case)
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('channel'):
                    # channel = "1.90.0" or channel = '1.90.0'
                    match = re.search(r'channel\s*=\s*["\']([^"\']+)["\']', line)
                    if match:
                        rust_version = match.group(1)
                        self.add_dependency(
                            component, "Language", "Rust", rust_version,
                            str(toolchain_path.relative_to(self.repo_root)),
                            "N/A", "Rust toolchain version"
                        )
                        break
        except Exception as e:
            self.failed_files.append({
                'file': str(toolchain_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def extract_cargo_toml_git_deps(self, cargo_path: Path, component: str) -> None:
        """Extract Git dependencies from Cargo.toml."""
        if not cargo_path.exists():
            self.failed_files.append({
                'file': str(cargo_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(cargo_path.relative_to(self.repo_root)))
            
            with open(cargo_path) as f:
                content = f.read()
            
            # Pattern to match: name = { git = "...", rev = "..." }
            # Example: modelexpress-client = { git = "https://github.com/ai-dynamo/modelexpress.git", rev = "a232220..." }
            git_dep_pattern = r'(\w+(?:-\w+)*)\s*=\s*\{[^}]*git\s*=\s*"([^"]+)"[^}]*rev\s*=\s*"([^"]+)"'
            
            for match in re.finditer(git_dep_pattern, content):
                dep_name = match.group(1)
                git_url = match.group(2)
                git_rev = match.group(3)
                
                # Extract repo name from URL
                repo_name = git_url.rstrip('/').split('/')[-1].replace('.git', '')
                
                # Get line number for GitHub URL
                line_num = content[:match.start()].count('\n') + 1
                
                self.add_dependency(
                    component, "Rust Git Dependency", repo_name, git_rev[:12],
                    str(cargo_path.relative_to(self.repo_root)),
                    str(line_num), f"Git dependency: {dep_name}"
                )
        except Exception as e:
            self.failed_files.append({
                'file': str(cargo_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def extract_k8s_recipe_yaml(self, yaml_path: Path, component: str) -> None:
        """Extract Git-based pip installs from K8s recipe YAML files."""
        if not yaml_path.exists():
            self.failed_files.append({
                'file': str(yaml_path.relative_to(self.repo_root)),
                'component': component,
                'reason': 'File not found'
            })
            return
        
        try:
            self.processed_files.add(str(yaml_path.relative_to(self.repo_root)))
            
            with open(yaml_path) as f:
                content = f.read()
            
            # Pattern to match: pip install git+https://github.com/...@COMMIT_SHA
            # Example: pip install git+https://github.com/ai-dynamo/aiperf.git@70af59489df24a601dba57604a7341966150b366
            git_pip_pattern = r'pip\s+install\s+git\+https://github\.com/([^/]+)/([^/@\s\.]+)(?:\.git)?@([a-f0-9]{40})'
            
            for match in re.finditer(git_pip_pattern, content):
                org_name = match.group(1)
                repo_name = match.group(2)  # Will not include .git due to [^/@\s\.]+
                commit_sha = match.group(3)
                
                # Get line number for reference
                line_num = content[:match.start()].count('\n') + 1
                
                self.add_dependency(
                    component, "Python Git Package", repo_name, commit_sha[:12],
                    str(yaml_path.relative_to(self.repo_root)),
                    str(line_num), f"Git-based pip install from {org_name}/{repo_name}"
                )
        except Exception as e:
            self.failed_files.append({
                'file': str(yaml_path.relative_to(self.repo_root)),
                'component': component,
                'reason': f'Extraction error: {str(e)}'
            })
    
    def extract_go_mod(self, go_mod_path: Path, component: str) -> None:
        """Extract Go module dependencies from go.mod."""
        if not go_mod_path.exists():
            print(f"Warning: {go_mod_path} not found")
            return
        
        with open(go_mod_path) as f:
            lines = f.readlines()
        
        in_require = False
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Extract Go version
            if stripped.startswith('go '):
                version = stripped.split()[1]
                self.add_dependency(
                    component, "Language", "go", version,
                    str(go_mod_path.relative_to(self.repo_root)),
                    str(i), "Go version"
                )
            
            # Extract toolchain
            if stripped.startswith('toolchain '):
                version = stripped.split()[1]
                self.add_dependency(
                    component, "Language", "go-toolchain", version,
                    str(go_mod_path.relative_to(self.repo_root)),
                    str(i), "Go toolchain version"
                )
            
            # Track require block
            if stripped.startswith('require ('):
                in_require = True
                continue
            elif stripped == ')' and in_require:
                in_require = False
                continue
            
            # Extract dependencies
            if in_require or (stripped.startswith('require ') and not '(' in stripped):
                # Handle single-line require
                if stripped.startswith('require '):
                    stripped = stripped[8:].strip()
                
                parts = stripped.split()
                if len(parts) >= 2:
                    module = parts[0]
                    version = parts[1]
                    
                    # Skip indirect dependencies for cleaner output (optional)
                    # if '// indirect' in line:
                    #     continue
                    
                    self.add_dependency(
                        component, "Go Module", module, version,
                        str(go_mod_path.relative_to(self.repo_root)),
                        str(i), "Direct dependency" if '// indirect' not in line else "Indirect dependency"
                    )

    def extract_install_script(self, script_path: Path, component: str) -> None:
        """Extract version information from installation scripts."""
        if not script_path.exists():
            print(f"Warning: {script_path} not found")
            return
        
        with open(script_path) as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Look for version assignments in bash scripts
            if '=' in line and any(keyword in line for keyword in ['VERSION', '_REF', '_VER']):
                # Extract bash variable assignments
                match = re.match(r'^\s*([A-Z_]+)="?([^"#\s]+)"?', line)
                if match:
                    var_name = match.group(1)
                    value = match.group(2)
                    
                    # Skip variables that are just defaults or empty
                    if value and value not in ['""', "''", '$2']:
                        self.add_dependency(
                            component, "Framework", var_name.replace("_", " ").title(), value,
                            str(script_path.relative_to(self.repo_root)),
                            str(i), f"From install script: {var_name}"
                        )

    def extract_all(self) -> None:
        """Extract all dependencies from all sources using configuration."""
        print("Extracting dependencies...")
        
        if 'components' not in self.config:
            print("Warning: No components defined in config. Using hardcoded paths.")
            self._extract_all_legacy()
            return
        
        # Process each component from config
        for component_name, component_config in self.config['components'].items():
            print(f"  - Processing {component_name}...")
            
            # Extract from Dockerfiles
            dockerfiles = component_config.get('dockerfiles', [])
            if dockerfiles:
                found_dockerfiles = self.discover_files(dockerfiles)
                if found_dockerfiles:
                    for dockerfile in found_dockerfiles:
                        self.extract_dockerfile_args(dockerfile, component_name)
                elif component_config.get('required', False):
                    self.warnings.append(f"No Dockerfiles found for {component_name}: {dockerfiles}")
            
            # Extract from installation scripts
            scripts = component_config.get('scripts', [])
            if scripts:
                found_scripts = self.discover_files(scripts)
                for script in found_scripts:
                    self.extract_install_script(script, component_name)
            
            # Extract from Go modules
            go_modules = component_config.get('go_modules', [])
            if go_modules:
                found_go_mods = self.discover_files(go_modules)
                for go_mod in found_go_mods:
                    self.extract_go_mod(go_mod, component_name)
            
            # Extract from requirements files
            requirements = component_config.get('requirements', [])
            if requirements:
                found_reqs = self.discover_requirements_files(requirements)
                for req_file in found_reqs:
                    # Determine category from filename
                    filename = req_file.name
                    if 'test' in filename:
                        category = "Python Package (Test)"
                    elif 'docs' in filename:
                        category = "Python Package (Docs)"
                    elif 'standard' in filename:
                        category = "Python Package (Standard)"
                    else:
                        category = "Python Package"
                    self.extract_requirements_file(req_file, component_name, category)
            
            # Extract from pyproject.toml files
            pyproject = component_config.get('pyproject', [])
            if pyproject:
                found_pyprojects = self.discover_files(pyproject)
                for pyproject_file in found_pyprojects:
                    self.extract_pyproject_toml(pyproject_file, component_name)
            
            # Extract from docker-compose.yml files
            docker_compose = component_config.get('docker_compose', [])
            if docker_compose:
                found_compose = self.discover_files(docker_compose)
                for compose_file in found_compose:
                    self.extract_docker_compose(compose_file, component_name)
            
            # Extract from Helm Chart.yaml files
            helm_charts = component_config.get('helm_charts', [])
            if helm_charts:
                found_charts = self.discover_files(helm_charts)
                for chart_file in found_charts:
                    self.extract_helm_chart(chart_file, component_name)
            
            # Extract from rust-toolchain.toml
            rust_toolchain = component_config.get('rust_toolchain', [])
            if rust_toolchain:
                found_toolchains = self.discover_files(rust_toolchain)
                for toolchain_file in found_toolchains:
                    self.extract_rust_toolchain(toolchain_file, component_name)
            
            # Extract from Cargo.toml Git dependencies
            cargo_tomls = component_config.get('cargo_toml', [])
            if cargo_tomls:
                found_cargo = self.discover_files(cargo_tomls)
                for cargo_file in found_cargo:
                    self.extract_cargo_toml_git_deps(cargo_file, component_name)
            
            # Extract from K8s recipe YAML files (pip install git+...)
            k8s_recipes = component_config.get('k8s_recipes', [])
            if k8s_recipes:
                found_recipes = self.discover_requirements_files(k8s_recipes)  # Use pattern-aware discovery
                for recipe_file in found_recipes:
                    self.extract_k8s_recipe_yaml(recipe_file, component_name)
        
        # Add note about transitive dependencies
        self.add_dependency(
            "shared", "Note", "transitive-dependencies", "N/A", "N/A", "N/A",
            "Transitive dependencies from vLLM, SGLang, and TensorRT-LLM are NOT captured in this CSV. "
            "These frameworks have their own dependency trees that would need to be extracted separately."
        )
        
        print(f"✓ Extracted {len(self.dependencies)} dependencies")
    
    def _extract_all_legacy(self) -> None:
        """Legacy extraction method (fallback when config unavailable)."""
        # TRT-LLM
        print("  - TRT-LLM Dockerfile...")
        self.extract_dockerfile_args(
            self.repo_root / "container/Dockerfile.trtllm", "trtllm"
        )
        
        # vLLM
        print("  - vLLM Dockerfile...")
        self.extract_dockerfile_args(
            self.repo_root / "container/Dockerfile.vllm", "vllm"
        )
        self.extract_install_script(
            self.repo_root / "container/deps/vllm/install_vllm.sh", "vllm"
        )
        
        # SGLang
        print("  - SGLang Dockerfile...")
        self.extract_dockerfile_args(
            self.repo_root / "container/Dockerfile.sglang", "sglang"
        )
        
        # Operator
        print("  - Operator Dockerfile...")
        self.extract_dockerfile_args(
            self.repo_root / "deploy/cloud/operator/Dockerfile", "operator"
        )
        self.extract_go_mod(
            self.repo_root / "deploy/cloud/operator/go.mod", "operator"
        )
        
        # Base Dockerfile (shared)
        print("  - Base Dockerfile...")
        self.extract_dockerfile_args(
            self.repo_root / "container/Dockerfile", "shared"
        )
        
        # Python requirements files
        print("  - Requirements files...")
        for req_file in ["requirements.txt", "requirements.test.txt", "requirements.docs.txt", "requirements.standard.txt"]:
            path = self.repo_root / "container/deps" / req_file
            if path.exists():
                category = "Python Package (Test)" if "test" in req_file else \
                          "Python Package (Docs)" if "docs" in req_file else \
                          "Python Package (Standard)" if "standard" in req_file else "Python Package"
                self.extract_requirements_file(path, "shared", category)
        
        # PyProject files
        print("  - PyProject files...")
        self.extract_pyproject_toml(self.repo_root / "pyproject.toml", "shared")
        self.extract_pyproject_toml(self.repo_root / "benchmarks/pyproject.toml", "shared")

    def write_csv(self, output_path: Path) -> None:
        """Write dependencies to CSV file."""
        print(f"Writing to {output_path}...")
        
        # Sort dependencies: First by Component, then Critical (Yes before No), then by name
        def sort_key(dep):
            component_order = {"trtllm": 0, "vllm": 1, "sglang": 2, "operator": 3, "shared": 4}
            component_rank = component_order.get(dep.get("Component", ""), 99)
            critical_rank = 0 if dep.get("Critical") == "Yes" else 1
            name = dep.get("Dependency Name", "")
            return (component_rank, critical_rank, name.lower())
        
        sorted_dependencies = sorted(self.dependencies, key=sort_key)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Component", "Category", "Dependency Name", "Version",
                "Source File", "GitHub URL", "Package Source URL",
                "Status", "Diff from Latest", "Diff from Release",
                "Critical", "NVIDIA Product", "Notes"
            ])
            writer.writeheader()
            writer.writerows(sorted_dependencies)
        
        # Print change summary if comparing with previous
        if self.previous_latest_dependencies or self.previous_release_dependencies:
            new_count = sum(1 for d in self.dependencies if d['Status'] == 'New')
            changed_count = sum(1 for d in self.dependencies if d['Status'] == 'Changed')
            unchanged_count = sum(1 for d in self.dependencies if d['Status'] == 'Unchanged')
            removed = self.get_removed_dependencies()
            
            print(f"✓ Written {len(self.dependencies)} dependencies to {output_path}")
            print(f"  Changes since previous version:")
            print(f"    New: {new_count}")
            print(f"    Changed: {changed_count}")
            print(f"    Removed: {len(removed)}")
            print(f"    Unchanged: {unchanged_count}")
            
            if removed:
                print(f"\n  Removed dependencies:")
                for dep in removed[:10]:  # Show first 10
                    critical_flag = " [CRITICAL]" if dep['Critical'] == 'Yes' else ""
                    print(f"    • {dep['Dependency Name']} (was: {dep['Version']}){critical_flag}")
                    print(f"      from {dep['Source File']}")
                if len(removed) > 10:
                    print(f"    ... and {len(removed) - 10} more")
        else:
            print(f"✓ Written {len(self.dependencies)} dependencies to {output_path}")

    def get_removed_dependencies(self) -> List[Dict[str, str]]:
        """
        Detect dependencies that were in the previous CSV but not in the current extraction.
        Returns list of removed dependencies with their previous information.
        """
        if not self.previous_latest_dependencies:
            return []
        
        # Build set of current dependency keys
        current_keys = set()
        for dep in self.dependencies:
            key = f"{dep['Component']}:{dep['Category']}:{dep['Dependency Name']}"
            current_keys.add(key)
        
        # Find dependencies in previous but not in current
        removed = []
        for prev_key, prev_dep in self.previous_latest_dependencies.items():
            if prev_key not in current_keys:
                removed.append({
                    'Component': prev_dep.get('Component', ''),
                    'Category': prev_dep.get('Category', ''),
                    'Dependency Name': prev_dep.get('Dependency Name', ''),
                    'Version': prev_dep.get('Version', ''),
                    'Source File': prev_dep.get('Source File', ''),
                    'Critical': prev_dep.get('Critical', 'No')
                })
        
        return removed
    
    def write_unversioned_report(self, output_path: Path) -> None:
        """Write a separate report of unversioned dependencies."""
        unversioned = [
            dep for dep in self.dependencies 
            if dep["Version"] in ["unspecified", "N/A", "", "latest"]
        ]
        
        if not unversioned:
            print("✓ No unversioned dependencies to report")
            return
        
        print(f"Writing unversioned dependencies report to {output_path}...")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "Component", "Category", "Dependency Name", "Version",
                "Source File", "GitHub URL", "Notes", "Recommendation"
            ])
            writer.writeheader()
            
            for dep in unversioned:
                dep_copy = dep.copy()
                dep_copy["Recommendation"] = "Pin to specific version for reproducible builds"
                writer.writerows([dep_copy])
        
        print(f"✓ Written {len(unversioned)} unversioned dependencies to {output_path}")

    def print_summary(self) -> None:
        """Print comprehensive summary statistics."""
        components = {}
        unversioned = []
        unversioned_by_component = {}
        
        for dep in self.dependencies:
            comp = dep["Component"]
            components[comp] = components.get(comp, 0) + 1
            
            # Track unversioned dependencies
            if dep["Version"] in ["unspecified", "N/A", "", "latest"]:
                unversioned.append(dep)
                if comp not in unversioned_by_component:
                    unversioned_by_component[comp] = []
                unversioned_by_component[comp].append(dep)
        
        total_deps = len(self.dependencies)
        
        # Print extraction summary
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"\nFiles Processed: {len(self.processed_files)}")
        if self.processed_files:
            for file in sorted(self.processed_files)[:10]:
                print(f"  ✓ {file}")
            if len(self.processed_files) > 10:
                print(f"  ... and {len(self.processed_files) - 10} more")
        
        if self.failed_files:
            print(f"\nFiles Failed: {len(self.failed_files)}")
            for failed in self.failed_files:
                print(f"  ✗ {failed['file']} ({failed['component']}): {failed['reason']}")
        
        if self.missing_files:
            print(f"\nFiles Missing: {len(self.missing_files)}")
            for missing in self.missing_files:
                req_str = "REQUIRED" if missing.get('required') else "optional"
                print(f"  - {missing['component']}/{missing['type']} ({req_str})")
                print(f"    Tried: {missing['patterns']}")
        
        if self.warnings:
            print(f"\nWarnings: {len(self.warnings)}")
            for warning in self.warnings[:5]:
                print(f"  ⚠ {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more warnings")
        
        print("\n" + "="*60)
        print("DEPENDENCY SUMMARY")
        print("="*60)
        
        print("\nSummary by component:")
        for comp, count in sorted(components.items()):
            print(f"  {comp:15s}: {count:3d} dependencies")
        
        print(f"\nTotal dependencies: {total_deps}")
        
        # Check for unversioned dependencies
        if unversioned:
            print(f"\n⚠️  WARNING: Found {len(unversioned)} unversioned/unpinned dependencies!")
            print(f"\nUnversioned dependencies by component:")
            for comp in sorted(unversioned_by_component.keys()):
                deps = unversioned_by_component[comp]
                print(f"\n  {comp} ({len(deps)} unversioned):")
                for dep in deps[:10]:  # Show first 10
                    print(f"    - {dep['Dependency Name']:30s} ({dep['Category']})")
                if len(deps) > 10:
                    print(f"    ... and {len(deps) - 10} more")
            
            print(f"\n  💡 Tip: Unversioned dependencies can lead to:")
            print(f"     - Non-reproducible builds")
            print(f"     - Unexpected breaking changes")
            print(f"     - Difficulty tracking security vulnerabilities")
            print(f"\n  Consider pinning versions in requirements files for better control.")
        else:
            print(f"\n✓ All dependencies have version specifiers")
        
        # Check against baseline and warn if exceeded
        if total_deps > self.baseline_count:
            increase = total_deps - self.baseline_count
            print(f"\n⚠️  WARNING: Dependency count has increased!")
            print(f"   Baseline: {self.baseline_count} dependencies")
            print(f"   Current:  {total_deps} dependencies")
            print(f"   Increase: +{increase} dependencies")
            print(f"\n   Please review new dependencies and update baseline if expected.")
        elif total_deps < self.baseline_count:
            decrease = self.baseline_count - total_deps
            print(f"\n✓ Dependency count decreased by {decrease} (baseline: {self.baseline_count})")
        else:
            print(f"\n✓ Dependency count matches baseline ({self.baseline_count})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract dependency versions from Dynamo Dockerfiles and requirements"
    )
    
    # Generate default output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_output = f"dependency_versions_{timestamp}.csv"
    
    parser.add_argument(
        "--output", "-o",
        default=default_output,
        help=f"Output CSV file path (default: {default_output})"
    )
    parser.add_argument(
        "--latest-csv",
        type=Path,
        default=None,
        help="Path to latest nightly CSV for comparison (default: auto-detect dependency_versions_latest.csv)"
    )
    parser.add_argument(
        "--release-csv",
        type=Path,
        default=None,
        help="Path to latest release CSV for comparison (default: auto-detect latest vX.X.X in releases/)"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (default: auto-detect)"
    )
    parser.add_argument(
        "--github-repo",
        default="ai-dynamo/dynamo",
        help="GitHub repository (default: ai-dynamo/dynamo)"
    )
    parser.add_argument(
        "--github-branch",
        default="main",
        help="GitHub branch for URLs (default: main)"
    )
    parser.add_argument(
        "--baseline",
        type=int,
        default=251,
        help="Baseline dependency count for warnings (default: 251)"
    )
    parser.add_argument(
        "--report-unversioned",
        action="store_true",
        help="Generate separate report of unversioned dependencies"
    )
    parser.add_argument(
        "--report-removed",
        type=str,
        help="Output removed dependencies to JSON file (e.g., removed.json)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (default: .github/workflows/extract_dependency_versions_config.yaml)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing required files (default: warn only)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and file paths without extracting"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files would be processed without extracting"
    )
    
    args = parser.parse_args()
    
    # Auto-detect repo root
    if args.repo_root is None:
        script_path = Path(__file__).resolve()
        # Script is in .github/workflows/ directory, repo root is two levels up
        repo_root = script_path.parent.parent.parent
    else:
        repo_root = args.repo_root
    
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    
    # Auto-detect latest nightly CSV if not specified
    latest_csv = args.latest_csv
    if latest_csv is None:
        # Look for dependency_versions_latest.csv in .github/reports/
        reports_dir = repo_root / ".github/reports"
        latest_candidate = reports_dir / "dependency_versions_latest.csv"
        if latest_candidate.exists():
            latest_csv = latest_candidate
            print(f"Auto-detected latest nightly CSV: {latest_csv.relative_to(repo_root)}")
    
    # Auto-detect latest release CSV if not specified
    release_csv = args.release_csv
    if release_csv is None:
        # Look for latest dependency_versions_vX.X.X.csv in .github/reports/releases/
        releases_dir = repo_root / ".github/reports/releases"
        if releases_dir.exists():
            release_csvs = sorted(releases_dir.glob("dependency_versions_v*.csv"), reverse=True)
            if release_csvs:
                release_csv = release_csvs[0]
                print(f"Auto-detected latest release CSV: {release_csv.relative_to(repo_root)}")
    
    print(f"Repository root: {repo_root}")
    print(f"Output file: {output_path}")
    print(f"GitHub repo: {args.github_repo}")
    print(f"GitHub branch: {args.github_branch}")
    print(f"Baseline count: {args.baseline}")
    if args.config:
        print(f"Config file: {args.config}")
    if latest_csv:
        print(f"Latest nightly CSV: {latest_csv}")
    if release_csv:
        print(f"Latest release CSV: {release_csv}")
    print()
    
    # Initialize extractor
    extractor = DependencyExtractor(repo_root, args.github_repo, args.github_branch, args.config, latest_csv, release_csv)
    extractor.baseline_count = args.baseline
    
    # Validate mode - check config and files without extracting
    if args.validate:
        print("Running validation...")
        print(f"\nConfiguration loaded: {'✓' if extractor.config else '✗'}")
        if extractor.warnings:
            print(f"\nConfiguration warnings:")
            for warning in extractor.warnings:
                print(f"  ⚠ {warning}")
        
        is_valid = extractor.validate_critical_files(strict_mode=args.strict)
        
        if extractor.missing_files:
            print(f"\nMissing files detected:")
            for missing in extractor.missing_files:
                req_str = "REQUIRED" if missing.get('required') else "optional"
                print(f"  - {missing['component']}/{missing['type']} ({req_str})")
                print(f"    Patterns: {missing['patterns']}")
        
        if is_valid:
            print("\n✓ Validation passed")
            return
        else:
            print("\n✗ Validation failed")
            exit(1)
    
    # Dry-run mode - show what would be processed
    if args.dry_run:
        print("Dry-run mode: showing files that would be processed...\n")
        
        if 'components' in extractor.config:
            for component_name, component_config in extractor.config['components'].items():
                print(f"{component_name}:")
                
                dockerfiles = component_config.get('dockerfiles', [])
                if dockerfiles:
                    found = extractor.discover_files(dockerfiles)
                    if found:
                        print(f"  Dockerfiles: {[str(f.relative_to(repo_root)) for f in found]}")
                    else:
                        print(f"  Dockerfiles: None found (patterns: {dockerfiles})")
                
                scripts = component_config.get('scripts', [])
                if scripts:
                    found = extractor.discover_files(scripts)
                    if found:
                        print(f"  Scripts: {[str(f.relative_to(repo_root)) for f in found]}")
                
                go_modules = component_config.get('go_modules', [])
                if go_modules:
                    found = extractor.discover_files(go_modules)
                    if found:
                        print(f"  Go modules: {[str(f.relative_to(repo_root)) for f in found]}")
                
                requirements = component_config.get('requirements', [])
                if requirements:
                    found = extractor.discover_requirements_files(requirements)
                    if found:
                        print(f"  Requirements: {[str(f.relative_to(repo_root)) for f in found]}")
                
                pyproject = component_config.get('pyproject', [])
                if pyproject:
                    found = extractor.discover_files(pyproject)
                    if found:
                        print(f"  PyProject: {[str(f.relative_to(repo_root)) for f in found]}")
                
                print()
        
        print("✓ Dry-run complete")
        return
    
    # Normal extraction mode
    extractor.extract_all()
    
    # Check if strict mode and there are failures
    if args.strict and (extractor.failed_files or extractor.missing_files):
        print("\n✗ Extraction failed in strict mode due to missing/failed files")
        extractor.print_summary()
        exit(1)
    
    # Write CSV
    extractor.write_csv(output_path)
    
    # Write unversioned report if requested
    if args.report_unversioned:
        unversioned_path = output_path.parent / f"{output_path.stem}_unversioned{output_path.suffix}"
        extractor.write_unversioned_report(unversioned_path)
    
    # Write removed dependencies report if requested
    if args.report_removed:
        removed_deps = extractor.get_removed_dependencies()
        removed_path = Path(args.report_removed)
        with open(removed_path, 'w') as f:
            json.dump({
                'count': len(removed_deps),
                'removed': removed_deps
            }, f, indent=2)
        print(f"✓ Written {len(removed_deps)} removed dependencies to {removed_path}")
    
    # Print summary
    extractor.print_summary()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
