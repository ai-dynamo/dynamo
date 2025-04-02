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

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import bentoml
from bentoml._internal.utils import add_experimental_docstring

HELM_VERSION = "v3.17.0"
KUBECTL_VERSION = "v1.29.0"

def get_sdk_path() -> Path:
    """Get the path to the Dynamo SDK installation."""
    return Path(bentoml.__file__).parent

def get_binary_path(binary_name: str) -> Optional[str]:
    """Get the path to a binary in the SDK's bin directory.
    
    Args:
        binary_name: Name of the binary to look for
        
    Returns:
        Path to the binary if it exists, None otherwise
    """
    sdk_path = get_sdk_path()
    binary_path = sdk_path / "cli" / "bin" / binary_name
    if binary_path.exists():
        return str(binary_path)
    return None

@add_experimental_docstring
def ensure_helm() -> str:
    """Ensure helm binary is available and return its path.
    
    Returns:
        Path to the helm binary
    """
    helm_path = get_binary_path("helm")
    if helm_path:
        return helm_path
        
    # Download helm if not found
    system = platform.system().lower()
    arch = platform.machine().lower()
    if arch == "x86_64":
        arch = "amd64"
    elif arch == "aarch64":
        arch = "arm64"
        
    helm_url = f"https://get.helm.sh/helm-{HELM_VERSION}-{system}-{arch}.tar.gz"
    
    sdk_path = get_sdk_path()
    bin_dir = sdk_path / "cli" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and extract helm
    subprocess.run(["curl", "-L", helm_url, "-o", "helm.tar.gz"], check=True)
    subprocess.run(["tar", "-xzf", "helm.tar.gz", "-C", str(bin_dir), "--strip-components=1", f"{system}-{arch}/helm"], check=True)
    os.remove("helm.tar.gz")
    
    return str(bin_dir / "helm")

@add_experimental_docstring
def ensure_kubectl() -> str:
    """Ensure kubectl binary is available and return its path.
    
    Returns:
        Path to the kubectl binary
    """
    kubectl_path = get_binary_path("kubectl")
    if kubectl_path:
        return kubectl_path
        
    # Download kubectl if not found
    system = platform.system().lower()
    arch = platform.machine().lower()
    if arch == "x86_64":
        arch = "amd64"
    elif arch == "aarch64":
        arch = "arm64"
        
    kubectl_url = f"https://dl.k8s.io/release/{KUBECTL_VERSION}/bin/{system}/{arch}/kubectl"
    
    sdk_path = get_sdk_path()
    bin_dir = sdk_path / "cli" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    # Download kubectl
    subprocess.run(["curl", "-L", kubectl_url, "-o", str(bin_dir / "kubectl")], check=True)
    os.chmod(str(bin_dir / "kubectl"), 0o755)
    
    return str(bin_dir / "kubectl") 