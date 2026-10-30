# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Prepare the in-cluster Harbor client; the task guest is x86_64."""

import hashlib
import json
import platform
import shutil
import urllib.request
from pathlib import Path

root = Path("/work")
arch = {"aarch64": "arm64", "x86_64": "amd64"}[platform.machine()]
url = (
    "https://github.com/agent-substrate/substrate/releases/download/"
    f"v1.0.0-rc.1/kubectl-ate-linux-{arch}"
)
cli = root / "kubectl-ate-release"
with urllib.request.urlopen(url, timeout=90) as response, cli.open("wb") as output:
    shutil.copyfileobj(response, output)
with urllib.request.urlopen(url + ".sha256", timeout=30) as response:
    expected = response.read().decode().split()[0]
with cli.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
if actual != expected:
    raise ValueError("kubectl-ate release checksum mismatch")
cli.chmod(0o755)
wrapper = Path("/usr/local/bin/kubectl-ate")
wrapper.write_text(
    "#!/bin/sh\nexec /work/kubectl-ate-release --endpoint api.ate-system.svc:443 "
    '--token-file /var/run/secrets/ateapi/token "$@"\n'
)
wrapper.chmod(0o755)

config = {
    "apiVersion": "v1",
    "kind": "Config",
    "current-context": "in-cluster",
    "clusters": [
        {
            "name": "cluster",
            "cluster": {
                "server": "https://kubernetes.default.svc",
                "certificate-authority": "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
            },
        }
    ],
    "users": [
        {
            "name": "runner",
            "user": {
                "tokenFile": "/var/run/secrets/kubernetes.io/serviceaccount/token",
            },
        }
    ],
    "contexts": [
        {
            "name": "in-cluster",
            "context": {
                "cluster": "cluster",
                "user": "runner",
                "namespace": "ate-env",
            },
        }
    ],
}
config_dir = Path.home() / ".kube"
config_dir.mkdir(exist_ok=True)
(config_dir / "config").write_text(json.dumps(config))
print(f"Prepared Harbor runner; verified kubectl-ate SHA256 {actual}")
