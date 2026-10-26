# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render Karmada exports/imports for existing regional backend/router Services."""

import argparse
import json


def render(namespace, stems, members):
    services = {}
    for stem in stems:
        services[f"{stem}-frontend"] = [{"name": "http", "port": 8000, "protocol": "TCP"}]
        services[f"{stem}-pool-relay"] = [
            {"name": "wan-grpc", "port": 50051, "protocol": "TCP"},
            {"name": "stats-grpc", "port": 50052, "protocol": "TCP"},
        ]
    services["global-router"] = [{"name": "http", "port": 8000, "protocol": "TCP"}]
    mcs_api = "multicluster.x-k8s.io/v1alpha1"
    items = [
        {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": namespace,
                "labels": {"namespace.karmada.io/skip-auto-propagation": "true"},
            },
        }
    ]
    for name, ports in services.items():
        metadata = {"name": name, "namespace": namespace}
        items.extend(
            [
                {"apiVersion": mcs_api, "kind": "ServiceExport", "metadata": metadata},
                {
                    "apiVersion": mcs_api,
                    "kind": "ServiceImport",
                    "metadata": metadata,
                    "spec": {"type": "ClusterSetIP", "ports": ports},
                },
            ]
        )
    for kind, suffix, dependencies in [
        ("ServiceExport", "exports", False),
        ("ServiceImport", "imports", True),
    ]:
        spec = {
            "resourceSelectors": [
                {"apiVersion": mcs_api, "kind": kind, "name": name} for name in services
            ],
            "propagateDeps": dependencies,
            "placement": {"clusterAffinity": {"clusterNames": members}},
        }
        if kind == "ServiceExport":
            # v1.19 initializes export informers from successfully applied Works.
            # Own only these isolated export objects, never the regional Services.
            spec["conflictResolution"] = "Overwrite"
        items.append(
            {
                "apiVersion": "policy.karmada.io/v1alpha1",
                "kind": "PropagationPolicy",
                "metadata": {"name": f"rr-{suffix}", "namespace": namespace},
                "spec": spec,
            }
        )
    return {"apiVersion": "v1", "kind": "List", "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="global-router-v2-rr")
    parser.add_argument(
        "--stem",
        action="append",
        help="Pool Service stem; repeat for each pool in the namespace (default rr-mocker)",
    )
    parser.add_argument("--member", action="append", required=True)
    args = parser.parse_args()
    stems = args.stem or ["rr-mocker"]
    print(json.dumps(render(args.namespace, stems, args.member), indent=2))


if __name__ == "__main__":
    main()
