<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# SIG Multicluster CRDs

`ServiceExport` and `ServiceImport` (`multicluster.x-k8s.io/v1alpha1`), installed in every
cluster that sets `multiCluster`. They are copied unchanged from Karmada 1.19.0,
`charts/karmada/_crds/bases/multicluster/`, and stored as JSON so the CDK app needs no YAML
parser.

To refresh them for another Karmada release `X.Y.Z`:

```bash
for kind in serviceexports serviceimports; do
  curl -fsSL "https://raw.githubusercontent.com/karmada-io/karmada/vX.Y.Z/charts/karmada/_crds/bases/multicluster/multicluster.x-k8s.io_${kind}.yaml" |
    python3 -c 'import json, sys, yaml; print(json.dumps(yaml.safe_load(sys.stdin), indent=2))' \
      > "multicluster.x-k8s.io_${kind}.json"
done
```
