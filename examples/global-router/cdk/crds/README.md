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

The upstream files carry this notice, which JSON cannot hold as a comment:

```text
Copyright 2020 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

To refresh them for another Karmada release `X.Y.Z`, run this from the CDK app's directory. A
file is replaced only when its download converts to a CRD; otherwise the loop prints
`Not refreshed`:

```bash
for kind in serviceexports serviceimports; do
  file="multicluster.x-k8s.io_${kind}"
  curl -fsSL -o "/tmp/${file}.yaml" "https://raw.githubusercontent.com/karmada-io/karmada/vX.Y.Z/charts/karmada/_crds/bases/multicluster/${file}.yaml" &&
    python3 -c 'import json, sys, yaml; crd = yaml.safe_load(open(sys.argv[1])); assert crd["kind"] == "CustomResourceDefinition"; print(json.dumps(crd, indent=2))' "/tmp/${file}.yaml" > "/tmp/${file}.json" &&
    mv "/tmp/${file}.json" "crds/${file}.json" ||
    echo "Not refreshed: ${file}"
done
```

Then set `KARMADA_VERSION` in `lib/multi-cluster.ts` to the same release, check the cfssl and
kubectl image pins next to it, and update the release in this file.
