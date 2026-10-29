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

`ServiceExport` and `ServiceImport` from the SIG Multicluster Multi-Cluster Services (MCS) API,
installed in every cluster that sets `multiCluster`. They are copied unchanged from
`sigs.k8s.io/mcs-api` v0.5.2, `config/crd/`, and stored as JSON so the CDK app needs no YAML
parser.

Both CRDs serve `multicluster.x-k8s.io/v1alpha1` and `multicluster.x-k8s.io/v1beta1`, with the
same schema and conversion strategy `None`, and store `v1beta1`. Karmada 1.19.0 reads and writes
`v1alpha1`, and an MCS controller built on `sigs.k8s.io/mcs-api` v0.5 uses `v1beta1`, so both
work with the same objects. The `v1alpha1` schema is a superset of the one in Karmada's chart: it
adds only optional fields, and a `ServiceImport` may list a second IP for dual-stack.

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

To refresh them for another mcs-api release `vX.Y.Z`, run this from the CDK app's directory. The
files at a release tag are the same as `config/crd/` in the Go module `sigs.k8s.io/mcs-api@vX.Y.Z`.
A file is replaced only when its download converts to a CRD that still serves both versions;
otherwise the loop prints `Not refreshed`:

```bash
for kind in serviceexports serviceimports; do
  file="multicluster.x-k8s.io_${kind}"
  curl -fsSL -o "/tmp/${file}.yaml" "https://raw.githubusercontent.com/kubernetes-sigs/mcs-api/vX.Y.Z/config/crd/${file}.yaml" &&
    python3 -c 'import json, sys, yaml; crd = yaml.safe_load(open(sys.argv[1])); assert crd["kind"] == "CustomResourceDefinition"; assert {"v1alpha1", "v1beta1"} <= {v["name"] for v in crd["spec"]["versions"] if v["served"]}; print(json.dumps(crd, indent=2))' "/tmp/${file}.yaml" > "/tmp/${file}.json" &&
    mv "/tmp/${file}.json" "crds/${file}.json" ||
    echo "Not refreshed: ${file}"
done
```

The loop refuses a release that stops serving `v1alpha1`: Karmada reads only `v1alpha1`, and in a
cluster that had Karmada's copies, Kubernetes rejects a CRD update that removes a version listed
in the CRD's `status.storedVersions`.

After a refresh, update the release in this file, in the `MCS_CRD_FILES` comment in
`lib/multi-cluster.ts`, and in the app's README, then run `pnpm test`, which checks the served and
storage versions.

Moving `KARMADA_VERSION` in `lib/multi-cluster.ts` to another Karmada release does not change
these files. Check the cfssl and kubectl image pins next to it, and check that the release still
reads an MCS version these files serve: its chart's copies are in
`charts/karmada/_crds/bases/multicluster/`.
