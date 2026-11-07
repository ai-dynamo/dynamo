<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Operator-only upgrade fixture

`units-old-operator.json` captures the renderer and request resolver at Dynamo
`20a9321cf16eb11e95fd12194c3033c210f7b2be`. It contains the input DGD, exact
serialized deployment contract (base64), old PCS, LPRs, and ConfigMap. The base
reader ran unchanged in a detached checkout; no new reader was made to emulate
legacy behavior by removing units. The fixture capture passed the base's
`TestCaptureOldUnitsRender` on 2026-11-07.

The artifact is the `units` case of `newUnitsTestRegistry`: LP20 partitions 7 and
8, 16 chips each, devicesPerNode 8, one unit with two nodes=2 reservations and an
XT connector. It also retains the legacy selected chain [7,8]. The old renderer
uses those legacy fields even though the artifact already contains units.

The authored DGD is `loadTestDGD(PipelineSingle, "units")` with the `config`
mounts removed before either reader runs. This isolates the operator lifecycle
from the separate units-template rejection test. It is not a claim that legacy
runtime binaries can run without their configuration. Runtime compatibility is
an external release gate.

To reproduce, put the embedded deployment bytes in a file registry entry named
`units`. In a test in the base controller/lpx package, load the embedded DGD,
construct `newLPXTestDeployment` and `newPreparedLPXTestReconciler`, and call
`renderPodCliqueSet` with the resolved workload and plan. Save its PCS and
resources with `selected.requests`. The current test assigns API identities,
client identity, and Grove observations through the existing test harness.

Do not regenerate this fixture with the new reader. The regression changes only
the operator implementation: DGD spec/generation, input revision, artifact bytes,
and build reference remain fixed. Equal geometry updates the PCS template and
retires requests only after Grove observes new role templates. The test preserves
the old ConfigMap until readiness and checks the settled reconcile for no changes.

Embedded deployment SHA-256: `17575a77dd63eac47ac980b51cc7dd9e7536920f1505b2db125ecc382dbcd711`.

Capture source (run as a temporary `_test.go` in the base controller/lpx package;
extract the embedded bytes to `/tmp/upgrade-deployment.bin` first):

```go
package lpx
import (
 "encoding/json"
 "net/url"
 "os"
 "path/filepath"
 "slices"
 "testing"
 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
 "github.com/stretchr/testify/require"
 corev1 "k8s.io/api/core/v1"
)
func TestCaptureOldUnitsRender(t *testing.T) {
 root:=t.TempDir()
 require.NoError(t,os.Mkdir(filepath.Join(root,"units"),0700))
 data,e:=os.ReadFile("/tmp/upgrade-deployment.bin");require.NoError(t,e)
 require.NoError(t,os.WriteFile(filepath.Join(root,"units","deployment.v1.capnp.bin"),data,0600))
 registry,e:=lpx.NewModelRegistry((&url.URL{Scheme:lpx.BuildSchemeFile,Path:root}).String(),nil);require.NoError(t,e)
 dgd:=loadTestDGD(t,lpx.PipelineSingle,"units")
 for i:=range dgd.Spec.Components[0].Roles {
  s:=&dgd.Spec.Components[0].Roles[i].PodTemplate.Spec
  for _, cs:=range [][]corev1.Container{s.Containers,s.InitContainers}{for j:=range cs {cs[j].VolumeMounts=slices.DeleteFunc(cs[j].VolumeMounts,func(m corev1.VolumeMount)bool{return m.Name=="config"})}}
 }
 child:=newLPXTestDeployment(t,dgd)
 r,selected:=newPreparedLPXTestReconciler(t,registry,t.Context(),child,dgd)
 pcs,resources,e:=r.renderPodCliqueSet(t.Context(),child,dgd,map[string]*lpx.Workload{selected.workload.ServingComponentName():selected.workload},map[string]*lpx.MaterializationPlan{selected.workload.ServingComponentName():selected.plan},nil);require.NoError(t,e)
 payload,e:=json.MarshalIndent(map[string]any{"base":"20a9321cf16eb11e95fd12194c3033c210f7b2be","deployment":data,"dgd":dgd,"pcs":pcs,"requests":selected.requests,"resources":resources},"","  ");require.NoError(t,e)
 require.NoError(t,os.WriteFile("/tmp/units-old-operator.json",append(payload,'\n'),0600))
}
```
