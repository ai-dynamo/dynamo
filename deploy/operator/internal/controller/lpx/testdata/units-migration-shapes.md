<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Same-artifact migration shapes

`TestUnitsMigrationShapes` compares the base reader at
`20a9321cf16eb11e95fd12194c3033c210f7b2be` with the units reader. Each JSON capture
contains its exact deployment bytes, unchanged DGD, old PCS, requests and
configuration. These are synthetic contracts with matching legacy and units
metadata, not the outputs of a new model compile or live migration results.
The base reader and renderer ran unchanged in a detached checkout.

| Shape | Geometry in both views | Agents before / after | PCS decision | Request-spec difference |
| --- | --- | --- | --- | --- |
| HX unpacked | Two 16-chip reservations, each extent [16,1,1,1], 16 identity pairs | 2 / 2 | Keep PCS; roll templates | Remove legacy allocation metadata |
| HX packed | Partitions [7,8], 8 chips each, one extent [16,1,1,1]; no connector | 1 / 1 | Keep PCS; roll templates | Remove legacy allocation metadata |
| XT hybrid with local root 9 | Remote chain [7,8], each c16, XT gap 0; local unit [9] | 4 / 4 | Keep PCS; roll templates | None |

All three requests change their WorkloadDigest annotation, and all three Pod
templates change. `pipelineRequestMatches` compares annotations as well as the
spec, so all three requests retire and are published again after Grove observes
the new role templates. Equal XT spec does not preserve the old request UID.
The scheduler does not interpret allocationMetadata. The test compares the complete request spec after accounting for the
stated metadata removal. It also compares the controller's immutable-layout
predicate, Agent counts, partitions, connectors and node-local mappings.
`TestUnitsOperatorOnlyUpgrade` covers the keep-PCS lifecycle. The separate
`TestUnitsReplacementAndRollback` covers incompatible geometry and foreground
replacement. These cases do not imply that every migration preserves layout,
or that a rollout is lossless.

The authored config mounts are removed before either reader runs, as in
`units-old-operator.md`. This isolates placement from the template refusal and
runtime-image compatibility gates. The hybrid input selects local root 9.

Embedded deployment SHA-256 values:

- `hx-unpacked`: `08ad3d7c968fad704f554c140cc13c8e6c8dc956a19cad62a68eb46db083b8b3`.
- `hx-packed`: `09d33a9df399db27bf59da81f528e0aebe53e6d51936dc05595f8bbc02330db1`.
- `xt-hybrid-local`: `543dfa9dfded0687aea1912efa321cb6d32592ca81fd5a19b0d46230904430fa`.

To reproduce, extract the three `deployment` base64 values into a JSON map from
shape name to base64 bytes at `/tmp/units-shape-artifacts.json`. Run this capture
test in the base controller/lpx package. Do not regenerate with the new reader.

```go
package lpx
import (
 "encoding/json"
 "net/url"
 "os"
 "path/filepath"
 "slices"
 "testing"
"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
 "github.com/stretchr/testify/require"
 corev1 "k8s.io/api/core/v1"
)
func TestCaptureOldShapeRenders(t *testing.T) {
 input,e:=os.ReadFile("/tmp/units-shape-artifacts.json");require.NoError(t,e)
 var artifacts map[string][]byte
 require.NoError(t,json.Unmarshal(input,&artifacts))
 for name,data:=range artifacts { t.Run(name,func(t *testing.T) {
 root:=t.TempDir()
 require.NoError(t,os.Mkdir(filepath.Join(root,"units"),0700))

 require.NoError(t,os.WriteFile(filepath.Join(root,"units","deployment.v1.capnp.bin"),data,0600))
 registry,e:=lpx.NewModelRegistry((&url.URL{Scheme:lpx.BuildSchemeFile,Path:root}).String(),nil);require.NoError(t,e)
 pipeline:=lpx.PipelineSingle
 if name=="xt-hybrid-local" {pipeline=lpx.PipelineHybrid}
 dgd:=loadTestDGD(t,pipeline,"units")
 if name=="xt-hybrid-local" {dgd.Spec.Components[0].LPX.Experimental=&v1beta1.LPXExperimentalSpec{LocalPartitions:&v1beta1.LPXLocalPartitions{Mode:v1beta1.LPXLocalPartitionsModeIDs,IDs:[]int64{9}}}}
 for i:=range dgd.Spec.Components[0].Roles {
  s:=&dgd.Spec.Components[0].Roles[i].PodTemplate.Spec
  for _, cs:=range [][]corev1.Container{s.Containers,s.InitContainers}{for j:=range cs {cs[j].VolumeMounts=slices.DeleteFunc(cs[j].VolumeMounts,func(m corev1.VolumeMount)bool{return m.Name=="config"})}}
 }
 child:=newLPXTestDeployment(t,dgd)
 r,selected:=newPreparedLPXTestReconciler(t,registry,t.Context(),child,dgd)
 pcs,resources,e:=r.renderPodCliqueSet(t.Context(),child,dgd,map[string]*lpx.Workload{selected.workload.ServingComponentName():selected.workload},map[string]*lpx.MaterializationPlan{selected.workload.ServingComponentName():selected.plan},nil);require.NoError(t,e)
 payload,e:=json.MarshalIndent(map[string]any{"base":"20a9321cf16eb11e95fd12194c3033c210f7b2be","deployment":data,"dgd":dgd,"pcs":pcs,"requests":selected.requests,"resources":resources},"","  ");require.NoError(t,e)
 require.NoError(t,os.WriteFile("/tmp/units-old-"+name+".json",append(payload,'\n'),0600))
}) }
}
```
