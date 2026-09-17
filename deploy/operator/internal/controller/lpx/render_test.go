// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	"gotest.tools/v3/golden"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

func newLPXRenderDeployment(t *testing.T, source *v1beta1.DynamoGraphDeployment) *v1alpha1.LPXGraphDeployment {
	t.Helper()
	revision, err := dynamo.LPXInputRevision(source, "")
	require.NoError(t, err)
	return &v1alpha1.LPXGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: source.Name, Namespace: source.Namespace, UID: "lpx-render-uid", Generation: 1,
			Annotations:     map[string]string{lpx.DGDGenerationAnnotation: strconv.FormatInt(source.Generation, 10)},
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(source, v1beta1.DynamoGraphDeploymentGVK)},
		},
		Spec: v1alpha1.LPXGraphDeploymentSpec{InputRevision: revision},
	}
}

func TestLPXRenderingChecksFinalPodCliqueSetSize(t *testing.T) {
	t.Log("Resolve a real LPU workload including inherited scheduler metadata")
	payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/node-local-v2-lpu-only.input.yaml")
	require.NoError(t, err)
	source := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, source))
	source.Annotations = map[string]string{"kai.scheduler/padding": ""}
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), source, newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, source)))
	require.NoError(t, err)
	config := &configv1alpha1.OperatorConfiguration{}
	kube := newTestLPXClient(t)
	runtimeConfig := &controller_common.RuntimeConfig{}
	pcs, _, err := renderPodCliqueSet(t.Context(), source, config, runtimeConfig, kube, nil,
		selected, plan, newLPXRenderDeployment(t, source))
	require.NoError(t, err)
	serialized, err := json.Marshal(pcs)
	require.NoError(t, err)

	t.Log("Accept exactly one MiB including final identity, discovery and scheduler metadata")
	source.Annotations["kai.scheduler/padding"] = strings.Repeat("x", lpx.MaxRenderedPodCliqueSetBytes-len(serialized))
	pcs, _, err = renderPodCliqueSet(t.Context(), source, config, runtimeConfig, kube, nil,
		selected, plan, newLPXRenderDeployment(t, source))
	require.NoError(t, err)
	serialized, err = json.Marshal(pcs)
	require.NoError(t, err)
	require.Len(t, serialized, lpx.MaxRenderedPodCliqueSetBytes)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Annotations, "kai.scheduler/padding")
	}

	t.Log("Reject one additional final-metadata byte as a selected-render failure before publication")
	source.Annotations["kai.scheduler/padding"] += "x"
	pcs, resources, err := renderPodCliqueSet(t.Context(), source, config, runtimeConfig, kube, nil,
		selected, plan, newLPXRenderDeployment(t, source))
	require.ErrorContains(t, err, "rendered LPX PodCliqueSet is 1048577 bytes; maximum is 1048576")
	require.Nil(t, pcs)
	require.Nil(t, resources)
}

func TestLPXHybridPreservesKVTransferTopology(t *testing.T) {
	t.Log("Create immutable native builds for this story's hybrid cases")
	registry := newTestDataModelRegistry(t, t.TempDir())

	for _, fixture := range []string{"node-local-v2-hybrid", "node-local-v3-hx-hybrid"} {
		t.Run(fixture, func(t *testing.T) {
			t.Log("Resolve a hybrid engine using a real ClusterTopologyBinding for KV transfer")
			payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/" + fixture + ".input.yaml")
			require.NoError(t, err)
			source := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, yaml.Unmarshal(payload, source))
			source.Spec.Experimental = &v1beta1.DynamoGraphDeploymentExperimentalSpec{
				KvTransferPolicy: &v1beta1.KvTransferPolicy{ClusterTopologyName: "fabric", Domain: "rack"},
			}
			binding := &grovev1alpha1.ClusterTopologyBinding{ObjectMeta: metav1.ObjectMeta{Name: "fabric"},
				Spec: grovev1alpha1.ClusterTopologyBindingSpec{Levels: []grovev1alpha1.TopologyLevel{
					{Domain: "zone", Key: "topology.kubernetes.io/zone"}, {Domain: "rack", Key: "nvidia.com/rack"},
				}},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			require.NoError(t, grovev1alpha1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).WithObjects(binding).Build()
			child := newLPXRenderDeployment(t, source)
			selected, err := lpx.ResolveSelectedWorkload(t.Context(), source, registry)
			require.NoError(t, err)
			plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, source)))
			require.NoError(t, err)
			config := &configv1alpha1.OperatorConfiguration{}
			pcs, _, err := renderPodCliqueSet(t.Context(), source, config, &controller_common.RuntimeConfig{}, kube, nil, selected, plan, child)
			require.NoError(t, err)

			t.Log("The Cyborg role receives every topology domain, not an empty mounted directory")
			items := map[string]string{}
			for _, clique := range pcs.Spec.Template.Cliques {
				if clique.Name != plan.CyborgTemplate {
					continue
				}
				require.Equal(t, binding.Name, clique.Annotations[commonconsts.KubeAnnotationTopologyClusterTopologyName])
				for _, volume := range clique.Spec.PodSpec.Volumes {
					if volume.Name == "topology-labels" {
						require.NotNil(t, volume.DownwardAPI)
						for _, item := range volume.DownwardAPI.Items {
							items[item.Path] = item.FieldRef.FieldPath
						}
					}
				}
			}
			require.Equal(t, map[string]string{
				"zone": "metadata.labels['" + commonconsts.DynamoTopologyLabelKey("zone") + "']",
				"rack": "metadata.labels['" + commonconsts.DynamoTopologyLabelKey("rack") + "']",
			}, items)

			t.Log("Binding changes and deletion must invalidate the render instead of silently losing routing metadata")
			binding.Spec.Levels = binding.Spec.Levels[:1]
			require.NoError(t, kube.Update(t.Context(), binding))
			_, _, err = renderPodCliqueSet(t.Context(), source, config, &controller_common.RuntimeConfig{}, kube, nil, selected, plan, child)
			require.ErrorContains(t, err, `domain "rack" does not exist`)
			require.NoError(t, kube.Delete(t.Context(), binding))
			_, _, err = renderPodCliqueSet(t.Context(), source, config, &controller_common.RuntimeConfig{}, kube, nil, selected, plan, child)
			require.ErrorContains(t, err, "was not found")
		})
	}
}

func TestLPXRenderingPreservesInputs(t *testing.T) {
	const restartToken = "2026-09-08T00:00:00Z"
	t.Log("Create immutable native builds while keeping each rendering's mutable inputs separate")
	registry := newTestDataModelRegistry(t, t.TempDir())

	for _, test := range []struct {
		name       string
		wantFamily lpxv1alpha1.TargetFamily
		wantMode   lpxv1alpha1.WorkloadMode
	}{
		{"node-local-v2-lpu-only", lpxv1alpha1.TargetFamilyXt8888, lpxv1alpha1.WorkloadModeV2LPUOnly},
		{"node-local-v2-hybrid", lpxv1alpha1.TargetFamilyXt8888, lpxv1alpha1.WorkloadModeV2StrictHybrid},
		{"node-local-v2-specdecode", lpxv1alpha1.TargetFamilyXt8888, lpxv1alpha1.WorkloadModeV2LPUOnly},
		{"node-local-v3-hx-lpu-only", lpxv1alpha1.TargetFamilyHx16x8x2x3, lpxv1alpha1.WorkloadModeV3HxLPUOnly},
		{"node-local-v3-hx-hybrid", lpxv1alpha1.TargetFamilyHx16x8x2x3, lpxv1alpha1.WorkloadModeV3HxStrictHybrid},
		{"node-local-v3-hx-specdecode", lpxv1alpha1.TargetFamilyHx16x8x2x3, lpxv1alpha1.WorkloadModeV3HxLPUOnly},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the authored engine and its exact child without changing either input")
			fixture := test.name
			payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/" + fixture + ".input.yaml")
			require.NoError(t, err)
			source := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, yaml.Unmarshal(payload, source))
			source.UID, source.Generation = "source-dgd-uid", 7
			source.Labels = map[string]string{
				"priorityClassName": "inference", "kai.scheduler/preemptibility": "NonPreemptible",
				"project": "test", "unrelated": "ignored",
			}
			metav1.SetMetaDataAnnotation(&source.ObjectMeta, "kai.scheduler/topology", "source-topology")
			metav1.SetMetaDataAnnotation(&source.ObjectMeta, "unrelated", "ignored")
			metav1.SetMetaDataAnnotation(&source.ObjectMeta, commonconsts.KubeAnnotationEnableMetrics, "false")
			source.Spec.Labels = map[string]string{"shared-label": "source"}
			source.Spec.Annotations = map[string]string{
				"shared-annotation": "source", "kai.scheduler/topology": "explicit-pcs-topology",
				"selected-metadata":         "from-dgd",
				lpx.DGDUIDAnnotation:        "stale",
				lpx.DGDGenerationAnnotation: "stale",
			}
			source.Spec.Env = []corev1.EnvVar{{Name: "LPX_DGD_ENV", Value: test.name}}
			for _, component := range lpx.Components(source) {
				component.GlobalDynamoNamespace = true
				component.RuntimeVersionOverride = "1.5.0"
				for _, role := range component.Roles {
					if role.PodTemplate != nil {
						metav1.SetMetaDataAnnotation(&role.PodTemplate.ObjectMeta, "selected-metadata", "from-component")
						metav1.SetMetaDataAnnotation(&role.PodTemplate.ObjectMeta, commonconsts.RestartAnnotation, "2026-09-07T00:00:00Z")
						metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, "shared-label", "from-component")
						metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, commonconsts.KubeLabelDynamoNamespace, "authored-namespace")
					}
				}
			}
			component := lpx.ServingComponent(source)
			component.ModelRef = &v1beta1.ModelReference{Name: "test/model"}
			component.MinAvailable = ptr.To(int32(1))
			hybrid := strings.HasSuffix(fixture, "-hybrid")
			singleXT := test.name == "node-local-v2-lpu-only"
			if hybrid {
				component.Replicas = ptr.To(int32(3))
				component.MinAvailable = ptr.To(int32(3))
				component.ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
			}
			if singleXT {
				t.Log("Keep XT's authored readonly mount at the canonical config path")
				main := &component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers[0]
				configMount := slices.IndexFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.MountPath == "/configs" })
				require.GreaterOrEqual(t, configMount, 0)
				main.VolumeMounts[configMount].ReadOnly = true
			}

			t.Log("Preserve role metadata and replica units through alpha/beta conversion before freezing the child")
			alpha := &v1alpha1.DynamoGraphDeployment{}
			require.NoError(t, alpha.ConvertFrom(source))
			source = &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, alpha.ConvertTo(source))
			component = lpx.ServingComponent(source)
			child := newLPXRenderDeployment(t, source)
			child.Annotations[dynamo.LPXRestartAnnotation] = restartToken
			child.Spec.InputRevision, err = dynamo.LPXInputRevision(source, restartToken)
			require.NoError(t, err)
			source.Generation++
			before, childBefore := source.DeepCopy(), child.DeepCopy()
			selected, err := lpx.ResolveSelectedWorkload(t.Context(), source, registry)
			require.NoError(t, err)
			plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, source)))
			require.NoError(t, err)
			config := &configv1alpha1.OperatorConfiguration{
				Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
			}

			t.Log("Project every model into the expected scheduler family, mode and physical Agent shape")
			require.Equal(t, lpx.BuildFamily(test.wantFamily), selected.BuildFamily())
			projections := selected.ModelProjections()
			require.Len(t, plan.Agents, len(projections))
			projectionIndices := make(map[string]int, len(projections))
			for index, projection := range projections {
				projectionIndices[plan.Agents[index].TemplateName] = index
				request := projection.RequestSpec(plan, "agents")
				require.Equal(t, lpxv1alpha1.ExecutionBackendNodeLocal, request.ExecutionBackend)
				require.NotNil(t, request.NodeLocal)
				require.Equal(t, test.wantFamily, request.TargetFamily)
				require.Equal(t, test.wantMode, request.WorkloadMode)
				require.NotEmpty(t, request.Partitions)
				require.Len(t, request.NodeLocal.PartitionMappings, len(request.Partitions))
				require.Equal(t, hybrid, request.CyborgPodCliqueRef != nil)
				if test.wantFamily == lpxv1alpha1.TargetFamilyHx16x8x2x3 {
					require.Len(t, request.Partitions, 1)
					require.Equal(t, int64(1), request.Partitions[0].CompilerPartitionID)
					require.Equal(t, []int64{16, 1, 1, 1}, *request.Partitions[0].Extent)
					require.Empty(t, request.PropSyncConnectors)
					require.Equal(t, 1, plan.Agents[index].Replicas)
				}
			}

			t.Log("Render twice and require identical resources with no input or shared-metadata mutation")
			first, firstResources, err := renderPodCliqueSet(t.Context(), source, config,
				&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, selected, plan, child)
			require.NoError(t, err)
			second, secondResources, err := renderPodCliqueSet(t.Context(), source, config,
				&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, selected, plan, child)
			require.NoError(t, err)
			require.Equal(t, first, second)
			require.Equal(t, firstResources, secondResources)
			require.Equal(t, "inference", first.Labels["priorityClassName"])
			require.Equal(t, "NonPreemptible", first.Labels["kai.scheduler/preemptibility"])
			require.Equal(t, "test", first.Labels["project"])
			require.Equal(t, "explicit-pcs-topology", first.Annotations["kai.scheduler/topology"])
			require.NotContains(t, first.Annotations, commonconsts.RestartAnnotation)
			require.NotContains(t, first.Labels, "unrelated")
			require.NotContains(t, first.Annotations, "unrelated")
			require.Equal(t, selected.Digest().String(), first.Annotations[lpx.WorkloadDigestAnnotation])

			t.Log("Stamp frozen source identity on every final resource despite a newer ordinary DGD generation")
			require.NotEqual(t, source.UID, child.UID)
			annotationMaps := []map[string]string{first.Annotations}
			for _, clique := range first.Spec.Template.Cliques {
				require.Equal(t, restartToken, clique.Annotations[commonconsts.RestartAnnotation])
				annotationMaps = append(annotationMaps, clique.Annotations)
			}
			for _, group := range first.Spec.Template.PodCliqueScalingGroupConfigs {
				require.NotContains(t, group.Annotations, commonconsts.RestartAnnotation)
				annotationMaps = append(annotationMaps, group.Annotations)
			}
			for _, resource := range firstResources {
				require.NotContains(t, resource.GetAnnotations(), commonconsts.RestartAnnotation)
				annotationMaps = append(annotationMaps, resource.GetAnnotations())
			}
			for _, annotations := range annotationMaps {
				require.Equal(t, string(metav1.GetControllerOf(child).UID), annotations[lpx.DGDUIDAnnotation])
				require.NotContains(t, annotations, lpx.DGDGenerationAnnotation)
				require.Equal(t, string(child.UID), annotations[dynamo.LPXDeploymentUIDAnnotation])
				require.NotContains(t, annotations, "lpx.nvidia.com/deployment-generation")
				require.NotContains(t, annotations, "lpx.nvidia.com/input-revision")
			}

			t.Log("Component and role order preserve the exact child, names and resources")
			equivalent := source.DeepCopy()
			slices.Reverse(equivalent.Spec.Components)
			for index := range equivalent.Spec.Components {
				slices.Reverse(equivalent.Spec.Components[index].Roles)
			}
			equivalentBefore := equivalent.DeepCopy()
			revision, err := dynamo.LPXInputRevision(equivalent, restartToken)
			require.NoError(t, err)
			require.Equal(t, child.Spec.InputRevision, revision)
			equivalentSelected, err := lpx.ResolveSelectedWorkload(t.Context(), equivalent, registry)
			require.NoError(t, err)
			equivalentPlan, err := equivalentSelected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(child))
			require.NoError(t, err)
			require.Equal(t, selected.Digest(), equivalentSelected.Digest())
			require.Equal(t, plan, equivalentPlan)
			equivalentPCS, equivalentResources, err := renderPodCliqueSet(t.Context(), equivalent, config,
				&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, equivalentSelected, equivalentPlan, child)
			require.NoError(t, err)
			require.Equal(t, first, equivalentPCS)
			require.Equal(t, firstResources, equivalentResources)
			require.Equal(t, equivalentBefore, equivalent)

			t.Log("Metadata edits and scale 1, 2, 10, 12, 10 preserve every role template and immutable config version")
			replicaCounts := []int32{1, 2, 10, 12, 10}
			if selected.Pipeline() == lpx.PipelineSpecDecode {
				replicaCounts = []int32{1} // Shared-draft topology supports one target engine.
			}
			for _, replicas := range replicaCounts {
				scaled := source.DeepCopy()
				metav1.SetMetaDataAnnotation(&scaled.ObjectMeta, "unrelated", "changed")
				lpx.ServingComponent(scaled).Replicas = ptr.To(replicas)
				scaledChild := child.DeepCopy()
				scaledChild.Generation++
				scaledChild.Spec.InputRevision, err = dynamo.LPXInputRevision(scaled, restartToken)
				require.NoError(t, err)
				workload, err := lpx.ResolveSelectedWorkload(t.Context(), scaled, registry)
				require.NoError(t, err)
				scaledPlan, err := workload.PlanNodeLocalMaterialization(plan.PodCliqueSetName)
				require.NoError(t, err)
				pcs, resources, err := renderPodCliqueSet(t.Context(), scaled, config,
					&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, workload, scaledPlan, scaledChild)
				require.NoError(t, err)
				require.Equal(t, firstResources, resources)
				require.Equal(t, first.Spec.Template.Cliques, pcs.Spec.Template.Cliques)
				require.Equal(t, replicas, *pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].Replicas)
			}

			t.Log("Connect every model projection to its rendered Agent while preserving shared runtime metadata and replica units")
			require.Len(t, first.Spec.Template.PodCliqueScalingGroupConfigs, 1)
			group := first.Spec.Template.PodCliqueScalingGroupConfigs[0]
			require.Equal(t, component.Replicas, group.Replicas)
			require.Equal(t, component.MinAvailable, group.MinAvailable)
			var conductor, gpuClique *grovev1alpha1.PodCliqueTemplateSpec
			serving := 0
			for index, clique := range first.Spec.Template.Cliques {
				if clique.Labels[dynamo.LPXServingLabel] == commonconsts.KubeLabelValueTrue {
					serving++
					require.NotEqual(t, lpxv1alpha1.PodRoleAgent, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
					require.Equal(t, dynamo.HashModelName("test/model"), clique.Labels[commonconsts.KubeLabelDynamoBaseModelHash])
				} else {
					require.NotContains(t, clique.Labels, commonconsts.KubeLabelDynamoDiscoveryEnabled)
					require.NotContains(t, clique.Labels, commonconsts.KubeLabelDynamoBaseModelHash)
				}
				require.Equal(t, ptr.To(clique.Spec.Replicas), clique.Spec.MinAvailable)
				require.NotEmpty(t, clique.Spec.PodSpec.Containers)
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "LPX_DGD_ENV", Value: test.name})
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: commonconsts.DynamoNamespaceEnvVar, Value: commonconsts.GlobalDynamoNamespace})
				require.Equal(t, commonconsts.GlobalDynamoNamespace, clique.Labels[commonconsts.KubeLabelDynamoNamespace])
				require.Equal(t, "from-component", clique.Labels["shared-label"])
				require.Equal(t, "false", clique.Annotations[commonconsts.KubeAnnotationEnableMetrics])
				require.Equal(t, "kubernetes", clique.Annotations[commonconsts.KubeAnnotationDynamoDiscoveryBackend])
				if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] != lpxv1alpha1.PodRoleCyborgWorker || test.wantFamily == lpxv1alpha1.TargetFamilyXt8888 {
					t.Log("Bind the generated immutable ConfigMap alongside authored volumes and mounts")
					configIndex := slices.IndexFunc(clique.Spec.PodSpec.Volumes, func(volume corev1.Volume) bool { return volume.Name == "config" })
					require.GreaterOrEqual(t, configIndex, 0)
					configVolume := clique.Spec.PodSpec.Volumes[configIndex]
					require.NotNil(t, configVolume.ConfigMap)
					generatedConfig := getResource[*corev1.ConfigMap](t, firstResources, configVolume.ConfigMap.Name)
					require.Equal(t, ptr.To(true), generatedConfig.Immutable)
				}
				switch clique.Annotations[lpxv1alpha1.PodRoleAnnotation] {
				case lpxv1alpha1.PodRoleAgent:
					index, found := projectionIndices[clique.Name]
					require.True(t, found, "unexpected or duplicate Agent %q", clique.Name)
					require.Equal(t, projections[index].Model(), clique.Annotations[lpxv1alpha1.PodModelAnnotation])
					require.Equal(t, projections[index].Digest().String(), clique.Annotations[lpx.WorkloadDigestAnnotation])
					require.Equal(t, int32(plan.Agents[index].Replicas), clique.Spec.Replicas)
					if singleXT {
						require.Contains(t, clique.Spec.PodSpec.Containers[0].VolumeMounts,
							corev1.VolumeMount{Name: "config", MountPath: "/configs", ReadOnly: true})
					}
					delete(projectionIndices, clique.Name)
				case lpxv1alpha1.PodRoleConductor:
					require.Nil(t, conductor)
					conductor = clique
				case lpxv1alpha1.PodRoleCyborgWorker:
					require.Nil(t, gpuClique)
					gpuClique = clique
				}

				t.Log("Mutate this rendered role without changing the second rendering or authored input")
				clique.Spec.PodSpec.Containers[0].Image = "rendered:changed"
				clique.Labels["shared-label"] = "rendered-role"
				clique.Annotations["shared-annotation"] = "rendered-role"
				require.NotEqual(t, clique.Spec.PodSpec.Containers[0].Image, second.Spec.Template.Cliques[index].Spec.PodSpec.Containers[0].Image)
				require.Equal(t, "from-component", second.Spec.Template.Cliques[index].Labels["shared-label"])
				require.Equal(t, "source", second.Spec.Template.Cliques[index].Annotations["shared-annotation"])
			}
			require.Empty(t, projectionIndices)
			require.Equal(t, 1, serving)
			if hybrid {
				require.Nil(t, conductor)
				require.NotNil(t, gpuClique)
				require.Equal(t, int32(2), gpuClique.Spec.Replicas)
				require.Equal(t, selected.Digest().String(), gpuClique.Annotations[lpx.WorkloadDigestAnnotation])
				require.Contains(t, gpuClique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "DYN_HEALTH_CHECK_ENABLED", Value: "true"})
			} else {
				require.NotNil(t, conductor)
				require.Equal(t, "from-component", conductor.Annotations["selected-metadata"])
				require.Equal(t, selected.Digest().String(), conductor.Annotations[lpx.WorkloadDigestAnnotation])
				require.Nil(t, gpuClique)
			}
			first.Labels["shared-label"] = "rendered"
			first.Annotations["shared-annotation"] = "rendered"
			require.Equal(t, "source", second.Labels["shared-label"])
			require.Equal(t, "source", second.Annotations["shared-annotation"])
			require.Equal(t, before, source)
			require.Equal(t, childBefore, child)
		})
	}
}

func TestLPXSpecDecodeConductorTemplate(t *testing.T) {
	t.Log("Give the speculative components distinct authored Agent and conductor commands")
	registry := newTestDataModelRegistry(t, t.TempDir())
	payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/node-local-v3-hx-specdecode.input.yaml")
	require.NoError(t, err)
	source := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, source))
	for _, component := range lpx.Components(source) {
		agent := component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
		agent.Spec.Containers[0].Image = component.ComponentName + "-runtime"
		agent.Spec.Containers[0].Command = []string{"/bin/quasar-entrypoint"}
		agent.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "AGENT_ONLY", Value: "kept"}}
	}
	target := lpx.ServingComponent(source)
	template := target.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	template.Spec.Containers[0].Image = "independent-conductor-runtime"
	template.Spec.Containers[0].Command = []string{"/bin/nova"}
	template.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "CONDUCTOR_ONLY", Value: "kept"}}
	template.Labels = map[string]string{"owner": "explicit-conductor"}
	template.Spec.NodeSelector = map[string]string{"runtime-role": "conductor"}
	template.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
			MatchExpressions: []corev1.NodeSelectorRequirement{{Key: "pool", Operator: corev1.NodeSelectorOpIn, Values: []string{"conductor"}}},
		}}},
	}}
	before := source.DeepCopy()
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), source, registry)
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, source)))
	require.NoError(t, err)

	t.Log("Render one shared conductor without changing either component's Agent template")
	pcs, _, err := renderPodCliqueSet(t.Context(), source,
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, selected, plan, newLPXRenderDeployment(t, source))
	require.NoError(t, err)

	t.Log("Keep Nova separate from Quasar and preserve each role's metadata and storage")
	require.Len(t, pcs.Spec.Template.Cliques, len(plan.Agents)+1)
	conductors := 0
	for _, clique := range pcs.Spec.Template.Cliques {
		container := clique.Spec.PodSpec.Containers[0]
		for _, mount := range container.VolumeMounts {
			require.True(t, slices.ContainsFunc(clique.Spec.PodSpec.Volumes, func(volume corev1.Volume) bool {
				return volume.Name == mount.Name
			}), "%s mount %s requires a volume", clique.Name, mount.Name)
		}
		if clique.Name == plan.ConductorTemplate {
			conductors++
			require.Equal(t, "independent-conductor-runtime", container.Image)
			require.Equal(t, []string{"/bin/nova"}, container.Command)
			configFlag := slices.Index(container.Args, "--datacenter-config-filepath")
			require.GreaterOrEqual(t, configFlag, 0)
			require.Equal(t, "/configs/datacenter.toml", container.Args[configFlag+1])
			require.Equal(t, "explicit-conductor", clique.Labels["owner"])
			require.Equal(t, template.Spec.NodeSelector, clique.Spec.PodSpec.NodeSelector)
			require.Equal(t, template.Spec.Affinity, clique.Spec.PodSpec.Affinity)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
		} else {
			component := clique.Labels[commonconsts.KubeLabelDynamoComponent]
			require.Equal(t, component+"-runtime", container.Image)
			require.Equal(t, []string{"/bin/quasar-entrypoint"}, container.Command)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
		}
	}
	require.Equal(t, 1, conductors)
	require.Equal(t, before, source)
}

func newTestDataModelRegistry(t *testing.T, registryRoot string) lpx.ModelRegistry {
	t.Helper()

	v2Builds := map[string]testV2GraphManifestFixture{
		"node-local-v2-connected": {
			topology:              "URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA",
			partitionCount:        2,
			numChips:              16,
			devicesPerNode:        8,
			selectedPropSyncChain: []uint32{0, 1},
		},
		"node-local-v2-connected-lpx": {
			topology:              "URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA",
			partitionCount:        2,
			numChips:              16,
			devicesPerNode:        8,
			selectedPropSyncChain: []uint32{0, 1},
			compilationMode:       manifestcapnpv2.CompilationMode_lpx,
			nonLPUDeviceTypes:     []manifestcapnpv2.DeviceType{manifestcapnpv2.DeviceType_cuda},
		},
		"node-local-v2-cpu-embeddings": {
			topology:                   "URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA",
			partitionCount:             3,
			numChips:                   16,
			devicesPerNode:             8,
			selectedPropSyncChain:      []uint32{0, 1, 2},
			compilationMode:            manifestcapnpv2.CompilationMode_lpx,
			nonLPUDeviceTypes:          []manifestcapnpv2.DeviceType{manifestcapnpv2.DeviceType_cuda},
			standaloneTokenEmbeddings:  true,
			supportsCPUEmbeddings:      true,
			runtimeTokenEmbeddingsPath: "runtime/text_embeddings.npz",
		},
		"llama3_2-1b-lpu-gpu-v2/build_0m851219t7py3mp8x1j5rg9j8c": {
			topology:          "URSA_V2_1__Q8__8C__G_96_25__KP_FEC__GHZ_1_0__DRACO_V1_1__G_106",
			partitionCount:    17,
			numChips:          8,
			devicesPerNode:    8,
			compilationMode:   manifestcapnpv2.CompilationMode_lpx,
			nonLPUDeviceTypes: []manifestcapnpv2.DeviceType{manifestcapnpv2.DeviceType_cuda},
		},
		"llama3_2-1b-lpu-v2/build_0m851219t7py3mp8x1j5rg9j8c": {
			topology:       "URSA_V2_1__Q8__8C__G_96_25__KP_FEC__GHZ_1_0__DRACO_V1_1__G_106",
			partitionCount: 17,
			numChips:       8,
			devicesPerNode: 8,
		},
	}
	for buildID, fixture := range v2Builds {
		writeTestGraphBuild(t, registryRoot, buildID, testV2GraphManifestCapnp(t, buildID, fixture))
	}

	v3Builds := map[string]testV3GraphManifestFixture{
		"node-local-v3-hx-connected": {},
		"node-local-v3-hx-connected-lpx": {
			compilationMode:   manifestcapnpv2.CompilationMode_lpx,
			nonLPUDeviceTypes: []manifestcapnpv2.DeviceType{manifestcapnpv2.DeviceType_cuda},
		},
		"node-local-v3-hx-sd-draft":  {},
		"node-local-v3-hx-sd-target": {},
	}
	for buildID, fixture := range v3Builds {
		writeTestGraphBuild(t, registryRoot, buildID, testV3GraphManifestCapnp(t, buildID, fixture))
	}

	registry, err := lpx.NewModelRegistry(registryRoot, nil)
	require.NoError(t, err)
	return registry
}

type testV2GraphManifestFixture struct {
	topology                   string
	partitionCount             int
	numChips                   uint32
	devicesPerNode             uint32
	selectedPropSyncChain      []uint32
	compilationMode            manifestcapnpv2.CompilationMode
	nonLPUDeviceTypes          []manifestcapnpv2.DeviceType
	standaloneTokenEmbeddings  bool
	supportsCPUEmbeddings      bool
	runtimeTokenEmbeddingsPath string
}

func writeTestGraphBuild(t *testing.T, registryRoot, buildID string, manifest []byte) {
	t.Helper()

	buildDir := filepath.Join(registryRoot, buildID)
	require.NoError(t, os.MkdirAll(buildDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.v2.capnp.bin"), manifest, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.json"), []byte(`{}`), 0o600))
}

func newTestGraphManifest(t *testing.T) (*capnp.Message, manifestcapnpv2.Manifest) {
	t.Helper()

	t.Log("Create the shared manifest header and tokenizer model")
	message, segment := capnp.NewSingleSegmentMessage(nil)
	manifest, err := manifestcapnpv2.NewRootManifest(segment)
	require.NoError(t, err)
	manifest.SetContractRevision(manifestcapnpv2.CurrentContractRevision)
	model, err := manifest.NewModel()
	require.NoError(t, err)
	tokenizer, err := model.NewTokenizer()
	require.NoError(t, err)
	require.NoError(t, tokenizer.SetPath("tokenizer"))
	stopTokens, err := tokenizer.NewStopTokens(1)
	require.NoError(t, err)
	stopTokens.Set(0, 1)
	return message, manifest
}

func populateTestGraphBuild(t *testing.T, manifest manifestcapnpv2.Manifest, wireBuildID string) {
	t.Helper()

	t.Log("Populate the compiler-owned build identity used only as opaque provenance")
	build, err := capnp.NewStruct(manifest.Segment(), capnp.ObjectSize{DataSize: 8, PointerCount: 12})
	require.NoError(t, err)
	require.NoError(t, build.SetText(11, wireBuildID))
	require.NoError(t, manifest.SetReserved3(build.ToPtr()))
}

func newTestGraphProgram(
	t *testing.T,
	manifest manifestcapnpv2.Manifest,
	mode manifestcapnpv2.CompilationMode,
	numLPUNodes uint32,
	sequenceLength uint32,
) (manifestcapnpv2.DeploymentInfo, manifestcapnpv2.ProgramConfig) {
	t.Helper()

	t.Log("Allocate the deployment, program, and runtime I/O contract shared by graph fixtures")
	deployment, err := manifest.NewDeployment()
	require.NoError(t, err)
	deployment.SetCompilationMode(mode)
	deployment.SetNumLpuNodes(numLPUNodes)
	program, err := deployment.NewProgram()
	require.NoError(t, err)
	program.SetBatchSize(1)
	program.SetSequenceLength(sequenceLength)
	program.SetInputSize(1)
	program.SetOutputSize(1)
	runtimeIO, err := deployment.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(0)
	runtimeIO.SetReserved1(1)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	return deployment, program
}

func testV2GraphManifestCapnp(t *testing.T, buildID string, fixture testV2GraphManifestFixture) []byte {
	t.Helper()

	message, manifest := newTestGraphManifest(t)
	populateTestGraphBuild(t, manifest, filepath.Base(buildID))

	deployment, program := newTestGraphProgram(t, manifest, fixture.compilationMode, uint32(fixture.partitionCount)*fixture.numChips/fixture.devicesPerNode, 8192)
	if len(fixture.selectedPropSyncChain) != 0 {
		chains, err := deployment.NewSelectedPropSyncChains(1)
		require.NoError(t, err)
		partitionIDs, err := chains.At(0).NewPartitionIds(int32(len(fixture.selectedPropSyncChain)))
		require.NoError(t, err)
		for index, partitionID := range fixture.selectedPropSyncChain {
			partitionIDs.Set(index, partitionID)
		}
	}
	program.SetNumKvCaches(1)
	program.SetNumBatchSplitDivisions(1)
	program.SetSupportsCpuEmbeddings(fixture.supportsCPUEmbeddings)
	program.SetStandaloneTokenEmbeddings(fixture.standaloneTokenEmbeddings)

	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	if fixture.runtimeTokenEmbeddingsPath != "" {
		runtimeAssets, err := artifacts.NewRuntimeAssets()
		require.NoError(t, err)
		require.NoError(t, runtimeAssets.SetTokenEmbeddingsPath(fixture.runtimeTokenEmbeddingsPath))
	}
	partitions, err := artifacts.NewPartitions(int32(fixture.partitionCount + len(fixture.nonLPUDeviceTypes)))
	require.NoError(t, err)
	for index := range fixture.partitionCount {
		partition := partitions.At(index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
		ref.SetPartitionId(uint32(index))
		detail, err := partition.Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", index)))
		require.NoError(t, detail.SetTopology(fixture.topology))
		detail.SetNumChips(fixture.numChips)
		detail.SetDevicesPerNode(fixture.devicesPerNode)
	}
	for index, deviceType := range fixture.nonLPUDeviceTypes {
		partition := partitions.At(fixture.partitionCount + index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(deviceType)
		ref.SetPartitionId(uint32(fixture.partitionCount + index))
		switch deviceType {
		case manifestcapnpv2.DeviceType_cuda:
			_, err = partition.Detail().NewCuda()
		case manifestcapnpv2.DeviceType_cpu:
			_, err = partition.Detail().NewCpu()
		default:
			t.Fatalf("unsupported non-LPU test device type %q", deviceType)
		}
		require.NoError(t, err)
	}

	data, err := message.Marshal()
	require.NoError(t, err)
	return data
}

func mustPlanSelectedLPX(t *testing.T, dgd *v1beta1.DynamoGraphDeployment, selected *lpx.SelectedWorkload) *lpx.MaterializationPlan {
	t.Helper()
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)
	return plan
}

type testV3GraphManifestFixture struct {
	compilationMode   manifestcapnpv2.CompilationMode
	nonLPUDeviceTypes []manifestcapnpv2.DeviceType
}

func testV3GraphManifestCapnp(t *testing.T, buildID string, fixture testV3GraphManifestFixture) []byte {
	t.Helper()

	message, manifest := newTestGraphManifest(t)
	populateTestGraphBuild(t, manifest, buildID)

	_, program := newTestGraphProgram(t, manifest, fixture.compilationMode, 2, 8192)
	program.SetNumKvCaches(1)

	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitions, err := artifacts.NewPartitions(int32(1 + len(fixture.nonLPUDeviceTypes)))
	require.NoError(t, err)
	partition := partitions.At(0)
	ref, err := partition.NewPartition()
	require.NoError(t, err)
	ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
	ref.SetPartitionId(1)
	detail, err := partition.Detail().NewLpu()
	require.NoError(t, err)
	require.NoError(t, detail.SetPath("part-1"))
	require.NoError(t, detail.SetTopology("opaque-v3-topology"))
	detail.SetNumChips(16)
	detail.SetDevicesPerNode(16)
	for offset, deviceType := range fixture.nonLPUDeviceTypes {
		partition := partitions.At(1 + offset)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(deviceType)
		ref.SetPartitionId(uint32(2 + offset))
		switch deviceType {
		case manifestcapnpv2.DeviceType_cuda:
			_, err = partition.Detail().NewCuda()
		case manifestcapnpv2.DeviceType_cpu:
			_, err = partition.Detail().NewCpu()
		default:
			t.Fatalf("unsupported non-LPU test device type %q", deviceType)
		}
		require.NoError(t, err)
	}

	data, err := message.Marshal()
	require.NoError(t, err)
	return data
}

func TestSingleV2ManifestDefaultsDriveConfigAndHash(t *testing.T) {
	controllerConfig := &configv1alpha1.OperatorConfiguration{
		Orchestrators: configv1alpha1.OrchestratorConfiguration{
			Grove: configv1alpha1.GroveConfiguration{TerminationDelay: metav1.Duration{Duration: 15 * time.Minute}},
		},
	}
	const buildID = "single-v2-manifest-defaults/build_manifest_defaults"
	registryRoot := t.TempDir()
	writeTestGraphBuild(t, registryRoot, buildID, testGbuildManifestCapnp(t, buildID))
	registry, err := lpx.NewModelRegistry(registryRoot, nil)
	require.NoError(t, err)

	type renderedConfig struct {
		modelConfig string
		hash        string
	}
	t.Log("Render manifest-derived, equivalent explicit, and overridden runtime settings")
	configs := make([]renderedConfig, 0, 3)
	for _, settingsJSON := range []string{"", `{"batch_size":1,"swa":{"chunked":false}}`, `{"sequence_length":65536}`} {
		payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/single_v2.input.yaml")
		require.NoError(t, err)

		var deployment v1beta1.DynamoGraphDeployment
		require.NoError(t, yaml.Unmarshal(payload, &deployment))
		require.Len(t, deployment.Spec.Components, 1)
		require.NotNil(t, deployment.Spec.Components[0].LPX)
		require.Len(t, deployment.Spec.Components[0].Roles, 2)
		deployment.Spec.Components[0].LPX = &v1beta1.LPXConfig{BuildID: buildID}
		if settingsJSON != "" {
			deployment.Spec.Components[0].LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(settingsJSON)}
		}

		selected, err := lpx.ResolveSelectedWorkload(t.Context(), &deployment, registry)
		require.NoError(t, err)
		child := newLPXRenderDeployment(t, &deployment)
		podCliqueSet, resources, err := renderPodCliqueSet(
			t.Context(),
			&deployment,
			controllerConfig,
			&controller_common.RuntimeConfig{},
			nil, nil,
			selected,
			mustPlanSelectedLPX(t, &deployment, selected),
			child,
		)
		require.NoError(t, err)

		var hash string
		for _, clique := range podCliqueSet.Spec.Template.Cliques {
			if value := clique.Annotations[commonconsts.AnnotationExtraResourcesHash]; value != "" {
				hash = value
				break
			}
		}
		require.NotEmpty(t, hash)
		configMap := getResource[*corev1.ConfigMap](t, resources, lpx.LPUConfigMapName(dynamo.PCSNameForLPX(child), hash))
		configs = append(configs, renderedConfig{modelConfig: configMap.Data["model_config.toml"], hash: hash})
	}

	t.Log("Preserve derived config and hash until an explicit value changes the output")
	derived, explicitButEqual, overridden := configs[0], configs[1], configs[2]

	buildPath := filepath.Join(registryRoot, buildID)
	require.Equal(t, fmt.Sprintf(`type = 'Single'

[iop]
batch_size = 1
cpu_embeddings = true
dkvc = true
input_size = 1
model_path = '%s'
num_dkvc_blocks = 256
num_kv_caches = 1
num_layers = 24
output_size = 1
prop_sync = false
sequence_length = 131072
stop_tokens = [200002, 199999, 200012]
tokenizer_path = '%s'
vocab_size = 201088

[iop.swa]
chunked = false
num_swa_dkvc_blocks = 1
swa_ctx_len = 128
swa_num_users = 8
swa_padding_len = 0

[scheduler]

[setup]
resolved_partitions_dir = '/configs'
setup_ops_format = 'agent_v2'
`, buildPath, filepath.Join(buildPath, "tokenizer")), derived.modelConfig)
	require.Equal(t, derived.modelConfig, explicitButEqual.modelConfig)
	require.Equal(t, derived.hash, explicitButEqual.hash)
	require.NotEqual(t, derived.modelConfig, overridden.modelConfig)
	require.NotEqual(t, derived.hash, overridden.hash)
}

func testGbuildManifestCapnp(t *testing.T, registryDir string) []byte {
	t.Helper()

	const topology = "URSA_V2_1__Q8__8C__G_96_25__KP_FEC__GHZ_1_0__DRACO_V1_1__G_106"

	msg, seg := capnp.NewSingleSegmentMessage(nil)
	manifest, err := manifestcapnpv2.NewRootManifest(seg)
	require.NoError(t, err)
	manifest.SetContractRevision(manifestcapnpv2.CurrentContractRevision)
	populateTestGraphBuild(t, manifest, filepath.Base(registryDir))

	model, err := manifest.NewModel()
	require.NoError(t, err)
	arch, err := model.NewArch()
	require.NoError(t, err)
	arch.SetNLayers(24)
	arch.SetVocabSize(201088)
	swa, err := arch.NewSwa()
	require.NoError(t, err)
	swa.SetMaxSeqLen(128)
	swa.SetPadding(0)
	swaNumUsers, err := swa.NewNumUsers()
	require.NoError(t, err)
	swaNumUsers.SetValue(8)
	tokenizer, err := model.NewTokenizer()
	require.NoError(t, err)
	require.NoError(t, tokenizer.SetPath("tokenizer"))
	stopTokens, err := tokenizer.NewStopTokens(3)
	require.NoError(t, err)
	stopTokens.Set(0, 200002)
	stopTokens.Set(1, 199999)
	stopTokens.Set(2, 200012)

	deployment, program := newTestGraphProgram(t, manifest, manifestcapnpv2.CompilationMode_lpuOnly, 1, 131072)
	_, err = deployment.NewSelectedPropSyncChains(0)
	require.NoError(t, err)
	dkvc, err := program.NewDkvc()
	require.NoError(t, err)
	dkvc.SetNumBlocksPerKvCache(256)
	program.SetNumKvCaches(1)
	program.SetNumBatchSplitDivisions(1)
	program.SetBatchFolding(false)
	program.SetSupportsCpuEmbeddings(true)
	program.SetSwaChunked(false)
	program.SetNumSwaDkvcBlocks(1)

	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitions, err := artifacts.NewPartitions(1)
	require.NoError(t, err)
	partition := partitions.At(0)
	ref, err := partition.NewPartition()
	require.NoError(t, err)
	ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
	ref.SetPartitionId(0)
	lpuDetail, err := partition.Detail().NewLpu()
	require.NoError(t, err)
	require.NoError(t, lpuDetail.SetPath("assemble/part-0"))
	require.NoError(t, lpuDetail.SetTopology(topology))
	lpuDetail.SetNumChips(8)
	lpuDetail.SetDevicesPerNode(8)

	data, err := msg.Marshal()
	require.NoError(t, err)
	return data
}

// Update this golden with:
// go test ./internal/controller/lpx -run TestGenerateGrovePodCliqueSet_FromDGDYaml -count=1 -args -update
func TestGenerateGrovePodCliqueSet_FromDGDYaml(t *testing.T) {
	controllerConfig := &configv1alpha1.OperatorConfiguration{
		Infrastructure: configv1alpha1.InfrastructureConfiguration{
			ETCDAddress: "etcd-address",
			NATSAddress: "nats-address",
		},
		Orchestrators: configv1alpha1.OrchestratorConfiguration{
			Grove: configv1alpha1.GroveConfiguration{
				TerminationDelay: metav1.Duration{Duration: 15 * time.Minute},
			},
		},
		LPX: configv1alpha1.LPXConfiguration{Enabled: true},
	}

	registryRoot := t.TempDir()
	testdataModelRegistry := newTestDataModelRegistry(t, registryRoot)
	kubeClient := newTestLPXClient(t)
	runtimeConfig := &controller_common.RuntimeConfig{}

	tests := []string{
		"from_dgd_yaml/lpx_v2_vllm",
		"from_dgd_yaml/node-local-v2-lpu-only",
		"from_dgd_yaml/node-local-v2-hybrid",
		"from_dgd_yaml/node-local-v2-specdecode",
		"from_dgd_yaml/node-local-v3-hx-lpu-only",
		"from_dgd_yaml/node-local-v3-hx-specdecode",
		"from_dgd_yaml/node-local-v3-hx-hybrid",
		"from_dgd_yaml/single_v2",
	}

	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			t.Log("Load the authored input before selection")
			b, err := os.ReadFile("../../dynamo/lpx/testdata/" + name + ".input.yaml")
			require.NoError(t, err)

			var dynamoDeployment v1beta1.DynamoGraphDeployment
			require.NoError(t, yaml.Unmarshal(b, &dynamoDeployment))
			if dynamoDeployment.UID == "" {
				dynamoDeployment.UID = "11111111-1111-4111-8111-111111111111"
			}
			if dynamoDeployment.Generation == 0 {
				dynamoDeployment.Generation = 1
			}
			selected, err := lpx.ResolveSelectedWorkload(t.Context(), &dynamoDeployment, testdataModelRegistry)
			require.NoError(t, err)
			plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, &dynamoDeployment)))
			require.NoError(t, err)
			t.Log("Render all LPX roles with the separate LPX PCS identity")
			got, extraResources, err := renderPodCliqueSet(
				t.Context(), &dynamoDeployment, controllerConfig, runtimeConfig,
				kubeClient, nil, selected,
				plan,
				newLPXRenderDeployment(t, &dynamoDeployment),
			)
			require.NoError(t, err)
			for _, clique := range got.Spec.Template.Cliques {
				component := dynamoDeployment.GetComponentByName(clique.Labels[commonconsts.KubeLabelDynamoComponent])
				require.NotNil(t, component)
				require.True(t, component.IsLPX())
				require.Equal(t, component.ComponentName, clique.Labels[lpx.StageLabel])
			}

			t.Log("Render conventional components independently from their ordinary-only input")
			ordinary := dynamoDeployment.DeepCopy()
			ordinary.Spec.Components = slices.DeleteFunc(ordinary.Spec.Components, func(component v1beta1.DynamoComponentDeploymentSharedSpec) bool {
				return component.IsLPX()
			})
			normal, err := dynamo.GenerateGrovePodCliqueSet(
				t.Context(), ordinary, controllerConfig, runtimeConfig,
				kubeClient, nil, nil, nil, nil,
			)
			require.NoError(t, err)
			require.NotEqual(t, normal.Name, got.Name)
			for _, clique := range normal.Spec.Template.Cliques {
				require.NotEqual(t, selected.LPXComponentName(), clique.Labels[commonconsts.KubeLabelDynamoComponent])
			}
			podCliqueSets := []*grovev1alpha1.PodCliqueSet{got}
			if len(normal.Spec.Template.Cliques) > 0 {
				podCliqueSets = append(podCliqueSets, normal)
			}

			for _, pcs := range podCliqueSets {
				sort.Slice(pcs.Spec.Template.Cliques, func(i, j int) bool {
					return pcs.Spec.Template.Cliques[i].Name < pcs.Spec.Template.Cliques[j].Name
				})
				for _, clique := range pcs.Spec.Template.Cliques {
					for i := range clique.Spec.PodSpec.Containers {
						slices.SortFunc(clique.Spec.PodSpec.Containers[i].Env, func(a, b corev1.EnvVar) int {
							return strings.Compare(a.Name, b.Name)
						})
					}
				}
				extraResources = append(extraResources, pcs)
			}

			var out []string
			for _, resource := range extraResources {
				b, err := yaml.Marshal(resource)
				if err != nil {
					t.Errorf("Marshal() error = %v", err)
				}
				out = append(out, string(b))
			}

			t.Log("Normalize temporary paths and their content hashes only at the golden boundary; incorrect hash annotations still fail")
			replacements := []string{registryRoot, "/testdata"}
			for _, resource := range extraResources {
				config, ok := resource.(*corev1.ConfigMap)
				if !ok {
					continue
				}
				actualHash := lpx.LPUConfigMapHash(config)
				normalized := config.DeepCopy()
				for key, value := range normalized.Data {
					normalized.Data[key] = strings.ReplaceAll(value, registryRoot, "/testdata")
				}
				normalizedHash := lpx.LPUConfigMapHash(normalized)
				replacements = append(replacements, actualHash, normalizedHash, actualHash[:16], normalizedHash[:16])
			}
			goldenPath, err := filepath.Abs("../../dynamo/lpx/testdata/" + name + ".yaml")
			require.NoError(t, err)
			const header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n" +
				"# SPDX-License-Identifier: Apache-2.0\n\n"
			golden.Assert(t, header+strings.NewReplacer(replacements...).Replace(strings.Join(out, "---\n")), goldenPath)
		})
	}
}

func TestGenerateGrovePodCliqueSet_ImplicitV2HybridPreservesAgentRuntime(t *testing.T) {
	t.Log("Load the pre-scheduler V2 hybrid DGD shape without scheduler intent and with an authored runtime command")
	payload, err := os.ReadFile(filepath.Join("../../dynamo/lpx/testdata", "from_dgd_yaml", "node-local-v2-hybrid.input.yaml"))
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	delete(dgd.Annotations, commonconsts.KubeAnnotationLPXSchedulerBackend)
	dgd.Spec.Scheduling = nil
	lpxComponent := dgd.GetComponentByName("lpu")
	require.NotNil(t, lpxComponent)
	require.NotNil(t, lpxComponent.LPX)
	lpxComponent.LPX.BuildID = "node-local-v2-cpu-embeddings"
	lpxComponent.ModelRef = &v1beta1.ModelReference{Name: "test/model"}

	t.Log("Customize the independent Cyborg batch while keeping its sidecar separate")
	cyborgRole := lpxComponent.ComponentRole(v1beta1.ComponentRoleLPXConductor)
	require.NotNil(t, cyborgRole)
	authoredCyborgMain := &cyborgRole.PodTemplate.Spec.Containers[0]
	require.Equal(t, commonconsts.MainContainerName, authoredCyborgMain.Name)
	authoredCyborgMain.Env = append(
		authoredCyborgMain.Env,
		corev1.EnvVar{Name: lpx.CyborgBatchSizeEnv, Value: "2"},
		corev1.EnvVar{Name: "USER_BATCH_REF", Value: "$(CYBORG_BATCH_SIZE)"},
	)

	t.Log("Keep a sidecar named cyborg separate from the main runtime container")
	sidecar := corev1.Container{Name: "cyborg", Image: "sidecar", Args: []string{"observe"}}
	cyborgRole.PodTemplate.Spec.Containers = append(cyborgRole.PodTemplate.Spec.Containers, sidecar)

	t.Log("Customize Agent metadata and placement with an image ENTRYPOINT and authored arguments")
	agentTemplate := lpxComponent.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
	agentMain := &agentTemplate.Spec.Containers[0]
	require.NotEmpty(t, agentMain.Command)
	require.NotEmpty(t, agentMain.Args)
	agentMain.Command = nil
	agentMain.Args = []string{"--instance-model-name", "test/model", "--agent-env-vars", "PIPELINE_ENV=kept"}
	agentMain.Env = append(agentMain.Env,
		corev1.EnvVar{Name: "CONTAINER_NAME", Value: commonconsts.MainContainerName},
		corev1.EnvVar{Name: "MAIN_CPU", ValueFrom: &corev1.EnvVarSource{ResourceFieldRef: &corev1.ResourceFieldSelector{
			ContainerName: commonconsts.MainContainerName, Resource: "requests.cpu",
		}}},
	)
	agentTemplate.Spec.Tolerations = []corev1.Toleration{
		{Key: "cluster.example/custom", Operator: corev1.TolerationOpExists},
		{Key: "lpu.nvidia.com/node", Operator: corev1.TolerationOpExists},
	}
	agentMain.VolumeMounts = append(agentMain.VolumeMounts,
		corev1.VolumeMount{Name: "config", MountPath: "/custom", ReadOnly: true},
	)
	customConfigVolume := corev1.Volume{
		Name: "config",
		VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
			LocalObjectReference: corev1.LocalObjectReference{Name: "authored-runtime-config"},
			DefaultMode:          ptr.To[int32](0o440),
		}},
	}
	agentTemplate.Spec.Volumes = append(agentTemplate.Spec.Volumes, customConfigVolume)

	t.Log("Project the scheduler-defaulted workload and preserve its conductorless LPU runtime")
	registry := newTestDataModelRegistry(t, t.TempDir())
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, registry)
	require.NoError(t, err)
	projections := selected.ModelProjections()
	require.Len(t, projections, 1)
	projection := projections[0]
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)
	require.Empty(t, plan.ConductorTemplate)
	require.Len(t, plan.Agents, 1)
	require.Equal(t, 6, plan.Agents[0].Replicas)
	request := projection.RequestSpec(plan, "test")
	require.Len(t, request.Partitions, 3)
	for index, partition := range request.Partitions {
		require.Equal(t, int64(index), partition.CompilerPartitionID)
	}

	controllerConfig := &configv1alpha1.OperatorConfiguration{
		Infrastructure: configv1alpha1.InfrastructureConfiguration{ETCDAddress: "etcd", NATSAddress: "nats"},
		Discovery:      configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
		LPX:            configv1alpha1.LPXConfiguration{Enabled: true},
	}
	pcs, extraResources, err := renderPodCliqueSet(
		t.Context(), dgd, controllerConfig, &controller_common.RuntimeConfig{},
		newTestLPXClient(t), nil, selected, plan, newLPXRenderDeployment(t, dgd),
	)
	require.NoError(t, err)

	t.Log("Verify LPX places the worker while retaining the authored process contract")
	var agent, cyborg *grovev1alpha1.PodCliqueTemplateSpec
	serving := 0
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Labels[dynamo.LPXServingLabel] == commonconsts.KubeLabelValueTrue {
			serving++
			require.Equal(t, lpxv1alpha1.PodRoleCyborgWorker, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
			require.Equal(t, dynamo.HashModelName("test/model"), clique.Labels[commonconsts.KubeLabelDynamoBaseModelHash])
		} else {
			require.NotContains(t, clique.Labels, commonconsts.KubeLabelDynamoDiscoveryEnabled)
			require.NotContains(t, clique.Labels, commonconsts.KubeLabelDynamoBaseModelHash)
		}
		switch clique.Annotations[lpxv1alpha1.PodRoleAnnotation] {
		case lpxv1alpha1.PodRoleConductor:
			t.Fatalf("implicit V2 hybrid runtime unexpectedly rendered conductor %q", clique.Name)
		case lpxv1alpha1.PodRoleAgent:
			agent = clique
		case lpxv1alpha1.PodRoleCyborgWorker:
			cyborg = clique
		}
	}
	require.NotNil(t, agent)
	require.NotNil(t, cyborg)
	require.Equal(t, 1, serving)
	require.Equal(t, lpx.SchedulerName, agent.Spec.PodSpec.SchedulerName)
	require.Equal(t, int32(plan.Agents[0].Replicas), agent.Spec.Replicas)
	require.Contains(t, cyborg.Spec.StartsAfter, agent.Name)
	require.NotContains(t, cyborg.Spec.StartsAfter, "cond")

	main := agent.Spec.PodSpec.Containers[0]
	require.Equal(t, "agent", main.Name)
	require.Nil(t, main.Command)
	require.Equal(t, agentMain.Args, main.Args)
	require.NotNil(t, main.SecurityContext)
	require.True(t, *main.SecurityContext.Privileged)
	require.False(t, main.Stdin)
	require.False(t, main.TTY)
	require.NotNil(t, main.StartupProbe)
	require.NotNil(t, main.ReadinessProbe)
	runtimeEnv := make(map[string]string, len(main.Env))
	for _, variable := range main.Env {
		runtimeEnv[variable.Name] = variable.Value
	}
	require.Equal(t, "agent", runtimeEnv["CONTAINER_NAME"])
	require.Equal(t, "agent", envValueSource(main.Env, "MAIN_CPU").ResourceFieldRef.ContainerName)
	require.Equal(t, "1024", runtimeEnv["GLUE_RDMA_PATH_MTU"])
	require.Equal(t, "mlx5_0", runtimeEnv["NIC_NAME"])
	require.Equal(t, "19878", runtimeEnv["READINESS_PORT"])
	require.Equal(t, "default", runtimeEnv["LPU_MODEL_NAME"])
	gasDirSource := envValueSource(main.Env, "GAS_DIR")
	require.NotNil(t, gasDirSource)
	require.NotNil(t, gasDirSource.ConfigMapKeyRef)
	lpuConfigName := lpx.LPUConfigMapName(plan.PodCliqueSetName, agent.Annotations[commonconsts.AnnotationExtraResourcesHash])
	require.Equal(t, lpuConfigName, gasDirSource.ConfigMapKeyRef.Name)
	require.Equal(t, "gas_dir", gasDirSource.ConfigMapKeyRef.Key)
	require.Equal(t, "topologies", envValueSource(main.Env, "TOPOLOGIES").ConfigMapKeyRef.Key)
	podIPSource := envValueSource(main.Env, "POD_IP")
	require.NotNil(t, podIPSource)
	require.NotNil(t, podIPSource.FieldRef)
	require.Equal(t, "status.podIP", podIPSource.FieldRef.FieldPath)
	require.NotContains(t, runtimeEnv, "PARTITION_ID")
	require.NotContains(t, runtimeEnv, "RANK_IN_PARTITION")

	t.Log("Preserve custom tolerations and mounts alongside the required Agent networking and storage")
	require.True(t, agent.Spec.PodSpec.HostIPC)
	require.True(t, agent.Spec.PodSpec.HostNetwork)
	require.Equal(t, corev1.DNSClusterFirstWithHostNet, agent.Spec.PodSpec.DNSPolicy)
	require.Equal(t, []corev1.Toleration{
		{Key: "cluster.example/custom", Operator: corev1.TolerationOpExists},
		{Key: "lpu.nvidia.com/node", Operator: corev1.TolerationOpExists},
		{Key: "lpu.nvidia.com/lpu", Operator: corev1.TolerationOpExists},
		{Key: "lpu.nvidia.com/node-v2", Operator: corev1.TolerationOpExists},
	}, agent.Spec.PodSpec.Tolerations)
	for _, name := range []string{"config", "hugepages", "host-dev", "host-sys", "ssh-secret"} {
		require.True(t, slices.ContainsFunc(agent.Spec.PodSpec.Volumes, func(volume corev1.Volume) bool { return volume.Name == name }), "missing volume %s", name)
	}
	require.True(t, slices.ContainsFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == "host-dev" }))
	require.True(t, slices.ContainsFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == "host-sys" }))
	require.True(t, slices.ContainsFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == "hugepages" }))
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "config", MountPath: "/custom", ReadOnly: true})
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "config", MountPath: "/configs"})
	require.Contains(t, agent.Spec.PodSpec.Volumes, customConfigVolume)
	require.True(t, main.Resources.Requests.Cpu().Equal(resource.MustParse("62")))
	require.Equal(t, resource.MustParse("4096Mi"), main.Resources.Requests[corev1.ResourceHugePagesPrefix+"2Mi"])

	t.Log("Verify Cyborg receives its generated config and InfiniBand bindings instead of fallback PVCs")
	require.Contains(t, cyborg.Spec.PodSpec.Containers, sidecar)
	require.NotEmpty(t, cyborg.Spec.PodSpec.Containers)
	cyborgMain := cyborg.Spec.PodSpec.Containers[0]
	require.Equal(t, authoredCyborgMain.Command, cyborgMain.Command)
	require.Equal(t, authoredCyborgMain.Args, cyborgMain.Args)
	require.Contains(t, cyborgMain.VolumeMounts, corev1.VolumeMount{Name: "config", MountPath: "/configs"})
	require.True(t, slices.ContainsFunc(cyborgMain.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == "infiniband" }))

	volumes := make(map[string]corev1.Volume, len(cyborg.Spec.PodSpec.Volumes))
	for _, volume := range cyborg.Spec.PodSpec.Volumes {
		volumes[volume.Name] = volume
	}
	configVolume := volumes["config"]
	require.NotNil(t, configVolume.ConfigMap)
	decodeConfigName := plan.PodCliqueSetName + "-decode-" + cyborg.Annotations[commonconsts.AnnotationExtraResourcesHash][:16]
	require.Equal(t, decodeConfigName, configVolume.ConfigMap.Name)
	require.Nil(t, configVolume.PersistentVolumeClaim)
	infinibandVolume := volumes["infiniband"]
	require.NotNil(t, infinibandVolume.HostPath)
	require.Equal(t, "/dev/infiniband", infinibandVolume.HostPath.Path)
	require.Nil(t, infinibandVolume.PersistentVolumeClaim)

	t.Log("Verify Cyborg receives every runtime value consumed by its startup command")
	cyborgEnv := make(map[string]string, len(cyborgMain.Env))
	for _, variable := range cyborgMain.Env {
		cyborgEnv[variable.Name] = variable.Value
	}
	require.Equal(t, "19877", cyborgEnv["RDMA_PORT"])
	require.Equal(t, "/tmp/lpu_servers", cyborgEnv["SERVER_HOSTS_FILE"])
	require.Equal(t, "1", cyborgEnv["TOTAL_REPLICAS"])
	require.Equal(t, "2", cyborgEnv[lpx.CyborgBatchSizeEnv])
	require.NotContains(t, cyborgEnv, "CYBORG_SWA_CACHE_IDS")
	batchIndex := slices.IndexFunc(cyborgMain.Env, func(variable corev1.EnvVar) bool {
		return variable.Name == "CYBORG_BATCH_SIZE"
	})
	userIndex := slices.IndexFunc(cyborgMain.Env, func(variable corev1.EnvVar) bool {
		return variable.Name == "USER_BATCH_REF"
	})
	require.NotEqual(t, -1, batchIndex)
	require.NotEqual(t, -1, userIndex)
	require.Less(t, batchIndex, userIndex)
	var tokenizerEnv *corev1.EnvVar
	for index := range cyborgMain.Env {
		if cyborgMain.Env[index].Name == "TOKENIZER_DIR" {
			tokenizerEnv = &cyborgMain.Env[index]
			break
		}
	}
	require.NotNil(t, tokenizerEnv)
	require.NotNil(t, tokenizerEnv.ValueFrom)
	require.NotNil(t, tokenizerEnv.ValueFrom.ConfigMapKeyRef)
	require.Equal(t, decodeConfigName, tokenizerEnv.ValueFrom.ConfigMapKeyRef.Name)
	require.Equal(t, "tokenizer_dir", tokenizerEnv.ValueFrom.ConfigMapKeyRef.Key)

	t.Log("Keep partition zero as the collapsed runtime root for all three physical partitions")
	lpuConfig := getResource[*corev1.ConfigMap](t, extraResources, lpuConfigName)
	require.Equal(t, "0", lpuConfig.Data["partition_ids"])
	require.Equal(t, "/nfs/node-local-v2-cpu-embeddings", lpuConfig.Data["gas_dir"])

	t.Log("Verify generated Cyborg and LPU config resource order and contents")
	decodeConfig := getResource[*corev1.ConfigMap](t, extraResources, decodeConfigName)
	require.Len(t, extraResources, 2)
	require.Equal(t, decodeConfigName, extraResources[0].GetName())
	require.Equal(t, lpuConfigName, extraResources[1].GetName())
	require.Equal(t, dgd.Namespace, decodeConfig.Namespace)
	require.Len(t, decodeConfig.Data, 2)
	require.Equal(t, []string{"lpx-${GROVE_PCSG_INDEX}-agt-0"}, strings.Split(decodeConfig.Data["lpu_servers"], "\n"))
	require.NotEmpty(t, decodeConfig.Data["tokenizer_dir"])
	decodeHash := lpx.LPUConfigMapHash(decodeConfig)
	require.Equal(t, decodeHash, cyborg.Annotations[commonconsts.AnnotationExtraResourcesHash])
}

func TestGenerateGrovePodCliqueSet_NodeLocalPreservesImageEntrypoint(t *testing.T) {
	t.Log("Share the immutable build registry across entrypoint cases")
	registry := newTestDataModelRegistry(t, t.TempDir())
	kubeClient := newTestLPXClient(t)
	modes := []struct {
		name            string
		file            string
		hybrid          bool
		conductorConfig bool
		hx              bool
	}{
		{name: "Single V2", file: "single_v2.input.yaml", conductorConfig: true},
		{name: "V2 LPU-only", file: "node-local-v2-lpu-only.input.yaml", conductorConfig: true},
		{name: "V2 hybrid", file: "node-local-v2-hybrid.input.yaml", hybrid: true, conductorConfig: true},
		{name: "HX LPU-only", file: "node-local-v3-hx-lpu-only.input.yaml", conductorConfig: true, hx: true},
		{name: "HX hybrid", file: "node-local-v3-hx-hybrid.input.yaml", hybrid: true, hx: true},
	}
	intents := []struct {
		name          string
		command       []string
		args          []string
		cyborgCommand []string
		cyborgArgs    []string
		hybridOnly    bool
		configPath    *string
		configRole    int
		envFrom       bool
		customProbes  bool
	}{
		{name: "image entrypoint and image command"},
		{name: "custom exec health probes", command: []string{"/opt/custom-runtime"}, customProbes: true},
		{name: "default startup without config mount", configPath: ptr.To(""), configRole: 1},
		{name: "default startup with misplaced config mount", configPath: ptr.To("/custom"), configRole: 1},
		{name: "default Agent without config mount", configPath: ptr.To("")},
		{name: "default Agent with misplaced config mount", configPath: ptr.To("/custom")},
		{name: "default Agent without config mount and unrelated EnvFrom", configPath: ptr.To(""), envFrom: true},
		{name: "image entrypoint with user arguments", args: []string{"serve"}},
		{name: "explicit executable with image arguments", command: []string{"/opt/custom-runtime"}},
		{name: "explicit executable and arguments", command: []string{"/opt/custom-runtime"}, args: []string{"serve"}},
		{name: "allocation flag in arguments", args: []string{"--allocation", "template-target"}},
		{name: "allocation flag in command", command: []string{"/opt/custom-runtime", "--allocation=template-target"}},
		{name: "allocation environment expansion", args: []string{"--allocation", "$(LPX_ALLOCATION)"}},
		{name: "argument terminator", command: []string{"/opt/custom-runtime"}, args: []string{"--", "literal argument"}},
		{name: "shell command", command: []string{"/bin/sh", "-c"}, args: []string{"exec /opt/custom-runtime --allocation $LPX_ALLOCATION"}},
		{
			name:          "explicit Cyborg shell remains user-owned",
			command:       []string{"/opt/custom-runtime"},
			args:          []string{"serve"},
			cyborgCommand: []string{"/bin/sh", "-c"},
			cyborgArgs:    []string{"exec /opt/cyborg-runtime serve"},
			hybridOnly:    true,
		},
		{
			name:          "detected SGLang Cyborg remains selected no-op",
			cyborgCommand: []string{"python3"},
			cyborgArgs:    []string{"-m", "dynamo.sglang"},
			hybridOnly:    true,
		},
		{
			name:          "detected TensorRT-LLM Cyborg remains selected no-op",
			cyborgCommand: []string{"python3"},
			cyborgArgs:    []string{"-m", "dynamo.trtllm"},
			hybridOnly:    true,
		},
	}

	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			payload, err := os.ReadFile(filepath.Join("../../dynamo/lpx/testdata", "from_dgd_yaml", mode.file))
			require.NoError(t, err)

			for _, intent := range intents {
				if intent.hybridOnly && !mode.hybrid || intent.configPath != nil && intent.configRole == 1 && !mode.conductorConfig {
					continue
				}
				t.Run(intent.name, func(t *testing.T) {
					t.Log("Render the LPX LPU and GPU roles from the same command intent")
					dgd := &v1beta1.DynamoGraphDeployment{}
					require.NoError(t, yaml.Unmarshal(payload, dgd))
					component := dgd.GetComponentByName("lpu")
					require.NotNil(t, component)
					agent := component.ComponentRole(v1beta1.ComponentRoleLPXAgent)
					templates := []*corev1.PodTemplateSpec{agent.PodTemplate, component.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate}
					for roleIndex, template := range templates {
						main := &template.Spec.Containers[0]
						require.Equal(t, commonconsts.MainContainerName, main.Name)
						command := intent.command
						args := intent.args
						if roleIndex == 1 && intent.cyborgCommand != nil {
							command = intent.cyborgCommand
							args = intent.cyborgArgs
						}
						main.Command = slices.Clone(command)
						main.Args = slices.Clone(args)
						main.StartupProbe, main.LivenessProbe, main.ReadinessProbe = nil, nil, nil
						if intent.customProbes {
							main.StartupProbe = &corev1.Probe{
								ProbeHandler:     corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"/custom-health", fmt.Sprint(roleIndex), "started"}}},
								FailureThreshold: 12, PeriodSeconds: 4, TimeoutSeconds: 2,
							}
							main.LivenessProbe = &corev1.Probe{
								ProbeHandler:     corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"/custom-health", fmt.Sprint(roleIndex), "live"}}},
								FailureThreshold: 4, PeriodSeconds: 6, TimeoutSeconds: 3,
							}
							main.ReadinessProbe = &corev1.Probe{
								ProbeHandler:     corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"/custom-health", fmt.Sprint(roleIndex), "ready"}}},
								FailureThreshold: 2, PeriodSeconds: 5, TimeoutSeconds: 1,
							}
						}
						main.Env = append(main.Env, corev1.EnvVar{Name: "LPX_ALLOCATION", Value: "forged-allocation"})
						if intent.envFrom && roleIndex == 0 {
							main.EnvFrom = []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{LocalObjectReference: corev1.LocalObjectReference{Name: "unrelated-env"}}}}
						}
						if intent.configPath != nil && roleIndex == intent.configRole {
							configIndex := slices.IndexFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.Name == "config" })
							require.NotEqual(t, -1, configIndex)
							require.Equal(t, "/configs", main.VolumeMounts[configIndex].MountPath)
							if *intent.configPath == "" {
								main.VolumeMounts = slices.Delete(main.VolumeMounts, configIndex, configIndex+1)
							} else {
								main.VolumeMounts[configIndex].MountPath = *intent.configPath
							}
						}
					}

					before := dgd.DeepCopy()
					selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, registry)
					require.NoError(t, err)
					pcs, _, err := renderPodCliqueSet(
						t.Context(),
						dgd,
						&configv1alpha1.OperatorConfiguration{},
						&controller_common.RuntimeConfig{},
						kubeClient,
						nil,
						selected,
						mustPlanSelectedLPX(t, dgd, selected),
						newLPXRenderDeployment(t, dgd),
					)
					require.Equal(t, before, dgd)
					if mode.hx && intent.configPath != nil && *intent.configPath != "" {
						require.ErrorContains(t, err, `reserves volume "config" at "/configs"`)
						return
					}
					require.NoError(t, err)

					t.Log("Verify every role preserves authored startup and health probes, including omission")
					var agents, conductors, cyborgs int
					for _, clique := range pcs.Spec.Template.Cliques {
						main := clique.Spec.PodSpec.Containers[0]
						authored := templates[1].Spec.Containers[0]
						if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent {
							authored = templates[0].Spec.Containers[0]
						}
						require.Equal(t, authored.Command, main.Command, "%s command", clique.Name)
						require.True(t, slices.Equal(authored.Args, main.Args), "%s args", clique.Name)
						require.Equal(t, authored.StartupProbe, main.StartupProbe, "%s startup", clique.Name)
						require.Equal(t, authored.LivenessProbe, main.LivenessProbe, "%s liveness", clique.Name)
						require.Equal(t, authored.ReadinessProbe, main.ReadinessProbe, "%s readiness", clique.Name)
						switch clique.Annotations[lpxv1alpha1.PodRoleAnnotation] {
						case lpxv1alpha1.PodRoleAgent:
							agents++
							require.False(t, slices.ContainsFunc(clique.Spec.PodSpec.Volumes, func(volume corev1.Volume) bool { return volume.Name == "tmp" }))
						case lpxv1alpha1.PodRoleConductor:
							conductors++
							require.Contains(t, main.Env, corev1.EnvVar{
								Name: "LPX_ALLOCATION", Value: strings.Join(clique.Spec.StartsAfter, ","),
							})
							require.NotContains(t, main.Env, corev1.EnvVar{Name: "LPX_ALLOCATION", Value: "forged-allocation"})
						case lpxv1alpha1.PodRoleCyborgWorker:
							cyborgs++
							require.NotEmpty(t, main.Ports)
						}
					}
					require.Positive(t, agents)
					if mode.hybrid {
						require.Zero(t, conductors)
						require.Equal(t, 1, cyborgs)
					} else {
						require.Equal(t, 1, conductors)
						require.Zero(t, cyborgs)
					}
				})
			}
		})
	}
}

func newTestLPXClient(t *testing.T) client.Client {
	t.Helper()
	return fake.NewClientBuilder().Build()
}

func envValueSource(envs []corev1.EnvVar, name string) *corev1.EnvVarSource {
	for index := range envs {
		if envs[index].Name == name {
			return envs[index].ValueFrom
		}
	}
	return nil
}

func getResource[T any](t *testing.T, resources []client.Object, name string) T {
	t.Helper()
	i := slices.IndexFunc(resources, func(resource client.Object) bool { return resource.GetName() == name })
	require.GreaterOrEqual(t, i, 0)
	return resources[i].(T)
}

func TestLPXRenderingPreservesCyborgOverrides(t *testing.T) {
	t.Log("Load an authored hybrid engine and override its independent leader template")
	payload, err := os.ReadFile(filepath.Join("../../dynamo/lpx/testdata", "from_dgd_yaml", "node-local-v2-hybrid.input.yaml"))
	require.NoError(t, err)
	source := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, source))
	leader := lpx.ServingComponent(source).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	leader.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{
		{Name: "config", MountPath: "/custom-config", ReadOnly: true},
		{Name: "infiniband", MountPath: "/custom-infiniband", ReadOnly: true},
		{Name: "model-storage", MountPath: "/nfs"},
	}
	leader.Spec.Containers[0].Env = []corev1.EnvVar{
		{Name: "TOKENIZER_DIR", Value: "/custom-tokenizer"},
		{Name: "TOTAL_REPLICAS", Value: "9"},
		{Name: "SERVER_HOSTS_FILE", Value: "/custom-servers"},
		{Name: lpx.CyborgBatchSizeEnv, Value: "2"},
	}
	authoredConfig := corev1.Volume{
		Name:         "config",
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	}
	leader.Spec.Volumes = append(leader.Spec.Volumes, authoredConfig)

	t.Log("Render the complete LPX workload with ordinary pod spec overrides")
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), source, newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan := mustPlanSelectedLPX(t, source, selected)
	pcs, _, err := renderPodCliqueSet(t.Context(), source,
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{}, newTestLPXClient(t), nil, selected,
		plan, newLPXRenderDeployment(t, source))
	require.NoError(t, err)
	cliqueIndex := slices.IndexFunc(pcs.Spec.Template.Cliques, func(clique *grovev1alpha1.PodCliqueTemplateSpec) bool {
		return clique.Name == plan.CyborgTemplate
	})
	require.GreaterOrEqual(t, cliqueIndex, 0)
	podSpec := pcs.Spec.Template.Cliques[cliqueIndex].Spec.PodSpec

	t.Log("Keep one config volume alongside the authored InfiniBand volume")
	require.Contains(t, podSpec.Volumes, leader.Spec.Volumes[0])
	var configVolumes []corev1.Volume
	for _, volume := range podSpec.Volumes {
		if volume.Name == authoredConfig.Name {
			configVolumes = append(configVolumes, volume)
		}
	}
	require.Equal(t, []corev1.Volume{authoredConfig}, configVolumes)

	t.Log("Preserve authored mounts and runtime environment over the generated Cyborg defaults")
	main := podSpec.Containers[0]
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "config", MountPath: "/custom-config", ReadOnly: true})
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "infiniband", MountPath: "/custom-infiniband", ReadOnly: true})
	require.False(t, slices.ContainsFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.MountPath == "/configs" }))
	env := make(map[string]string, len(main.Env))
	for _, variable := range main.Env {
		env[variable.Name] = variable.Value
	}
	require.Equal(t, "/custom-tokenizer", env["TOKENIZER_DIR"])
	require.Equal(t, "9", env["TOTAL_REPLICAS"])
	require.Equal(t, "/custom-servers", env["SERVER_HOSTS_FILE"])

	t.Log("Preserve the authored command and host-file override without adding a launcher")
	require.Equal(t, leader.Spec.Containers[0].Command, main.Command)
	require.Equal(t, leader.Spec.Containers[0].Args, main.Args)
}
