// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"capnproto.org/go/capnp/v3"
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	"gotest.tools/v3/golden"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

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
	kubeClient := fake.NewClientBuilder().Build()
	runtimeConfig := &commoncontroller.RuntimeConfig{}

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
			r := &graphReconciler{config: controllerConfig, runtimeConfig: runtimeConfig}
			got, extraResources, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, &dynamoDeployment), &dynamoDeployment, selected, plan)
			require.NoError(t, err)
			for _, clique := range got.Spec.Template.Cliques {
				require.Equal(t, lpx.SchedulerName, clique.Spec.PodSpec.SchedulerName)
				component := dynamoDeployment.GetComponentByName(clique.Labels[consts.KubeLabelDynamoComponent])
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
				require.NotEqual(t, lpx.SchedulerName, clique.Spec.PodSpec.SchedulerName)
				require.NotEqual(t, selected.LPXComponentName(), clique.Labels[consts.KubeLabelDynamoComponent])
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

			t.Log("Replace computed content hashes with stable golden placeholders; mismatched references remain visible")
			replacements := []string{registryRoot, "/testdata"}
			for _, resource := range extraResources {
				config, ok := resource.(*corev1.ConfigMap)
				if !ok {
					continue
				}
				hash := lpx.LPUConfigMapHash(config)
				role := strings.TrimPrefix(strings.TrimSuffix(config.Name, "-"+hash[:16]), got.Name+"-")
				replacements = append(replacements,
					hash, "<"+role+"-config-hash>",
					hash[:16], "<"+role+"-config-hash-prefix>",
				)
			}
			goldenPath, err := filepath.Abs("../../dynamo/lpx/testdata/" + name + ".yaml")
			require.NoError(t, err)
			const header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n" +
				"# SPDX-License-Identifier: Apache-2.0\n\n"
			golden.Assert(t, header+strings.NewReplacer(replacements...).Replace(strings.Join(out, "---\n")), goldenPath)
		})
	}
}

func TestLPXRenderingChecksFinalPodCliqueSetSize(t *testing.T) {
	t.Log("Resolve a real LPU workload including inherited scheduler metadata")
	payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/node-local-v2-lpu-only.input.yaml")
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	dgd.Annotations = map[string]string{"kai.scheduler/padding": ""}
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)
	config := &configv1alpha1.OperatorConfiguration{}
	runtimeConfig := &commoncontroller.RuntimeConfig{}
	r := &graphReconciler{config: config, runtimeConfig: runtimeConfig}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, selected, plan)
	require.NoError(t, err)
	serialized, err := json.Marshal(pcs)
	require.NoError(t, err)

	t.Log("Accept exactly one MiB including final identity, discovery and scheduler metadata")
	dgd.Annotations["kai.scheduler/padding"] = strings.Repeat("x", lpx.MaxRenderedPodCliqueSetBytes-len(serialized))
	pcs, _, err = r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, selected, plan)
	require.NoError(t, err)
	serialized, err = json.Marshal(pcs)
	require.NoError(t, err)
	require.Len(t, serialized, lpx.MaxRenderedPodCliqueSetBytes)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Annotations, "kai.scheduler/padding")
	}

	t.Log("Reject one additional final-metadata byte as a selected-render failure before publication")
	dgd.Annotations["kai.scheduler/padding"] += "x"
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, selected, plan)
	require.ErrorContains(t, err, "rendered LPX PodCliqueSet is 1048577 bytes; maximum is 1048576")
	require.Nil(t, pcs)
	require.Nil(t, resources)
}

func TestLPXRenderingPreservesCyborgOverrides(t *testing.T) {
	t.Log("Load an authored hybrid engine and override its independent leader template")
	payload, err := os.ReadFile(filepath.Join("../../dynamo/lpx/testdata", "from_dgd_yaml", "node-local-v2-hybrid.input.yaml"))
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	leader := lpx.ServingComponent(dgd).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	leader.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{
		{Name: "config", MountPath: "/custom-config", ReadOnly: true},
		{Name: "infiniband", MountPath: "/custom-infiniband", ReadOnly: true},
		{Name: "model-storage", MountPath: "/nfs"},
	}
	leader.Spec.Containers[0].Env = []corev1.EnvVar{
		{Name: "TOKENIZER_DIR", Value: "/custom-tokenizer"},
		{Name: "TOTAL_REPLICAS", Value: "9"},
		{Name: "SERVER_HOSTS_FILE", Value: "/custom-servers"},
	}
	for _, name := range []string{"CYBORG_BATCH_SIZE", "CYBORG_FPGA_GPI_IO_FPGA_COUNT", "CYBORG_SWA_CACHE_IDS"} {
		leader.Spec.Containers[0].Env = append(leader.Spec.Containers[0].Env, corev1.EnvVar{
			Name: name, ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: "runtime-settings"}, Key: name,
			}},
		})
	}
	authoredConfig := corev1.Volume{
		Name:         "config",
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	}
	leader.Spec.Volumes = append(leader.Spec.Volumes, authoredConfig)

	t.Log("Render the complete LPX workload with ordinary pod spec overrides")
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan := mustPlanSelectedLPX(t, dgd, selected)
	r := &graphReconciler{config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, selected, plan)
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

	t.Log("Preserve authored mounts and runtime environment without interpreting literal or valueFrom settings")
	main := podSpec.Containers[0]
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "config", MountPath: "/custom-config", ReadOnly: true})
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "infiniband", MountPath: "/custom-infiniband", ReadOnly: true})
	require.False(t, slices.ContainsFunc(main.VolumeMounts, func(mount corev1.VolumeMount) bool { return mount.MountPath == "/configs" }))
	for _, variable := range leader.Spec.Containers[0].Env {
		require.Contains(t, main.Env, variable)
	}

	t.Log("Preserve the authored command and host-file override without adding a launcher")
	require.Equal(t, leader.Spec.Containers[0].Command, main.Command)
	require.Equal(t, leader.Spec.Containers[0].Args, main.Args)
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
			dgd := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, yaml.Unmarshal(payload, dgd))
			dgd.UID, dgd.Generation = "source-dgd-uid", 7
			dgd.Labels = map[string]string{
				"priorityClassName": "inference", "kai.scheduler/preemptibility": "NonPreemptible",
				"project": "test", "unrelated": "ignored",
				consts.KubeLabelKaiSchedulerQueue: "lpx-queue",
			}
			metav1.SetMetaDataAnnotation(&dgd.ObjectMeta, "kai.scheduler/topology", "source-topology")
			metav1.SetMetaDataAnnotation(&dgd.ObjectMeta, "unrelated", "ignored")
			metav1.SetMetaDataAnnotation(&dgd.ObjectMeta, consts.KubeAnnotationEnableMetrics, "false")
			dgd.Spec.Labels = map[string]string{"shared-label": "source"}
			dgd.Spec.Annotations = map[string]string{
				"shared-annotation": "source", "kai.scheduler/topology": "explicit-pcs-topology",
				"selected-metadata":         "from-dgd",
				lpx.DGDUIDAnnotation:        "stale",
				lpx.DGDGenerationAnnotation: "stale",
			}
			dgd.Spec.Env = []corev1.EnvVar{{Name: "LPX_DGD_ENV", Value: test.name}}
			for _, component := range lpx.Components(dgd) {
				component.GlobalDynamoNamespace = true
				component.RuntimeVersionOverride = "1.5.0"
				for _, role := range component.Roles {
					if role.PodTemplate != nil {
						metav1.SetMetaDataAnnotation(&role.PodTemplate.ObjectMeta, "selected-metadata", "from-component")
						metav1.SetMetaDataAnnotation(&role.PodTemplate.ObjectMeta, consts.RestartAnnotation, "2026-09-07T00:00:00Z")
						metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, "shared-label", "from-component")
						metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, consts.KubeLabelKaiSchedulerQueue, "lpx-queue")
						metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, consts.KubeLabelDynamoNamespace, "authored-namespace")
					}
				}
			}
			component := lpx.ServingComponent(dgd)
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

			t.Log("Freeze the authored role metadata and replica units before rendering")
			child := newLPXRenderDeployment(t, dgd)
			child.Annotations[dynamo.LPXRestartAnnotation] = restartToken
			child.Spec.InputRevision, err = dynamo.LPXInputRevision(dgd, restartToken)
			require.NoError(t, err)
			dgd.Generation++
			before, childBefore := dgd.DeepCopy(), child.DeepCopy()
			selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, registry)
			require.NoError(t, err)
			plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
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
			r := &graphReconciler{config: config, runtimeConfig: &commoncontroller.RuntimeConfig{}}
			first, firstResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, selected, plan)
			require.NoError(t, err)
			second, secondResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, selected, plan)
			require.NoError(t, err)
			require.Equal(t, first, second)
			require.Equal(t, firstResources, secondResources)
			require.Equal(t, "inference", first.Labels["priorityClassName"])
			require.Equal(t, "NonPreemptible", first.Labels["kai.scheduler/preemptibility"])
			require.Equal(t, "lpx-queue", first.Labels[consts.KubeLabelKaiSchedulerQueue])
			require.Equal(t, "test", first.Labels["project"])
			require.Equal(t, "explicit-pcs-topology", first.Annotations["kai.scheduler/topology"])
			require.NotContains(t, first.Annotations, consts.RestartAnnotation)
			require.NotContains(t, first.Labels, "unrelated")
			require.NotContains(t, first.Annotations, "unrelated")
			require.Equal(t, selected.Digest().String(), first.Annotations[lpx.WorkloadDigestAnnotation])

			t.Log("Stamp frozen source identity on every final resource despite a newer ordinary DGD generation")
			require.NotEqual(t, dgd.UID, child.UID)
			annotationMaps := []map[string]string{first.Annotations}
			for _, clique := range first.Spec.Template.Cliques {
				require.Equal(t, restartToken, clique.Annotations[consts.RestartAnnotation])
				annotationMaps = append(annotationMaps, clique.Annotations)
			}
			for _, group := range first.Spec.Template.PodCliqueScalingGroupConfigs {
				require.NotContains(t, group.Annotations, consts.RestartAnnotation)
				annotationMaps = append(annotationMaps, group.Annotations)
			}
			for _, resource := range firstResources {
				require.NotContains(t, resource.GetAnnotations(), consts.RestartAnnotation)
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
			equivalent := dgd.DeepCopy()
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
			equivalentPCS, equivalentResources, err := r.renderPodCliqueSet(t.Context(), child, equivalent, equivalentSelected, equivalentPlan)
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
				scaled := dgd.DeepCopy()
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
				pcs, resources, err := r.renderPodCliqueSet(t.Context(), scaledChild, scaled, workload, scaledPlan)
				require.NoError(t, err)
				require.Equal(t, firstResources, resources)
				require.Equal(t, first.Spec.Template.Cliques, pcs.Spec.Template.Cliques)
				require.Equal(t, first.Spec.Template.PodCliqueScalingGroupConfigs, pcs.Spec.Template.PodCliqueScalingGroupConfigs)
			}

			t.Log("Connect every model projection to its rendered Agent while preserving shared runtime metadata and replica units")
			require.Len(t, first.Spec.Template.PodCliqueScalingGroupConfigs, 1)
			group := first.Spec.Template.PodCliqueScalingGroupConfigs[0]
			require.Equal(t, ptr.To(ptr.Deref(component.MinAvailable, 1)), group.Replicas)
			require.Equal(t, component.MinAvailable, group.MinAvailable)
			var conductor, gpuClique *grovev1alpha1.PodCliqueTemplateSpec
			serving := 0
			for index, clique := range first.Spec.Template.Cliques {
				require.Equal(t, lpx.SchedulerName, clique.Spec.PodSpec.SchedulerName, "role %s must use the LPX backend", clique.Name)
				require.Equal(t, "lpx-queue", clique.Labels[consts.KubeLabelKaiSchedulerQueue])
				if clique.Labels[dynamo.LPXServingLabel] == consts.KubeLabelValueTrue {
					serving++
					require.NotEqual(t, lpxv1alpha1.PodRoleAgent, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
					require.Equal(t, dynamo.HashModelName("test/model"), clique.Labels[consts.KubeLabelDynamoBaseModelHash])
				} else {
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoBaseModelHash)
				}
				require.Equal(t, ptr.To(clique.Spec.Replicas), clique.Spec.MinAvailable)
				require.NotEmpty(t, clique.Spec.PodSpec.Containers)
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "LPX_DGD_ENV", Value: test.name})
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: consts.DynamoNamespaceEnvVar, Value: consts.GlobalDynamoNamespace})
				require.Equal(t, consts.GlobalDynamoNamespace, clique.Labels[consts.KubeLabelDynamoNamespace])
				require.Equal(t, "from-component", clique.Labels["shared-label"])
				require.Equal(t, "false", clique.Annotations[consts.KubeAnnotationEnableMetrics])
				require.Equal(t, "kubernetes", clique.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend])
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
			require.Equal(t, before, dgd)
			require.Equal(t, childBefore, child)
		})
	}
}

func TestLPXSpecDecodeConductorTemplate(t *testing.T) {
	t.Log("Give the speculative components distinct authored Agent and conductor commands")
	registry := newTestDataModelRegistry(t, t.TempDir())
	payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/node-local-v3-hx-specdecode.input.yaml")
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	for _, component := range lpx.Components(dgd) {
		agent := component.ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate
		agent.Spec.Containers[0].Image = component.ComponentName + "-runtime"
		agent.Spec.Containers[0].Command = []string{"/bin/quasar-entrypoint"}
		agent.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "AGENT_ONLY", Value: "kept"}}
	}
	target := lpx.ServingComponent(dgd)
	template := target.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	template.Spec.Containers[0].Image = "independent-conductor-runtime"
	template.Spec.Containers[0].Command = []string{"/bin/nova"}
	template.Spec.Containers[0].Env = append(template.Spec.Containers[0].Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
	template.Labels = map[string]string{"owner": "explicit-conductor"}
	template.Spec.NodeSelector = map[string]string{"runtime-role": "conductor"}
	template.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
			MatchExpressions: []corev1.NodeSelectorRequirement{{Key: "pool", Operator: corev1.NodeSelectorOpIn, Values: []string{"conductor"}}},
		}}},
	}}
	before := dgd.DeepCopy()
	selected, err := lpx.ResolveSelectedWorkload(t.Context(), dgd, registry)
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)

	t.Log("Render one shared conductor without changing either component's Agent template")
	r := &graphReconciler{config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, selected, plan)
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
			require.NotContains(t, container.Args, "--datacenter-config-filepath")
			require.Contains(t, container.Env, corev1.EnvVar{
				Name: "NOVA_NODE_NAME_TEMPLATE", Value: "${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-{rack}-{node}.${GROVE_HEADLESS_SERVICE}",
			})
			require.Equal(t, "explicit-conductor", clique.Labels["owner"])
			require.Equal(t, template.Spec.NodeSelector, clique.Spec.PodSpec.NodeSelector)
			require.Equal(t, template.Spec.Affinity, clique.Spec.PodSpec.Affinity)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
		} else {
			component := clique.Labels[consts.KubeLabelDynamoComponent]
			require.Equal(t, component+"-runtime", container.Image)
			require.Equal(t, []string{"/bin/quasar-entrypoint"}, container.Command)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
		}
	}
	require.Equal(t, 1, conductors)
	require.Equal(t, before, dgd)
}

func TestRuntimeTemplateChangesPreservePartitionConfig(t *testing.T) {
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

	t.Log("Render runtime overrides supplied directly by the conductor template")
	var partitionData map[string]string
	for _, env := range []corev1.EnvVar{
		{Name: "NOVA_BATCH_SIZE", Value: "1"},
		{Name: "NOVA_SEQUENCE_LENGTH", Value: "65536"},
		{Name: "A_CUSTOM_MODEL_PATH", Value: "$(LPX_MODEL_PATH)"},
	} {
		payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/single_v2.input.yaml")
		require.NoError(t, err)
		var deployment v1beta1.DynamoGraphDeployment
		require.NoError(t, yaml.Unmarshal(payload, &deployment))
		component := &deployment.Spec.Components[0]
		component.LPX = &v1beta1.LPXConfig{BuildID: buildID}
		template := component.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
		template.Spec.Containers[0].Env = []corev1.EnvVar{env}

		selected, err := lpx.ResolveSelectedWorkload(t.Context(), &deployment, registry)
		require.NoError(t, err)
		child := newLPXRenderDeployment(t, &deployment)
		plan := mustPlanSelectedLPX(t, &deployment, selected)
		r := &graphReconciler{config: controllerConfig, runtimeConfig: &commoncontroller.RuntimeConfig{}}
		pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, &deployment, selected, plan)
		require.NoError(t, err)

		t.Log("Keep authored environment values without changing shared partition files")
		var hash string
		for _, clique := range pcs.Spec.Template.Cliques {
			if clique.Name == plan.ConductorTemplate {
				require.Equal(t, corev1.EnvVar{Name: "LPX_MODEL_PATH", Value: filepath.Join("/nfs", buildID)}, clique.Spec.PodSpec.Containers[0].Env[0])
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, env)
				hash = clique.Annotations[consts.AnnotationExtraResourcesHash]
			}
		}
		require.NotEmpty(t, hash)
		configMap := getResource[*corev1.ConfigMap](t, resources, lpx.LPUConfigMapName(dynamo.PCSNameForLPX(child), hash))
		require.NotContains(t, configMap.Data, "model_config.toml")
		require.NotContains(t, configMap.Data, "datacenter.toml")
		if partitionData == nil {
			partitionData = configMap.Data
		} else {
			require.Equal(t, partitionData, configMap.Data)
		}
	}
}

func getResource[T any](t *testing.T, resources []client.Object, name string) T {
	t.Helper()
	i := slices.IndexFunc(resources, func(resource client.Object) bool { return resource.GetName() == name })
	require.GreaterOrEqual(t, i, 0)
	return resources[i].(T)
}

func mustPlanSelectedLPX(t *testing.T, dgd *v1beta1.DynamoGraphDeployment, selected *lpx.SelectedWorkload) *lpx.MaterializationPlan {
	t.Helper()
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)
	return plan
}

func newLPXRenderDeployment(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) *v1alpha1.LPXGraphDeployment {
	t.Helper()
	deployment := newLPXTestDeployment(t, dgd)
	deployment.UID = "lpx-render-uid"
	return deployment
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

func populateTestGraphBuild(t *testing.T, manifest manifestcapnpv2.Manifest, wireBuildID string) {
	t.Helper()

	t.Log("Populate the compiler-owned build identity used only as opaque provenance")
	build, err := capnp.NewStruct(manifest.Segment(), capnp.ObjectSize{DataSize: 8, PointerCount: 12})
	require.NoError(t, err)
	require.NoError(t, build.SetText(11, wireBuildID))
	require.NoError(t, manifest.SetReserved3(build.ToPtr()))
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

	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
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

type testV2GraphManifestFixture struct {
	topology              string
	partitionCount        int
	numChips              uint32
	devicesPerNode        uint32
	selectedPropSyncChain []uint32
	compilationMode       manifestcapnpv2.CompilationMode
	nonLPUDeviceTypes     []manifestcapnpv2.DeviceType
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

type testV3GraphManifestFixture struct {
	compilationMode   manifestcapnpv2.CompilationMode
	nonLPUDeviceTypes []manifestcapnpv2.DeviceType
}

func writeTestGraphBuild(t *testing.T, registryRoot, buildID string, manifest []byte) {
	t.Helper()

	buildDir := filepath.Join(registryRoot, buildID)
	require.NoError(t, os.MkdirAll(buildDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.v2.capnp.bin"), manifest, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.json"), []byte(`{}`), 0o600))
}

func renderLPXTestPodCliqueSet(
	t *testing.T, ctx context.Context, reconciler *graphReconciler,
	deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment, desired *lpxTestWorkload,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	rendered, _, err := reconciler.renderPodCliqueSet(ctx, deployment, dgd, desired.workload, desired.plan)
	require.NoError(t, err)
	return rendered
}

func newLPXTestRegistryWithPartitionsAndMode(
	t *testing.T,
	buildID string,
	partitionIDs []int,
	compilationMode manifestcapnpv2.CompilationMode,
) lpx.ModelRegistry {
	t.Helper()
	root := t.TempDir()
	writeLPXTestBuild(t, root, buildID, partitionIDs, compilationMode)
	registryURL := (&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String()
	registry, err := lpx.NewModelRegistry(registryURL, nil)
	require.NoError(t, err)
	return registry
}

func writeLPXTestBuild(
	t *testing.T,
	root string,
	buildID string,
	partitionIDs []int,
	compilationMode manifestcapnpv2.CompilationMode,
) {
	t.Helper()
	require.NotEmpty(t, partitionIDs)
	buildDir := filepath.Join(root, buildID)
	require.NoError(t, os.Mkdir(buildDir, 0o700))

	t.Log("Encode the V2 manifest header and the tokenizer consumed by rendered runtimes")
	message, manifest := newTestGraphManifest(t)

	t.Log("Retain the opaque compiler provenance without the removed V1 publication contract")
	populateTestGraphBuild(t, manifest, buildID)

	t.Log("Describe the same one-batch runtime with explicit V2 runtime I/O and prop-sync evidence")
	deployment, program := newTestGraphProgram(t, manifest, compilationMode, uint32(len(partitionIDs)*2), 8192)
	program.SetNumKvCaches(1)
	program.SetNumBatchSplitDivisions(1)
	chains, err := deployment.NewSelectedPropSyncChains(1)
	require.NoError(t, err)
	selectedIDs, err := chains.At(0).NewPartitionIds(int32(len(partitionIDs)))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		selectedIDs.Set(index, uint32(partitionID))
	}

	t.Log("Package the LPU partitions and the hybrid CUDA marker in the flat V2 artifact inventory")
	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	partitionCount := len(partitionIDs)
	if compilationMode == manifestcapnpv2.CompilationMode_lpx {
		partitionCount++
	}
	partitions, err := artifacts.NewPartitions(int32(partitionCount))
	require.NoError(t, err)
	for index, partitionID := range partitionIDs {
		partition := partitions.At(index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(manifestcapnpv2.DeviceType_lpu)
		ref.SetPartitionId(uint32(partitionID))
		detail, err := partition.Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", partitionID)))
		require.NoError(t, detail.SetTopology("URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA"))
		detail.SetNumChips(16)
		detail.SetDevicesPerNode(8)
	}
	if compilationMode == manifestcapnpv2.CompilationMode_lpx {
		partition := partitions.At(len(partitionIDs))
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(manifestcapnpv2.DeviceType_cuda)
		ref.SetPartitionId(uint32(len(partitionIDs)))
		_, err = partition.Detail().NewCuda()
		require.NoError(t, err)
	}

	t.Log("Publish the required V2 compiler manifest")
	payload, err := message.Marshal()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.v2.capnp.bin"), payload, 0o600))
}
