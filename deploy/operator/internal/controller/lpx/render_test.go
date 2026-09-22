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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
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
		"from_dgd_yaml/lpu-cyborg-specdec",
		"from_dgd_yaml/lpu-gpu-specdecode",
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
			t.Log("Resolve and render every selected workload into the shared PCS")
			r := &graphReconciler{config: controllerConfig, runtimeConfig: runtimeConfig, modelRegistry: testdataModelRegistry}
			child := newLPXRenderDeployment(t, &dynamoDeployment)
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, &dynamoDeployment)
			require.NoError(t, err)
			got, extraResources, err := r.renderPodCliqueSet(t.Context(), child, &dynamoDeployment, workloads, plans)
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
				require.False(t, dynamoDeployment.GetComponentByName(clique.Labels[consts.KubeLabelDynamoComponent]).IsLPX())
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

			goldenPath, err := filepath.Abs("../../dynamo/lpx/testdata/" + name + ".yaml")
			require.NoError(t, err)
			const header = "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n" +
				"# SPDX-License-Identifier: Apache-2.0\n\n"
			golden.Assert(t, header+strings.Join(out, "---\n"), goldenPath)
		})
	}
}

func TestLPXRenderingIncludesDiscoveryServices(t *testing.T) {
	for _, tc := range []struct {
		name         string
		pipeline     lpx.Pipeline
		backend      configv1alpha1.DiscoveryBackend
		override     configv1alpha1.DiscoveryBackend
		wantServices bool
	}{
		{name: "discovery disabled", pipeline: lpx.PipelineSingle},
		{name: "operator enables Kubernetes discovery", pipeline: lpx.PipelineSingle, backend: configv1alpha1.DiscoveryBackendKubernetes, wantServices: true},
		{name: "DGD enables hybrid discovery", pipeline: lpx.PipelineLPX, override: configv1alpha1.DiscoveryBackendKubernetes, wantServices: true},
		{name: "DGD disables Kubernetes discovery", pipeline: lpx.PipelineSingle, backend: configv1alpha1.DiscoveryBackendKubernetes, override: configv1alpha1.DiscoveryBackendEtcd},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve two workloads with authored metadata and the selected discovery backend")
			child, dgd, registry := newLPXTestDGD(t, tc.pipeline)
			dgd.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
			second := dgd.Spec.Components[0].DeepCopy()
			second.ComponentName = "second"
			dgd.Spec.Components = append(dgd.Spec.Components, *second)
			dgd.Spec.Labels = map[string]string{"example.com/team": "inference"}
			dgd.Spec.Annotations = map[string]string{"example.com/description": "serving"}
			if tc.override != "" {
				dgd.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(tc.override)
			}
			r := newLPXTestReconciler(t, registry, child, dgd)
			r.config.Discovery.Backend = tc.backend
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			before := dgd.DeepCopy()

			t.Log("Render ConfigMaps and optional Services without writing resources or mutating the DGD")
			pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
			require.NoError(t, err)
			require.Equal(t, before, dgd)
			serviceCount, configMapCount := 0, 0
			for _, resource := range resources {
				switch resource.(type) {
				case *corev1.Service:
					serviceCount++
				case *corev1.ConfigMap:
					configMapCount++
				}
			}
			require.Positive(t, configMapCount)
			if !tc.wantServices {
				require.Zero(t, serviceCount)
				return
			}
			require.Equal(t, len(plans), serviceCount)

			t.Log("Only serving roles advertise model discovery")
			for _, clique := range pcs.Spec.Template.Cliques {
				if clique.Labels[dynamo.LPXServingLabel] == consts.KubeLabelValueTrue {
					require.NotEqual(t, lpxv1alpha1.PodRoleAgent, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
					require.Equal(t, dynamo.HashModelName("test/model"), clique.Labels[consts.KubeLabelDynamoBaseModelHash])
				} else {
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoBaseModelHash)
				}
			}

			t.Log("Each Service carries discovery metadata and selects its own serving component and PCS")
			for groupName, plan := range plans {
				service := getResource[*corev1.Service](t, resources, plan.ResourcePrefix+"-serve")
				component := dgd.GetComponentByName(groupName)
				require.Equal(t, child.Namespace, service.Namespace)
				require.Equal(t, "inference", service.Labels["example.com/team"])
				require.Equal(t, map[string]string{
					"example.com/description":    "serving",
					lpx.DeploymentNameAnnotation: child.Name,
					deploymentUIDAnnotation:      string(child.UID),
					lpx.DGDUIDAnnotation:         string(dgd.UID),
				}, service.Annotations)
				require.Equal(t, string(child.UID), service.Labels[deploymentUIDLabel])
				require.Equal(t, consts.KubeLabelValueTrue, service.Labels[consts.KubeLabelDynamoDiscoveryEnabled])
				require.Equal(t, consts.DiscoveryBackendKubernetes, service.Labels[consts.KubeLabelDynamoDiscoveryBackend])
				require.Equal(t, map[string]string{
					consts.KubeLabelDynamoComponent:     groupName,
					consts.KubeLabelDynamoComponentType: string(component.ComponentType),
					consts.KubeLabelDynamoNamespace:     dgd.GetDynamoNamespaceForComponent(component),
					dynamo.LPXServingLabel:              consts.KubeLabelValueTrue,
					grovecommon.LabelPartOfKey:          pcs.Name,
				}, service.Spec.Selector)
			}
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
	selected, err := lpx.ResolveWorkload(t.Context(), dgd, singleGroupComponents(t, dgd), newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	config := &configv1alpha1.OperatorConfiguration{}
	runtimeConfig := &commoncontroller.RuntimeConfig{}
	r := &graphReconciler{config: config, runtimeConfig: runtimeConfig}
	plan := mustPlanSelectedLPX(t, dgd, selected)
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
	require.NoError(t, err)
	serialized, err := json.Marshal(pcs)
	require.NoError(t, err)

	t.Log("Accept exactly one MiB including final identity, discovery and scheduler metadata")
	dgd.Annotations["kai.scheduler/padding"] = strings.Repeat("x", lpx.MaxRenderedPodCliqueSetBytes-len(serialized))
	pcs, _, err = r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
	require.NoError(t, err)
	serialized, err = json.Marshal(pcs)
	require.NoError(t, err)
	require.Len(t, serialized, lpx.MaxRenderedPodCliqueSetBytes)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Annotations, "kai.scheduler/padding")
	}

	t.Log("Reject one additional final-metadata byte as a selected-render failure before publication")
	dgd.Annotations["kai.scheduler/padding"] += "x"
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
	require.ErrorContains(t, err, "rendered LPX PodCliqueSet is 1048577 bytes; maximum is 1048576")
	require.Nil(t, pcs)
	require.Nil(t, resources)
}

func TestLPXRenderingPreservesCyborgOverrides(t *testing.T) {
	t.Log("Load an authored hybrid workload and override its independent leader template")
	payload, err := os.ReadFile(filepath.Join("../../dynamo/lpx/testdata", "from_dgd_yaml", "node-local-v2-hybrid.input.yaml"))
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	leader := dgd.GetComponentByName("lpu").ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
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
	selected, err := lpx.ResolveWorkload(t.Context(), dgd, singleGroupComponents(t, dgd), newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan := mustPlanSelectedLPX(t, dgd, selected)
	r := &graphReconciler{config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
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
	const mutatedValue = "rendered-change"
	t.Log("Create compiler builds for each supported runtime shape")
	registry := newTestDataModelRegistry(t, t.TempDir())
	for _, name := range []string{
		"node-local-v2-lpu-only", "node-local-v2-hybrid", "node-local-v2-specdecode",
		"node-local-v3-hx-lpu-only", "node-local-v3-hx-hybrid", "node-local-v3-hx-specdecode",
	} {
		t.Run(name, func(t *testing.T) {
			t.Log("Resolve authored inputs with shared and role-specific metadata")
			payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/" + name + ".input.yaml")
			require.NoError(t, err)
			dgd := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, yaml.Unmarshal(payload, dgd))
			dgd.Spec.Labels = map[string]string{"shared-label": "source"}
			dgd.Spec.Annotations = map[string]string{"shared-annotation": "source"}
			for _, component := range lpx.Components(dgd) {
				for _, role := range component.Roles {
					metav1.SetMetaDataLabel(&role.PodTemplate.ObjectMeta, "shared-label", "role")
				}
			}
			child := newLPXRenderDeployment(t, dgd)
			before, childBefore := dgd.DeepCopy(), child.DeepCopy()
			r := &graphReconciler{
				config: &configv1alpha1.OperatorConfiguration{
					Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
				},
				runtimeConfig: &commoncontroller.RuntimeConfig{}, modelRegistry: registry,
			}
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			plan := plans["lpu"]
			planBefore := plan.ForReplica(plan.ReplicaIndex)

			t.Log("Render twice and require identical independently owned output")
			first, firstResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
			require.NoError(t, err)
			second, secondResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
			require.NoError(t, err)
			require.Equal(t, first, second)
			require.Equal(t, firstResources, secondResources)
			secondBefore := second.DeepCopy()
			resourcesBefore := make([]client.Object, len(secondResources))
			for i, resource := range secondResources {
				resourcesBefore[i] = resource.DeepCopyObject().(client.Object)
			}

			t.Log("Mutate rendered Pods, metadata and resources without changing other output or inputs")
			first.Labels["shared-label"] = mutatedValue
			first.Annotations["shared-annotation"] = mutatedValue
			for _, clique := range first.Spec.Template.Cliques {
				clique.Spec.PodSpec.Containers[0].Image = "rendered:changed"
				clique.Labels["shared-label"] = mutatedValue
				clique.Annotations["shared-annotation"] = mutatedValue
			}
			for _, group := range first.Spec.Template.PodCliqueScalingGroupConfigs {
				group.Annotations[lpx.WorkloadDigestAnnotation] = mutatedValue
			}
			for _, resource := range firstResources {
				resource.GetAnnotations()[deploymentUIDAnnotation] = mutatedValue
				switch resource := resource.(type) {
				case *corev1.ConfigMap:
					resource.Data["rendered-only"] = mutatedValue
				case *corev1.Service:
					resource.Spec.Selector[dynamo.LPXServingLabel] = mutatedValue
				}
			}
			require.Equal(t, secondBefore, second)
			require.Equal(t, resourcesBefore, secondResources)
			require.Equal(t, before, dgd)
			require.Equal(t, childBefore, child)
			require.Equal(t, planBefore, plan)
		})
	}
}

func TestLPXRenderingMetadata(t *testing.T) {
	const restartToken = "2026-09-08T00:00:00Z"
	t.Log("Author scheduler metadata and stale identity annotations before freezing the child")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.Labels = map[string]string{"project": "inference", "unrelated": "ignored"}
	dgd.Annotations["kai.scheduler/topology"] = "source-topology"
	dgd.Spec.Annotations = map[string]string{
		"kai.scheduler/topology": "explicit-topology",
		lpx.DGDUIDAnnotation:     "stale", lpx.DGDGenerationAnnotation: "stale",
	}
	child.Annotations[dynamo.LPXRestartAnnotation] = restartToken
	dgd.Generation++
	r := newLPXTestReconciler(t, registry, child, dgd)
	r.config.Discovery.Backend = configv1alpha1.DiscoveryBackendKubernetes
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)

	t.Log("Inherit scheduler metadata while explicit PCS metadata takes precedence")
	require.Equal(t, "inference", pcs.Labels["project"])
	require.NotContains(t, pcs.Labels, "unrelated")
	require.Equal(t, "explicit-topology", pcs.Annotations["kai.scheduler/topology"])
	require.NotContains(t, pcs.Annotations, consts.RestartAnnotation)

	t.Log("Stamp the frozen source identity everywhere and the restart token only on Pods")
	annotationMaps := []map[string]string{pcs.Annotations}
	for _, clique := range pcs.Spec.Template.Cliques {
		require.Equal(t, restartToken, clique.Annotations[consts.RestartAnnotation])
		annotationMaps = append(annotationMaps, clique.Annotations)
	}
	for _, group := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		require.NotContains(t, group.Annotations, consts.RestartAnnotation)
		annotationMaps = append(annotationMaps, group.Annotations)
	}
	for _, resource := range resources {
		require.NotContains(t, resource.GetAnnotations(), consts.RestartAnnotation)
		require.Equal(t, string(child.UID), resource.GetLabels()[deploymentUIDLabel])
		annotationMaps = append(annotationMaps, resource.GetAnnotations())
	}
	for _, annotations := range annotationMaps {
		require.Equal(t, string(metav1.GetControllerOf(child).UID), annotations[lpx.DGDUIDAnnotation])
		require.Equal(t, string(child.UID), annotations[deploymentUIDAnnotation])
		require.NotContains(t, annotations, lpx.DGDGenerationAnnotation)
		require.NotContains(t, annotations, "lpx.nvidia.com/deployment-generation")
		require.NotContains(t, annotations, "lpx.nvidia.com/input-revision")
	}
}

func TestLPXReplicaChangesPreserveRenderedTemplates(t *testing.T) {
	t.Log("Render a hybrid workload at its initial capacity")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	before, beforeResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)

	t.Log("Changing only workload capacity preserves templates, configuration and the graph digest")
	dgd.Spec.Components[0].Replicas = ptr.To(int32(12))
	workloads, plans, err = r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	after, afterResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans)
	require.NoError(t, err)
	require.Equal(t, before, after)
	require.Equal(t, beforeResources, afterResources)
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
	target := dgd.GetComponentByName("lpu")
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
	selected, err := lpx.ResolveWorkload(t.Context(), dgd, singleGroupComponents(t, dgd), registry)
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)))
	require.NoError(t, err)

	t.Log("Render one shared conductor without changing either component's Agent template")
	r := &graphReconciler{config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
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

		selected, err := lpx.ResolveWorkload(t.Context(), &deployment, singleGroupComponents(t, &deployment), registry)
		require.NoError(t, err)
		child := newLPXRenderDeployment(t, &deployment)
		plan := mustPlanSelectedLPX(t, &deployment, selected)
		r := &graphReconciler{config: controllerConfig, runtimeConfig: &commoncontroller.RuntimeConfig{}}
		pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, &deployment, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan})
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

func mustPlanSelectedLPX(t *testing.T, dgd *v1beta1.DynamoGraphDeployment, selected *lpx.Workload) *lpx.MaterializationPlan {
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
	rendered, _, err := reconciler.renderPodCliqueSet(ctx, deployment, dgd, map[string]*lpx.Workload{desired.workload.ServingComponentName(): desired.workload}, map[string]*lpx.MaterializationPlan{desired.workload.ServingComponentName(): desired.plan})
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

// singleGroupComponents reads the one admitted workload in a test fixture.
func singleGroupComponents(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) []string {
	t.Helper()
	groups := lpx.ComponentGroups(dgd)
	require.Len(t, groups, 1)
	for _, components := range groups {
		return components
	}
	return nil
}

func TestHybridWorkloadUsesLPXSchedulerWithKaiEnabled(t *testing.T) {
	t.Log("Resolve a hybrid workload with KAI integration enabled")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineLPX)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, KaiScheduler: true}}

	t.Log("Use the LPX scheduler for every role and leave queue selection to the PCS")
	rendered, err := dynamo.RenderLPXWorkloadTemplates(dgd, r.config, runtimeConfig, nil, workloads["lpx"], plans["lpx"])
	require.NoError(t, err)
	require.NotEmpty(t, rendered.Cliques)
	for _, clique := range rendered.Cliques {
		require.Equal(t, lpx.SchedulerName, clique.Spec.PodSpec.SchedulerName)
		require.NotContains(t, clique.Labels, consts.KubeLabelKaiSchedulerQueue, "clique %s", clique.Name)
	}
}
