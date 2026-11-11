// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/testfixture"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	"gotest.tools/v3/golden"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
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
		"from_dgd_yaml/units-qwen",
		"from_dgd_yaml/units-gpt-oss",

		"from_dgd_yaml/lpx_v2_vllm",
		"from_dgd_yaml/node-local-v2-lpu-only",
		"from_dgd_yaml/node-local-v2-hybrid",
		"from_dgd_yaml/node-local-v2-hybrid-all-local",
		"from_dgd_yaml/lpx-v2-local-partitions",
		"from_dgd_yaml/node-local-v3-hx-lpu-only",
		"from_dgd_yaml/node-local-v3-hx-specdecode",
		"from_dgd_yaml/node-local-v3-hx-agent-specdecode",
		"from_dgd_yaml/node-local-v3-hx-hybrid",
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
			r := &graphReconciler{Client: newLPXGPUClient(t), config: controllerConfig, runtimeConfig: runtimeConfig, modelRegistry: testdataModelRegistry}
			child := newLPXRenderDeployment(t, &dynamoDeployment)
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, &dynamoDeployment)
			require.NoError(t, err)
			got, extraResources, err := r.renderPodCliqueSet(t.Context(), child, &dynamoDeployment, workloads, plans, nil)
			require.NoError(t, err)
			servingHash, err := dynamo.ComputeDGDWorkersSpecHash(&dynamoDeployment)
			require.NoError(t, err)
			for _, clique := range got.Spec.Template.Cliques {
				require.Equal(t, v1alpha1.LPXSchedulerName, clique.Spec.PodSpec.SchedulerName)
				component := dynamoDeployment.GetComponentByName(clique.Labels[consts.KubeLabelDynamoComponent])
				require.NotNil(t, component)
				require.True(t, component.IsLPX())
				require.Equal(t, servingHash, clique.Labels[consts.KubeLabelDynamoWorkerHash])
				// A runtime sidecar keeps the ordinary worker namespace and serving suffix.
				if runtime := common.FindContainerByName(clique.Spec.PodSpec.InitContainers, consts.RuntimeContainerName); runtime != nil {
					require.Contains(t, runtime.Env, corev1.EnvVar{Name: consts.DynamoNamespaceWorkerSuffixEnvVar, Value: servingHash})
					require.Contains(t, runtime.Env, corev1.EnvVar{Name: consts.DynamoNamespaceEnvVar, Value: dynamoDeployment.GetDynamoNamespaceForComponent(component)})
					continue
				}
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: consts.DynamoNamespaceWorkerSuffixEnvVar})
				require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: consts.DynamoNamespaceEnvVar, Value: dynamoDeployment.GetDynamoNamespaceForComponent(component) + "-" + servingHash})
			}

			t.Log("Render ordinary components independently from their explicit selection")
			normal, err := dynamo.GenerateGrovePodCliqueSet(
				t.Context(), &dynamoDeployment, (*v1beta1.DynamoComponentDeploymentSharedSpec).ManagedByExternalController, controllerConfig, runtimeConfig,
				kubeClient, nil, nil, nil, true, nil,
			)
			require.NoError(t, err)
			require.NotEqual(t, normal.Name, got.Name)
			for _, clique := range normal.Spec.Template.Cliques {
				require.NotEqual(t, v1alpha1.LPXSchedulerName, clique.Spec.PodSpec.SchedulerName)
				component := dynamoDeployment.GetComponentByName(clique.Labels[consts.KubeLabelDynamoComponent])
				require.False(t, component.IsLPX())
				if dynamo.IsWorkerComponent(string(component.ComponentType)) {
					require.Equal(t, servingHash, clique.Labels[consts.KubeLabelDynamoWorkerHash])
					require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: consts.DynamoNamespaceWorkerSuffixEnvVar, Value: servingHash})
					require.Contains(t, clique.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: consts.DynamoNamespaceEnvVar, Value: dynamoDeployment.GetDynamoNamespaceForComponent(component)})
				}
			}
			podCliqueSets := []*grovev1alpha1.PodCliqueSet{got}
			if len(normal.Spec.Template.Cliques) > 0 {
				podCliqueSets = append(podCliqueSets, normal)
			}

			for _, pcs := range podCliqueSets {
				// Assert revision bindings above; keep compiler-layout goldens independent of serving hashes.
				for _, clique := range pcs.Spec.Template.Cliques {
					if clique.Labels[consts.KubeLabelDynamoWorkerHash] == "" {
						continue
					}
					delete(clique.Labels, consts.KubeLabelDynamoWorkerHash)
					container := &clique.Spec.PodSpec.Containers[0]
					if runtime := common.FindContainerByName(clique.Spec.PodSpec.InitContainers, consts.RuntimeContainerName); runtime != nil {
						container = runtime
					}
					container.Env = slices.DeleteFunc(container.Env, func(env corev1.EnvVar) bool { return env.Name == consts.DynamoNamespaceWorkerSuffixEnvVar })
					index := slices.IndexFunc(container.Env, func(env corev1.EnvVar) bool { return env.Name == consts.DynamoNamespaceEnvVar })
					container.Env[index].Value = clique.Labels[consts.KubeLabelDynamoNamespace]
				}
				sort.Slice(pcs.Spec.Template.Cliques, func(i, j int) bool {
					return pcs.Spec.Template.Cliques[i].Name < pcs.Spec.Template.Cliques[j].Name
				})
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

const lpxSidecarImage = "example.invalid/dynamo-sidecar:test"

// loadLPXInput reads an LPX testdata input DGD.
func loadLPXInput(t *testing.T, fixture string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	payload, err := os.ReadFile("../../dynamo/lpx/testdata/from_dgd_yaml/" + fixture + ".input.yaml")
	require.NoError(t, err)
	dgd := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, dgd))
	return dgd
}

// serveFromAgent replaces the conductor of an LPU-only "lpu" component with an
// Agent runtime sidecar; main runs a neutral agent, and both read the model path.
func serveFromAgent(dgd *v1beta1.DynamoGraphDeployment) {
	component := dgd.GetComponentByName("lpu")
	component.Roles = []v1beta1.ComponentRoleSpec{*component.ComponentRole(v1beta1.ComponentRoleLPXAgent)}
	agent := &component.Roles[0].PodTemplate.Spec
	agent.Containers[0].Command = []string{"/bin/agent"}
	agent.Containers[0].Env = []corev1.EnvVar{{Name: "LPU_IE_MODEL_PATH", Value: "$(LPX_MODEL_PATH)"}}
	agent.InitContainers = append(agent.InitContainers, corev1.Container{
		Name: consts.RuntimeContainerName, Image: lpxSidecarImage, RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		Args:         []string{"lpx", "--model-path", "$(LPX_MODEL_PATH)/tokenizer"},
		Env:          []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: "18090"}},
		VolumeMounts: []corev1.VolumeMount{{Name: "model-storage", MountPath: "/nfs", ReadOnly: true}},
		ReadinessProbe: &corev1.Probe{
			HTTPGet:       &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(18090)},
			PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
		},
	})
}

// sidecarPullSecret grants a pull secret only to the runtime sidecar image.
type sidecarPullSecret struct{}

func (sidecarPullSecret) GetSecrets(_, image string) ([]string, error) {
	if image != lpxSidecarImage {
		return nil, nil
	}
	return []string{"sidecar-pull-secret"}, nil
}

// renderLPXServing renders dgd with the golden test's infrastructure addresses
// and returns its cliques and the one clique that serves.
func renderLPXServing(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) ([]*grovev1alpha1.PodCliqueTemplateSpec, *grovev1alpha1.PodCliqueTemplateSpec) {
	t.Helper()
	config := &configv1alpha1.OperatorConfiguration{Infrastructure: configv1alpha1.InfrastructureConfiguration{ETCDAddress: "etcd-address", NATSAddress: "nats-address"}}
	r := &graphReconciler{
		Client: newLPXGPUClient(t), config: config, runtimeConfig: &commoncontroller.RuntimeConfig{},
		modelRegistry: newTestDataModelRegistry(t, t.TempDir()), dockerSecretRetriever: sidecarPullSecret{},
	}
	child := newLPXRenderDeployment(t, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	pcs, _, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	var serving *grovev1alpha1.PodCliqueTemplateSpec
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Labels[dynamo.LPXServingLabel] == consts.KubeLabelValueTrue {
			serving = clique
		}
	}
	require.NotNil(t, serving)
	return pcs.Spec.Template.Cliques, serving
}

// TestLPXAgentRuntimeSidecarRendering pins the runtime sidecar of an LPU-only
// model, whose Agent hosts the engine: it runs after the authored init
// containers with shared runtime env and an authored host-network readiness probe.
func TestLPXAgentRuntimeSidecarRendering(t *testing.T) {
	t.Log("Render an LPU-only model with graph env; its only clique, the Agent, serves")
	dgd := loadLPXInput(t, "node-local-v3-hx-lpu-only")
	serveFromAgent(dgd)
	dgd.Spec.Env = []corev1.EnvVar{{Name: "GRAPH_SETTING", Value: "shared"}}
	cliques, serving := renderLPXServing(t, dgd)
	require.Len(t, cliques, 1)
	require.Equal(t, "agt", serving.Name)

	t.Log("The sidecar stays after the authored init container, restarts as a sidecar, and brings its pull secret")
	pod := serving.Spec.PodSpec
	require.True(t, pod.HostNetwork)
	require.Len(t, pod.InitContainers, 2)
	require.Equal(t, "prepare-lp30", pod.InitContainers[0].Name)
	main, runtime := pod.Containers[0], pod.InitContainers[1]
	require.Equal(t, consts.RuntimeContainerName, runtime.Name)
	require.Nil(t, runtime.Command)
	require.Equal(t, ptr.To(corev1.ContainerRestartPolicyAlways), runtime.RestartPolicy)
	require.Equal(t, []corev1.LocalObjectReference{{Name: "sidecar-pull-secret"}}, pod.ImagePullSecrets)

	t.Log("Operator-owned bindings precede authored environment references")
	modelPath := corev1.EnvVar{Name: "LPX_MODEL_PATH", Value: "/nfs/node-local-v3-hx-connected"}
	require.Equal(t, modelPath, main.Env[1])
	require.Equal(t, modelPath, runtime.Env[0])

	t.Log("The LPU-only engine keeps authored env, graph env, model path and the explicit local-unit selection")
	require.Equal(t, []corev1.EnvVar{
		{Name: "LPX_LOCAL_PARTITION_IDS", Value: ""},
		modelPath,
		{Name: "GRAPH_SETTING", Value: "shared"},
		{Name: "LPU_IE_MODEL_PATH", Value: "$(LPX_MODEL_PATH)"},
	}, main.Env)

	t.Log("The sidecar keeps its runtime and graph env, including the authored system port")
	servingHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	namespace := dgd.GetDynamoNamespaceForComponent(dgd.GetComponentByName("lpu"))
	require.Equal(t, []corev1.EnvVar{
		modelPath,
		{Name: consts.DynamoComponentEnvVar, Value: "lpx"},
		{Name: consts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
		{Name: consts.DynamoNamespaceEnvVar, Value: namespace},
		{Name: consts.DynamoNamespaceWorkerSuffixEnvVar, Value: servingHash},
		{Name: "DYN_PARENT_DGD_K8S_NAME", Value: dgd.Name},
		{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: dgd.Namespace},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		{Name: "DYN_SYSTEM_PORT", Value: "18090"},
		{Name: "ETCD_ENDPOINTS", Value: "etcd-address"},
		{Name: "GRAPH_SETTING", Value: "shared"},
		{Name: "NATS_SERVER", Value: "nats-address"},
		{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
		{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
		{Name: "POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.uid"}}},
	}, runtime.Env)

	t.Log("The host-network sidecar declares no ports and keeps only its authored numeric readiness probe")
	require.Empty(t, runtime.Ports)
	require.Nil(t, runtime.StartupProbe)
	require.Nil(t, runtime.LivenessProbe)
	require.Equal(t, &corev1.Probe{
		HTTPGet:       &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(18090)},
		PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
	}, runtime.ReadinessProbe)
}

// TestLPXAgentRuntimeSidecarKeepsAuthoredSystemPort pins template ownership of a
// host-network sidecar's port and probes: authored ones that name the system port
// render as written.
func TestLPXAgentRuntimeSidecarKeepsAuthoredSystemPort(t *testing.T) {
	t.Log("Author the system port with HTTP readiness and TCP liveness on the Agent runtime sidecar")
	dgd := loadLPXInput(t, "node-local-v3-hx-lpu-only")
	serveFromAgent(dgd)
	authored := &dgd.GetComponentByName("lpu").Roles[0].PodTemplate.Spec.InitContainers[1]
	ports := []corev1.ContainerPort{{Name: consts.DynamoSystemPortName, ContainerPort: 18090, Protocol: corev1.ProtocolTCP}}
	readiness := &corev1.Probe{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromString(consts.DynamoSystemPortName)}}
	liveness := &corev1.Probe{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromString(consts.DynamoSystemPortName)}}
	authored.Ports, authored.ReadinessProbe, authored.LivenessProbe = ports, readiness, liveness

	t.Log("Rendering keeps the authored port and probes and adds no startup probe")
	_, serving := renderLPXServing(t, dgd)
	runtime := serving.Spec.PodSpec.InitContainers[1]
	require.Equal(t, consts.RuntimeContainerName, runtime.Name)
	require.Equal(t, ports, runtime.Ports)
	require.Equal(t, readiness, runtime.ReadinessProbe)
	require.Equal(t, liveness, runtime.LivenessProbe)
	require.Nil(t, runtime.StartupProbe)
}

// TestLPXConductorRuntimeSidecarRendering pins the runtime sidecar of a hybrid
// model, whose Cyborg conductor hosts the engine: the model path comes first,
// and sidecar mode keeps the serving suffix separate, as for an ordinary worker.
// TestNativeSidecarRendering owns the rest of the sidecar-mode output.
func TestLPXConductorRuntimeSidecarRendering(t *testing.T) {
	t.Log("Render a hybrid model whose conductor hosts the runtime sidecar")
	dgd := loadLPXInput(t, "node-local-v2-hybrid")
	_, serving := renderLPXServing(t, dgd)

	t.Log("The serving conductor's first init container is the sidecar")
	runtime := serving.Spec.PodSpec.InitContainers[0]
	require.Equal(t, consts.RuntimeContainerName, runtime.Name)

	t.Log("The model path comes first; only sidecar mode gives the sidecar the namespace and a separate serving suffix")
	require.Equal(t, corev1.EnvVar{Name: "LPX_MODEL_PATH", Value: "/nfs/node-local-v2-connected-lpx"}, runtime.Env[0])
	servingHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	namespace := dgd.GetDynamoNamespaceForComponent(dgd.GetComponentByName("lpu"))
	require.Contains(t, runtime.Env, corev1.EnvVar{Name: consts.DynamoNamespaceEnvVar, Value: namespace})
	require.Contains(t, runtime.Env, corev1.EnvVar{Name: consts.DynamoNamespaceWorkerSuffixEnvVar, Value: servingHash})

	t.Log("On the Pod network the sidecar keeps the system port and its default probes")
	require.Equal(t, []corev1.ContainerPort{{
		Name: consts.DynamoSystemPortName, ContainerPort: int32(consts.DynamoSystemPort), Protocol: corev1.ProtocolTCP,
	}}, runtime.Ports)
	require.NotNil(t, runtime.StartupProbe)
	require.NotNil(t, runtime.LivenessProbe)
	require.NotNil(t, runtime.ReadinessProbe)
}

// TestLPXHostNetworkConductorRuntimeSidecarRendering pins template ownership of a
// host-network hybrid conductor's sidecar ports and probes: defaults are dropped
// and only the authored numeric readiness probe remains.
func TestLPXHostNetworkConductorRuntimeSidecarRendering(t *testing.T) {
	t.Log("Render a hybrid model whose host-network conductor hosts the runtime sidecar")
	dgd := loadLPXInput(t, "node-local-v2-hybrid")
	conductor := &dgd.GetComponentByName("lpu").ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
	conductor.HostNetwork = true
	readiness := &corev1.Probe{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt(18090)}}
	conductor.InitContainers = []corev1.Container{{
		Name: consts.RuntimeContainerName, Image: lpxSidecarImage, RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		Args:           []string{"lpx", "--model-path", "$(LPX_MODEL_PATH)/tokenizer"},
		Env:            []corev1.EnvVar{{Name: "DYN_SYSTEM_PORT", Value: "18090"}},
		VolumeMounts:   []corev1.VolumeMount{{Name: "model-storage", MountPath: "/nfs", ReadOnly: true}},
		ReadinessProbe: readiness,
	}}
	_, serving := renderLPXServing(t, dgd)

	t.Log("The sidecar declares no ports and keeps only its authored readiness probe")
	runtime := serving.Spec.PodSpec.InitContainers[0]
	require.Equal(t, consts.RuntimeContainerName, runtime.Name)
	require.Empty(t, runtime.Ports)
	require.Nil(t, runtime.StartupProbe)
	require.Nil(t, runtime.LivenessProbe)
	require.Equal(t, readiness, runtime.ReadinessProbe)
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
		{name: "DGD enables hybrid discovery", pipeline: lpx.PipelineHybrid, override: configv1alpha1.DiscoveryBackendKubernetes, wantServices: true},
		{name: "DGD disables Kubernetes discovery", pipeline: lpx.PipelineSingle, backend: configv1alpha1.DiscoveryBackendKubernetes, override: configv1alpha1.DiscoveryBackendEtcd},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Resolve two workloads with authored metadata and the selected discovery backend")
			child, dgd, registry := newLPXTestDGD(t, tc.pipeline)
			dgd.Spec.Components[0].ModelRef = &v1beta1.ModelReference{Name: "test/model"}
			second := dgd.Spec.Components[0].DeepCopy()
			second.ComponentName = "second"
			dgd.Spec.Components = append(dgd.Spec.Components, *second)
			dgd.Spec.Env = []corev1.EnvVar{{Name: consts.DynamoNamespaceEnvVar, Value: "authored"}, {Name: consts.DynamoNamespaceWorkerSuffixEnvVar, Value: "custom"}}
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

			t.Log("Render optional Services without writing resources or mutating the DGD")
			pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
			require.NoError(t, err)
			require.Equal(t, before, dgd)
			serviceCount := 0
			for _, resource := range resources {
				switch resource.(type) {
				case *corev1.Service:
					serviceCount++
				}
			}
			if !tc.wantServices {
				require.Zero(t, serviceCount)
				return
			}
			require.Equal(t, len(plans), serviceCount)

			t.Log("An image update retains discovery selectors for both serving revisions")
			dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec.Containers[0].Image += "-next"
			nextPCS, nextResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
			require.NoError(t, err)

			t.Log("Only serving roles advertise model discovery")
			for index, clique := range pcs.Spec.Template.Cliques {
				require.Subset(t, clique.Spec.PodSpec.Containers[0].Env, dgd.Spec.Env)
				if clique.Labels[dynamo.LPXServingLabel] == consts.KubeLabelValueTrue {
					require.NotEqual(t, lpxv1alpha1.PodRoleAgent, clique.Annotations[lpxv1alpha1.PodRoleAnnotation])
					require.Equal(t, dynamo.HashModelName("test/model"), clique.Labels[consts.KubeLabelDynamoBaseModelHash])
					service := getResource[*corev1.Service](t, resources, plans[clique.Labels[consts.KubeLabelDynamoComponent]].ResourcePrefix+"-serve")
					require.NotEqual(t, clique.Labels[consts.KubeLabelDynamoWorkerHash], nextPCS.Spec.Template.Cliques[index].Labels[consts.KubeLabelDynamoWorkerHash])
					for _, revision := range []*grovev1alpha1.PodCliqueTemplateSpec{clique, nextPCS.Spec.Template.Cliques[index]} {
						require.True(t, labels.SelectorFromSet(service.Spec.Selector).Matches(labels.Merge(revision.Labels, map[string]string{grovecommon.LabelPartOfKey: pcs.Name})))
					}
				} else {
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
					require.NotContains(t, clique.Labels, consts.KubeLabelDynamoBaseModelHash)
				}
			}

			t.Log("Each Service carries discovery metadata and selects its own serving component and PCS")
			for groupName, plan := range plans {
				service := getResource[*corev1.Service](t, resources, plan.ResourcePrefix+"-serve")
				component := dgd.GetComponentByName(groupName)
				require.Equal(t, service.Spec.Selector, getResource[*corev1.Service](t, nextResources, service.Name).Spec.Selector)
				require.Equal(t, child.Namespace, service.Namespace)
				require.Equal(t, "inference", service.Labels["example.com/team"])
				require.Equal(t, map[string]string{
					"example.com/description":    "serving",
					lpx.DeploymentNameAnnotation: child.Name,
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
	selected, err := lpx.ResolveWorkload(t.Context(), newLPXGPUClient(t), dgd, singleGroupComponents(t, dgd), newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	config := &configv1alpha1.OperatorConfiguration{}
	runtimeConfig := &commoncontroller.RuntimeConfig{}
	r := &graphReconciler{Client: newLPXGPUClient(t), config: config, runtimeConfig: runtimeConfig}
	plan := mustPlanSelectedLPX(t, dgd, selected)
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan}, nil)
	require.NoError(t, err)
	serialized, err := json.Marshal(pcs)
	require.NoError(t, err)

	t.Log("Accept exactly one MiB including final identity, discovery and scheduler metadata")
	dgd.Annotations["kai.scheduler/padding"] = strings.Repeat("x", lpx.MaxRenderedPodCliqueSetBytes-len(serialized))
	pcs, _, err = r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan}, nil)
	require.NoError(t, err)
	serialized, err = json.Marshal(pcs)
	require.NoError(t, err)
	require.Len(t, serialized, lpx.MaxRenderedPodCliqueSetBytes)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Annotations, "kai.scheduler/padding")
	}

	t.Log("Reject one additional final-metadata byte as a selected-render failure before publication")
	dgd.Annotations["kai.scheduler/padding"] += "x"
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan}, nil)
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
		{Name: "runtime-config", MountPath: "/custom-config", ReadOnly: true},
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
				Name: "runtime-settings", Key: name,
			}},
		})
	}
	authoredConfig := corev1.Volume{
		Name:     "runtime-config",
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}
	leader.Spec.Volumes = append(leader.Spec.Volumes, authoredConfig)

	t.Log("Render the complete LPX workload with ordinary pod spec overrides")
	selected, err := lpx.ResolveWorkload(t.Context(), newLPXGPUClient(t), dgd, singleGroupComponents(t, dgd), newTestDataModelRegistry(t, t.TempDir()))
	require.NoError(t, err)
	plan := mustPlanSelectedLPX(t, dgd, selected)
	r := &graphReconciler{Client: newLPXGPUClient(t), config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan}, nil)
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
	require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "runtime-config", MountPath: "/custom-config", ReadOnly: true})
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
		"node-local-v2-lpu-only", "node-local-v2-hybrid",
		"node-local-v3-hx-lpu-only", "node-local-v3-hx-hybrid", "node-local-v3-hx-specdecode", "node-local-v3-hx-lpu-only served from the agent",
	} {
		t.Run(name, func(t *testing.T) {
			t.Log("Resolve authored inputs with shared and role-specific metadata")
			fixture, agentServed := strings.CutSuffix(name, " served from the agent")
			dgd := loadLPXInput(t, fixture)
			if agentServed {
				serveFromAgent(dgd)
			}
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
				Client: newLPXGPUClient(t),
				config: &configv1alpha1.OperatorConfiguration{
					Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendKubernetes},
				},
				runtimeConfig: &commoncontroller.RuntimeConfig{}, modelRegistry: registry,
			}
			workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
			require.NoError(t, err)
			plan := plans["lpu"]
			planBefore := *plan
			planBefore.Agents = slices.Clone(plan.Agents)

			t.Log("Render twice and require identical independently owned output")
			first, firstResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
			require.NoError(t, err)
			second, secondResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
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
				resource.GetAnnotations()[lpx.DeploymentNameAnnotation] = mutatedValue
				switch resource := resource.(type) {
				case *corev1.Service:
					resource.Spec.Selector[dynamo.LPXServingLabel] = mutatedValue
				}
			}
			require.Equal(t, secondBefore, second)
			require.Equal(t, resourcesBefore, secondResources)
			require.Equal(t, before, dgd)
			require.Equal(t, childBefore, child)
			require.Equal(t, planBefore, *plan)
		})
	}
}

func TestLPXRenderingMetadata(t *testing.T) {
	const restartToken = "2026-09-08T00:00:00Z"
	t.Log("Author scheduler metadata and stale identity annotations before freezing the child")
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	second := dgd.Spec.Components[0].DeepCopy()
	second.ComponentName = "independent"
	dgd.Spec.Components = append(dgd.Spec.Components, *second)
	dgd.Labels = map[string]string{"project": "inference", "unrelated": "ignored"}
	dgd.Annotations["kai.scheduler/topology"] = "source-topology"
	dgd.Spec.Annotations = map[string]string{
		"kai.scheduler/topology":     "explicit-topology",
		lpx.DeploymentNameAnnotation: "stale",
	}
	child.Annotations[dynamo.LPXRestartAnnotation] = restartToken
	dgd.Generation++
	r := newLPXTestReconciler(t, registry, child, dgd)
	r.config.Discovery.Backend = configv1alpha1.DiscoveryBackendKubernetes
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	require.Len(t, workloads, 2)
	pcs, resources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 2)

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
		require.Equal(t, child.Name, annotations[lpx.DeploymentNameAnnotation])
		require.NotContains(t, annotations, "lpx.nvidia.com/deployment-uid")
		require.NotContains(t, annotations, "scheduling.lpu.nvidia.com/dgd-uid")
		require.NotContains(t, annotations, "lpx.nvidia.com/deployment-generation")
		require.NotContains(t, annotations, "lpx.nvidia.com/input-revision")
	}
}

func TestLPXReplicaChangesUpdateCyborgTemplate(t *testing.T) {
	t.Log("Render omitted hybrid capacity for two I/O endpoints with two clients each")
	root := t.TempDir()
	const buildID = "split-io"
	writeTestGraphBuild(t, root, buildID, testXTGraphManifestCapnp(t, testXTGraphManifestFixture{
		partitionCount: 1, devicesPerNode: 8,
		units:             connectedXTUnits(1, 0),
		nonLPUDeviceTypes: []commoncapnpv1.DeviceType{commoncapnpv1.DeviceType_cuda},
		ioFPGACount:       2, ioFanoutFactor: 2,
	}))
	registry, err := lpx.NewModelRegistry(root, nil)
	require.NoError(t, err)
	dgd := loadTestDGD(t, lpx.PipelineHybrid, buildID)
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = nil
	child := newLPXTestDeployment(t, dgd)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	before, beforeResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	cyborgIndex := slices.IndexFunc(before.Spec.Template.Cliques, func(clique *grovev1alpha1.PodCliqueTemplateSpec) bool {
		return clique.Name == plans["lpx"].CyborgTemplate
	})
	require.NotEqual(t, -1, cyborgIndex)
	cyborg := before.Spec.Template.Cliques[cyborgIndex]
	require.EqualValues(t, 4, cyborg.Spec.Replicas)
	require.Equal(t, new(int32(1)), cyborg.Spec.MinAvailable)

	t.Log("Persist explicit Cyborg capacity while preserving the group seed, configuration and graph digest")
	dgd.Spec.Components[0].Replicas = new(int32(12))
	dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = new(int32(8))
	workloads, plans, err = r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	after, afterResources, err := r.renderPodCliqueSet(t.Context(), child, dgd, workloads, plans, nil)
	require.NoError(t, err)
	require.EqualValues(t, 8, after.Spec.Template.Cliques[cyborgIndex].Spec.Replicas)
	before.Spec.Template.Cliques[cyborgIndex].Spec.Replicas = 8
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
		agent.Spec.Containers[0].Command = []string{"/custom/agent"}
		agent.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "AGENT_ONLY", Value: "kept"}}
	}
	target := dgd.GetComponentByName("lpu")
	template := target.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	template.Spec.Containers[0].Image = "independent-conductor-runtime"
	template.Spec.Containers[0].Command = []string{"/custom/conductor"}
	template.Spec.Containers[0].Env = append(template.Spec.Containers[0].Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
	template.Labels = map[string]string{"owner": "explicit-conductor"}
	template.Spec.NodeSelector = map[string]string{"runtime-role": "conductor"}
	template.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
			MatchExpressions: []corev1.NodeSelectorRequirement{{Key: "pool", Operator: corev1.NodeSelectorOpIn, Values: []string{"conductor"}}},
		}}},
	}}
	before := dgd.DeepCopy()
	selected, err := lpx.ResolveWorkload(t.Context(), newLPXGPUClient(t), dgd, singleGroupComponents(t, dgd), registry)
	require.NoError(t, err)
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)), dgd.GetComponentByName(selected.ServingComponentName()))
	require.NoError(t, err)

	t.Log("Render one shared conductor without changing either component's Agent template")
	r := &graphReconciler{Client: newLPXGPUClient(t), config: &configv1alpha1.OperatorConfiguration{}, runtimeConfig: &commoncontroller.RuntimeConfig{}}
	pcs, _, err := r.renderPodCliqueSet(t.Context(), newLPXRenderDeployment(t, dgd), dgd, map[string]*lpx.Workload{selected.ServingComponentName(): selected}, map[string]*lpx.MaterializationPlan{selected.ServingComponentName(): plan}, nil)
	require.NoError(t, err)

	t.Log("Keep the conductor separate from the Agents and preserve each role's metadata and storage")
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
			require.Equal(t, []string{"/custom/conductor"}, container.Command)
			require.NotContains(t, container.Args, "--datacenter-config-filepath")
			require.Contains(t, container.Env, corev1.EnvVar{
				Name: "DRAFT_MODEL_PATH", Value: "$(LPX_DRAFT_MODEL_PATH)",
			})
			require.Equal(t, "explicit-conductor", clique.Labels["owner"])
			require.Equal(t, template.Spec.NodeSelector, clique.Spec.PodSpec.NodeSelector)
			require.Equal(t, template.Spec.Affinity, clique.Spec.PodSpec.Affinity)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
		} else {
			component := clique.Labels[consts.KubeLabelDynamoComponent]
			require.Equal(t, component+"-runtime", container.Image)
			require.Equal(t, []string{"/custom/agent"}, container.Command)
			require.Contains(t, container.Env, corev1.EnvVar{Name: "AGENT_ONLY", Value: "kept"})
			require.NotContains(t, container.Env, corev1.EnvVar{Name: "CONDUCTOR_ONLY", Value: "kept"})
		}
	}
	require.Equal(t, 1, conductors)
	require.Equal(t, before, dgd)
}

func getResource[T any](t *testing.T, resources []client.Object, name string) T {
	t.Helper()
	i := slices.IndexFunc(resources, func(resource client.Object) bool { return resource.GetName() == name })
	require.GreaterOrEqual(t, i, 0)
	return resources[i].(T)
}

func mustPlanSelectedLPX(t *testing.T, dgd *v1beta1.DynamoGraphDeployment, selected *lpx.Workload) *lpx.MaterializationPlan {
	t.Helper()
	plan, err := selected.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(newLPXRenderDeployment(t, dgd)), dgd.GetComponentByName(selected.ServingComponentName()))
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

	xtBuilds := map[string]testXTGraphManifestFixture{
		"node-local-v2-connected": {
			partitionCount: 2,
			devicesPerNode: 8,
			units:          connectedXTUnits(2, 0, 1),
		},
		"node-local-v2-connected-lpx": {
			partitionCount:    2,
			devicesPerNode:    8,
			units:             connectedXTUnits(2, 0, 1),
			nonLPUDeviceTypes: []commoncapnpv1.DeviceType{commoncapnpv1.DeviceType_cuda},
		},
		"llama3_2-1b-lpu-gpu-v2/build_0m851219t7py3mp8x1j5rg9j8c": {
			partitionCount:    17,
			units:             independentXTUnits(17),
			devicesPerNode:    8,
			nonLPUDeviceTypes: []commoncapnpv1.DeviceType{commoncapnpv1.DeviceType_cuda},
		},
	}
	for buildID, fixture := range xtBuilds {
		writeTestGraphBuild(t, registryRoot, buildID, testXTGraphManifestCapnp(t, fixture))
	}
	hxBuilds := map[string]testHXGraphManifestFixture{
		"node-local-v3-hx-connected": {units: singleHXUnit()},
		"node-local-v3-hx-connected-lpx": {units: singleHXUnit(),
			nonLPUDeviceTypes: []commoncapnpv1.DeviceType{commoncapnpv1.DeviceType_cuda},
		},
		"node-local-v3-hx-sd-draft":  {units: singleHXUnit()},
		"node-local-v3-hx-sd-target": {units: singleHXUnit()},
	}
	for buildID, fixture := range hxBuilds {
		writeTestGraphBuild(t, registryRoot, buildID, testHXGraphManifestCapnp(t, fixture))
	}
	for _, name := range []string{"qwen", "gpt-oss"} {
		writeUnitsGraphBuild(t, registryRoot, "units-"+name, name)
	}

	registry, err := lpx.NewModelRegistry(registryRoot, nil)
	require.NoError(t, err)
	return registry
}

func testXTGraphManifestCapnp(t *testing.T, fixture testXTGraphManifestFixture) []byte {
	t.Helper()
	partitions := make([]testfixture.Partition, 0, fixture.partitionCount+len(fixture.nonLPUDeviceTypes))
	for index := range fixture.partitionCount {
		partitions = append(partitions, testfixture.Partition{ID: uint32(index), DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: deploymentcapnpv1.LpuArchitecture_lp20, DevicesPerNode: fixture.devicesPerNode})
	}
	for index, deviceType := range fixture.nonLPUDeviceTypes {
		partitions = append(partitions, testfixture.Partition{ID: uint32(fixture.partitionCount + index), DeviceType: deviceType})
	}
	manifest := testfixture.NewDeployment(t, partitions, fixture.units, max(1, fixture.ioFPGACount), max(1, fixture.ioFanoutFactor))
	data, err := manifest.Message().Marshal()
	require.NoError(t, err)
	return data
}

type testXTGraphManifestFixture struct {
	partitionCount int

	devicesPerNode uint32
	units          []testfixture.Unit

	nonLPUDeviceTypes []commoncapnpv1.DeviceType
	ioFPGACount       uint32
	ioFanoutFactor    uint32
}

func testHXGraphManifestCapnp(t *testing.T, fixture testHXGraphManifestFixture) []byte {
	t.Helper()
	count, firstID := 1, uint32(1)
	if fixture.lpuPartitions != 0 {
		count, firstID = fixture.lpuPartitions, fixture.firstPartitionID
	}
	partitions := make([]testfixture.Partition, 0, count+len(fixture.nonLPUDeviceTypes))
	for index := range count {
		partitions = append(partitions, testfixture.Partition{ID: firstID + uint32(index), DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: deploymentcapnpv1.LpuArchitecture_lp30, DevicesPerNode: 16})
	}
	for index, deviceType := range fixture.nonLPUDeviceTypes {
		partitions = append(partitions, testfixture.Partition{ID: firstID + uint32(count+index), DeviceType: deviceType})
	}
	manifest := testfixture.NewDeployment(t, partitions, fixture.units, 1, 1)
	data, err := manifest.Message().Marshal()
	require.NoError(t, err)
	return data
}

type testHXGraphManifestFixture struct {
	units []testfixture.Unit

	nonLPUDeviceTypes []commoncapnpv1.DeviceType
	// lpuPartitions is the number of single-node LPU partitions; zero selects one
	// partition with ID 1 on a two-node deployment.
	lpuPartitions int
	// firstPartitionID is the first compiler ID when lpuPartitions is nonzero.
	firstPartitionID uint32
}

func writeTestGraphBuild(t *testing.T, registryRoot, buildID string, manifest []byte) {
	t.Helper()

	buildDir := filepath.Join(registryRoot, buildID)
	require.NoError(t, os.MkdirAll(buildDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "deployment.v1.capnp.bin"), manifest, 0o600))
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, "manifest.json"), []byte(`{}`), 0o600))
}

func renderLPXTestPodCliqueSet(
	t *testing.T, ctx context.Context, reconciler *graphReconciler,
	deployment *v1alpha1.LPXGraphDeployment, dgd *v1beta1.DynamoGraphDeployment, desired *lpxTestWorkload,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	rendered, _, err := reconciler.renderPodCliqueSet(ctx, deployment, dgd, map[string]*lpx.Workload{desired.workload.ServingComponentName(): desired.workload}, map[string]*lpx.MaterializationPlan{desired.workload.ServingComponentName(): desired.plan}, nil)
	require.NoError(t, err)
	require.NoError(t, reconciler.stampClientIdentity(ctx, dgd, rendered))
	return rendered
}

func newLPXTestRegistryWithPartitionsAndPipeline(
	t *testing.T,
	buildID string,
	partitionIDs []int,
	pipeline lpx.Pipeline,
) lpx.ModelRegistry {
	t.Helper()
	root := t.TempDir()
	writeLPXTestBuild(t, root, buildID, partitionIDs, pipeline)
	registryURL := (&url.URL{Scheme: lpx.BuildSchemeFile, Path: root}).String()
	registry, err := lpx.NewModelRegistry(registryURL, nil)
	require.NoError(t, err)
	return registry
}

func writeLPXTestBuild(t *testing.T, root, buildID string, partitionIDs []int, pipeline lpx.Pipeline) {
	t.Helper()
	require.NotEmpty(t, partitionIDs)
	ids := make([]uint32, len(partitionIDs))
	partitions := make([]testfixture.Partition, 0, len(partitionIDs)+1)
	for i, id := range partitionIDs {
		ids[i] = uint32(id)
		partitions = append(partitions, testfixture.Partition{ID: uint32(id), DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: deploymentcapnpv1.LpuArchitecture_lp20, DevicesPerNode: 8})
	}
	if pipeline == lpx.PipelineHybrid {
		partitions = append(partitions, testfixture.Partition{ID: uint32(len(partitionIDs)), DeviceType: commoncapnpv1.DeviceType_cuda})
	}
	manifest := testfixture.NewDeployment(t, partitions, connectedXTUnits(2, ids...), 1, 1)
	payload, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeTestGraphBuild(t, root, buildID, payload)
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
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)
	r := newLPXTestReconciler(t, registry, child, dgd)
	workloads, plans, err := r.resolveWorkloads(t.Context(), child, dgd)
	require.NoError(t, err)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, KaiScheduler: true}}

	t.Log("Use the LPX scheduler for every role and leave queue selection to the PCS")
	rendered, err := dynamo.RenderLPXWorkloadTemplates(dgd, r.config, runtimeConfig, nil, workloads["lpx"], plans["lpx"])
	require.NoError(t, err)
	require.NotEmpty(t, rendered.Cliques)
	for _, clique := range rendered.Cliques {
		require.Equal(t, v1alpha1.LPXSchedulerName, clique.Spec.PodSpec.SchedulerName)
		require.NotContains(t, clique.Labels, consts.KubeLabelKaiSchedulerQueue, "clique %s", clique.Name)
	}
}

func TestStampClientIdentity(t *testing.T) {
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	dgd.UID = "dgd-uid"
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)

	t.Log("The first render creates the DGD's key; agents get its certificate and mount, every clique the UID")
	pcs := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)
	secret := &corev1.Secret{}
	require.NoError(t, r.Get(ctx, client.ObjectKey{Namespace: dgd.Namespace, Name: "lpx-id-dgd-uid"}, secret))
	require.True(t, metav1.IsControlledBy(secret, dgd))
	require.True(t, ptr.Deref(secret.Immutable, false))
	require.NotEmpty(t, secret.Data["private.key"])
	require.NotEmpty(t, secret.Data["private.key.pub"])
	block, _ := pem.Decode(secret.Data["client.pem"])
	require.NotNil(t, block)
	marker := fmt.Sprintf("lpx-id-dgd-uid/%x", sha256.Sum256(block.Bytes))
	agents := 0
	for _, clique := range pcs.Spec.Template.Cliques {
		require.Equal(t, "dgd-uid", clique.Labels[consts.KubeLabelDynamoGraphDeploymentUID])
		require.Equal(t, marker, clique.Annotations[clientIdentityAnnotation], clique.Name)
		mounted := slices.ContainsFunc(clique.Spec.PodSpec.Containers, func(c corev1.Container) bool {
			return slices.ContainsFunc(c.VolumeMounts, func(m corev1.VolumeMount) bool { return m.MountPath == "/etc/lpx/identity" })
		})
		if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] != lpxv1alpha1.PodRoleAgent {
			require.False(t, mounted, clique.Name)
			continue
		}
		agents++
		require.True(t, mounted, clique.Name)
		require.Equal(t, base64.StdEncoding.EncodeToString(block.Bytes), clique.Annotations[clientCertAnnotation])
	}
	require.Positive(t, agents)
	for _, group := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		require.Equal(t, marker, group.Annotations[clientIdentityAnnotation])
	}

	t.Log("Later renders reuse the key, so the rollout marker stays put")
	again := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)
	require.Equal(t, pcs.Spec.Template.Cliques, again.Spec.Template.Cliques)
	require.Equal(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, again.Spec.Template.PodCliqueScalingGroupConfigs)

	t.Log("References to the cluster-wide MPI SSH Secret get the DGD's own key pair; template items still resolve")
	config := *r.config
	config.MPI.SSHSecretName = "ssh-secret" // the fixtures' templates mount this Secret
	r.config = &config
	fresh, _, err := r.renderPodCliqueSet(ctx, child, dgd, map[string]*lpx.Workload{desired.workload.ServingComponentName(): desired.workload}, map[string]*lpx.MaterializationPlan{desired.workload.ServingComponentName(): desired.plan}, nil)
	require.NoError(t, err)
	custom := []corev1.KeyToPath{{Key: "private.key.pub", Path: "authorized_keys"}}
	spec := &fresh.Spec.Template.Cliques[0].Spec.PodSpec
	spec.Volumes = append(spec.Volumes, corev1.Volume{Name: "ssh-projected",
		Projected: &corev1.ProjectedVolumeSource{Sources: []corev1.VolumeProjection{{Secret: &corev1.SecretProjection{
			Name: "ssh-secret", Items: custom}}}}})
	require.NoError(t, r.stampClientIdentity(ctx, dgd, fresh))
	plain := 0
	for _, clique := range fresh.Spec.Template.Cliques {
		for _, volume := range clique.Spec.PodSpec.Volumes {
			if volume.Name == "ssh-secret" {
				plain++
				require.Equal(t, "lpx-id-dgd-uid", volume.Secret.SecretName, clique.Name)
				require.Equal(t, []corev1.KeyToPath{{Key: "private.key", Path: "private.key"}, {Key: "private.key.pub", Path: "private.key.pub"}},
					volume.Secret.Items, clique.Name)
			}
		}
	}
	require.Positive(t, plain)
	projected := spec.Volumes[len(spec.Volumes)-1].Projected.Sources[0].Secret
	require.Equal(t, "lpx-id-dgd-uid", projected.Name)
	require.Equal(t, custom, projected.Items)

	t.Log("A Secret the DGD does not control is never adopted")
	secret.OwnerReferences = nil
	require.NoError(t, r.Update(ctx, secret))
	require.ErrorContains(t, r.stampClientIdentity(ctx, dgd, again), "is not controlled by this DGD")
}

func TestRecreatedClientIdentityRollsEveryClique(t *testing.T) {
	t.Log("Render a running workload with its client identity")
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	live := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)

	t.Log("The Secret is deleted while the workload runs; the next render generates new keys under the same name")
	require.NoError(t, r.Delete(ctx, &corev1.Secret{Namespace: dgd.Namespace, Name: "lpx-id-" + string(dgd.UID)}))
	recreated := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)

	t.Log("Every clique template changes, so Grove rolls each Pod without replacing the PodCliqueSet")
	for i, clique := range recreated.Spec.Template.Cliques {
		require.NotEqual(t, live.Spec.Template.Cliques[i].Annotations[clientIdentityAnnotation], clique.Annotations[clientIdentityAnnotation], clique.Name)
	}
	require.True(t, podCliqueSetLayoutMatches(live, recreated))
}

func TestClientIdentityRejectsAuthoredVolumeOrMount(t *testing.T) {
	for name, author := range map[string]func(*corev1.PodSpec){
		"volume": func(spec *corev1.PodSpec) {
			spec.Volumes = append(spec.Volumes, corev1.Volume{Name: identityVolume, EmptyDir: &corev1.EmptyDirVolumeSource{}})
		},
		"mount": func(spec *corev1.PodSpec) {
			spec.Volumes = append(spec.Volumes, corev1.Volume{Name: "authored", EmptyDir: &corev1.EmptyDirVolumeSource{}})
			spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, corev1.VolumeMount{Name: "authored", MountPath: identityMountPath + "/"})
		},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			author(&dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec)
			r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			name := desired.workload.ServingComponentName()
			pcs, _, err := r.renderPodCliqueSet(ctx, child, dgd, map[string]*lpx.Workload{name: desired.workload}, map[string]*lpx.MaterializationPlan{name: desired.plan}, nil)
			require.NoError(t, err)
			require.ErrorContains(t, r.stampClientIdentity(ctx, dgd, pcs), "LPX agent template")
		})
	}
}

func TestClientIdentityMountsOnlyAgentMainContainer(t *testing.T) {
	for name, sidecars := range map[string][]corev1.Container{
		"main only": nil,
		// A sidecar named after the role must not take main's identity.
		"sidecar named agent": {{Name: v1beta1.ComponentRoleLPXAgent, Image: "sidecar"}},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			child, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			spec := &dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec
			spec.Containers = append(spec.Containers, sidecars...)
			r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)

			t.Log("Render the PodCliqueSet and stamp the DGD's client identity")
			pcs := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)

			t.Log("Every agent mounts the identity into its main container and nowhere else")
			mount := corev1.VolumeMount{Name: identityVolume, MountPath: identityMountPath, ReadOnly: true}
			agents := 0
			for _, clique := range pcs.Spec.Template.Cliques {
				if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] != lpxv1alpha1.PodRoleAgent {
					continue
				}
				agents++
				require.Len(t, clique.Spec.PodSpec.Containers, 1+len(sidecars), clique.Name)
				var mounted []string
				for _, container := range clique.Spec.PodSpec.Containers {
					if slices.Contains(container.VolumeMounts, mount) {
						mounted = append(mounted, container.Name)
					}
				}
				require.Equal(t, []string{consts.MainContainerName}, mounted, clique.Name)
			}
			require.Positive(t, agents)
		})
	}
}

func TestOnlyLPXAgentsPublishACertificate(t *testing.T) {
	ctx := t.Context()
	child, dgd, registry := newLPXTestDGD(t, lpx.PipelineHybrid)

	t.Log("A hybrid DGD authors a certificate on its GPU worker")
	const forged = "Zm9yZ2Vk"
	template := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate
	if template.Annotations == nil {
		template.Annotations = map[string]string{}
	}
	template.Annotations[clientCertAnnotation] = forged
	r, desired := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
	pcs := renderLPXTestPodCliqueSet(t, ctx, r, child, dgd, desired)

	t.Log("Agents publish the operator's certificate; every other clique publishes none")
	others := 0
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] == lpxv1alpha1.PodRoleAgent {
			require.NotEqual(t, forged, clique.Annotations[clientCertAnnotation], clique.Name)
			continue
		}
		others++
		require.NotContains(t, clique.Annotations, clientCertAnnotation, clique.Name)
	}
	require.Positive(t, others)
}

// newLPXGPUClient supplies the GPU claim referenced by the authored render fixtures.
func newLPXGPUClient(t *testing.T) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, resourcev1.AddToScheme(scheme))
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		&resourcev1.DeviceClass{Name: "gpu.nvidia.com", Spec: resourcev1.DeviceClassSpec{Selectors: []resourcev1.DeviceSelector{{CEL: &resourcev1.CELDeviceSelector{Expression: `device.driver == "gpu.nvidia.com"`}}}}},
		&resourcev1.ResourceClaimTemplate{Name: "decode-gpu", Namespace: "test", Spec: resourcev1.ResourceClaimTemplateSpec{Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{Name: "gpu", Exactly: &resourcev1.ExactDeviceRequest{DeviceClassName: "gpu.nvidia.com", AllocationMode: resourcev1.DeviceAllocationModeExactCount, Count: 1}}}}}}},
	).Build()
}

func writeUnitsGraphBuild(t *testing.T, root, buildID, shape string) {
	t.Helper()
	width, count := uint32(16), 1
	extra := commoncapnpv1.DeviceType_lpu
	switch shape {
	case "qwen":
		count, extra = 2, commoncapnpv1.DeviceType_cpu
	case "gpt-oss":
		width, count, extra = 8, 14, commoncapnpv1.DeviceType_cuda
	}
	partitions := make([]testfixture.Partition, 0, count+1)
	architecture := deploymentcapnpv1.LpuArchitecture_lp20
	if width == 16 {
		architecture = deploymentcapnpv1.LpuArchitecture_lp30
	}
	for i := range count {
		partitions = append(partitions, testfixture.Partition{ID: uint32(i), DeviceType: commoncapnpv1.DeviceType_lpu, Architecture: architecture, DevicesPerNode: width})
	}
	if extra != commoncapnpv1.DeviceType_lpu {
		partitions = append(partitions, testfixture.Partition{DeviceType: extra})
	}
	groups := make([][][]uint32, count)
	for i := range count {
		groups[i] = [][]uint32{{uint32(i)}}
	}
	switch shape {
	case "qwen":
		groups = [][][]uint32{{{0}, {1}}}
	case "gpt-oss":
		groups = append(groups[:12], [][]uint32{{12}, {13}})
	}
	units := make([]testfixture.Unit, len(groups))
	for i, group := range groups {
		for _, members := range group {
			reservation := testfixture.Reservation{IDs: members, Nodes: 1}
			if width == 16 {
				reservation.HX = []uint8{1, 1, 1}
			}
			units[i].Reservations = append(units[i].Reservations, reservation)
		}
		for j := 1; j < len(group); j++ {
			connector := testfixture.Connector{HX: width == 16}
			if connector.HX {
				for device := range width {
					connector.Pairs = append(connector.Pairs, [2]uint32{device, device})
				}
			}
			units[i].Connectors = append(units[i].Connectors, connector)
		}
	}
	manifest := testfixture.NewDeployment(t, partitions, units, 1, 1)
	checkpoint, err := manifest.NewCheckpoint()
	require.NoError(t, err)
	checkpoint.SetProvider(deploymentcapnpv1.CheckpointProvider_huggingface)
	require.NoError(t, checkpoint.SetModel(checkpointTestModel))
	require.NoError(t, checkpoint.SetRevision(checkpointTestRevision))
	data, err := manifest.Message().Marshal()
	require.NoError(t, err)
	writeTestGraphBuild(t, root, buildID, data)
}

func connectedXTUnits(hosts uint32, ids ...uint32) []testfixture.Unit {
	unit := testfixture.Unit{}
	for i, id := range ids {
		unit.Reservations = append(unit.Reservations, testfixture.Reservation{IDs: []uint32{id}, Nodes: hosts})
		if i > 0 {
			unit.Connectors = append(unit.Connectors, testfixture.Connector{})
		}
	}
	return []testfixture.Unit{unit}
}
func independentXTUnits(count int) []testfixture.Unit {
	units := make([]testfixture.Unit, count)
	for i := range count {
		units[i] = connectedXTUnits(1, uint32(i))[0]
	}
	return units
}
func singleHXUnit() []testfixture.Unit {
	return []testfixture.Unit{{Reservations: []testfixture.Reservation{{IDs: []uint32{1}, HX: []uint8{1, 1, 1}}}}}
}
