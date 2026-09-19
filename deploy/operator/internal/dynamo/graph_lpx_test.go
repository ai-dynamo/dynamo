// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"os"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

func TestBackendDetectionSkipsNonWorkers(t *testing.T) {
	for _, componentType := range []string{commonconsts.ComponentTypeLPX, commonconsts.ComponentTypeFrontend, commonconsts.ComponentTypePlanner} {
		t.Run(componentType, func(t *testing.T) {
			t.Log("Ignore ambiguous GPU commands and explicit backends for non-workers")
			backend, err := determineBackendFramework(componentType, nil,
				[]string{"python -m dynamo.vllm; python -m dynamo.sglang"}, "trtllm")
			require.NoError(t, err)
			require.Equal(t, BackendFrameworkNoop, backend)
		})
	}
}

func TestRenderSelectedLPXRolePodSettings(t *testing.T) {
	tests := []struct {
		name             string
		authored         *corev1.PodSecurityContext
		sharedMemorySize *resource.Quantity
		volumes          []corev1.Volume
		mounts           []corev1.VolumeMount
	}{
		{
			name: "no authored security context",
		},
		{
			name:     "authored fsGroup without policy",
			authored: &corev1.PodSecurityContext{FSGroup: ptr.To(int64(2000))},
		},
		{
			name: "authored always policy",
			authored: &corev1.PodSecurityContext{
				FSGroup:             ptr.To(int64(2000)),
				FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeAlways),
			},
		},
		{
			name:     "authored empty security context",
			authored: &corev1.PodSecurityContext{},
		},
		{
			name: "authored root security context",
			authored: &corev1.PodSecurityContext{
				RunAsUser: ptr.To(int64(0)), RunAsGroup: ptr.To(int64(0)), RunAsNonRoot: ptr.To(false),
			},
		},
		{
			name: "explicit shared memory size", sharedMemorySize: ptr.To(resource.MustParse("64Mi")),
			volumes: []corev1.Volume{{Name: "shared-memory", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium: corev1.StorageMediumMemory, SizeLimit: ptr.To(resource.MustParse("64Mi")),
			}}}},
			mounts: []corev1.VolumeMount{{Name: "shared-memory", MountPath: "/dev/shm"}},
		},
		{name: "explicit zero shared memory size", sharedMemorySize: ptr.To(resource.MustParse("0"))},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author a Cyborg role using the shared LPX role base renderer")
			source := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "hybrid", Namespace: "test"},
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "engine", ComponentType: v1beta1.ComponentTypeDecode, SharedMemorySize: test.sharedMemorySize,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:      []corev1.Container{{Name: "main", Image: "cyborg:test"}},
					SecurityContext: test.authored,
				}},
			}

			t.Log("Render the role before runtime-specific lowering")
			template, err := renderSelectedLPXRole(component, source, nil,
				&configv1alpha1.OperatorConfiguration{}, &mockSecretsRetriever{}, DiscoveryContext{},
				&podTemplateRuntimeDefaults{ComponentDefaults: NewWorkerDefaults()})
			require.NoError(t, err)

			t.Log("Retain authored security and use the ordinary shared memory default unless resized or disabled")
			if test.sharedMemorySize == nil {
				test.volumes = []corev1.Volume{{Name: "shared-memory", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{
					Medium: corev1.StorageMediumMemory, SizeLimit: ptr.To(resource.MustParse("8Gi")),
				}}}}
				test.mounts = []corev1.VolumeMount{{Name: "shared-memory", MountPath: "/dev/shm"}}
			}
			require.Equal(t, test.authored, template.Spec.SecurityContext)
			require.ElementsMatch(t, test.volumes, template.Spec.Volumes)
			require.ElementsMatch(t, test.mounts, template.Spec.Containers[0].VolumeMounts)
		})
	}
}

func TestLPXPCSNameUsesStableMaterializationIdentity(t *testing.T) {
	deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "chat", Namespace: "workloads", UID: "materialization-uid"}}
	name := PCSNameForLPX(deployment)
	t.Log("Keep the deployment name visible with four hexadecimal identity characters")
	require.Equal(t, "chat-3efb", name)
	t.Log("An ordinary DGD named chat-lpx must not collide with the LPX PCS of chat")
	require.NotEqual(t, PCSNameForDGD("chat-lpx", nil), name)
	require.NotEqual(t, PCSNameForDGD(deployment.Name, nil), name)

	for _, test := range []struct {
		name   string
		mutate func(*v1alpha1.LPXGraphDeployment)
	}{
		{"replacement materialization", func(d *v1alpha1.LPXGraphDeployment) { d.UID = "replacement-materialization-uid" }},
		{"different namespace", func(d *v1alpha1.LPXGraphDeployment) { d.Namespace = "another-namespace" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := deployment.DeepCopy()
			test.mutate(changed)
			require.NotEqual(t, name, PCSNameForLPX(changed))
		})
	}
	t.Log("Materialization bookkeeping must not rename the workload")
	deployment.Annotations = map[string]string{"unrelated": "changed"}
	deployment.Generation++
	deployment.Spec.InputRevision = "sha256:changed"
	require.Equal(t, name, PCSNameForLPX(deployment))

	deployment.Name = "chat.example"
	dotted := PCSNameForLPX(deployment)
	deployment.Name = "chat-example"
	require.NotEqual(t, dotted, PCSNameForLPX(deployment), "normalizing dots must not erase materialization identity")

	t.Log("Names sharing the same visible prefix still identify different deployments")
	deployment.Name = strings.Repeat("a", 33) + "-first"
	first := PCSNameForLPX(deployment)
	deployment.Name = strings.Repeat("a", 33) + "-second"
	require.NotEqual(t, first, PCSNameForLPX(deployment))
}

func TestLPXPCSNamePreservesReadablePrefixWithinGroveBudget(t *testing.T) {
	for _, test := range []struct {
		name   string
		prefix string
	}{
		{name: "c", prefix: "c"},
		{name: "gpt-oss-120b-production", prefix: "gpt-oss-120b-production"},
		{name: "test-models-llama3-8b-lp20", prefix: "test-models-llama3-8b-lp20"},
		{name: "test-models-gpt-oss-20b-lp20-b300", prefix: "test-models-gpt-oss-20b-lp20-b300"},
		{name: "chat.example", prefix: "chat-example"},
		{name: "120b-chat", prefix: "lpx-120b-chat"},
		{name: strings.Repeat("a", 33), prefix: strings.Repeat("a", 33)},
		{name: strings.Repeat("a", 32) + "-chat", prefix: strings.Repeat("a", 32)},
		{name: strings.Repeat("long.", 40) + "chat", prefix: "long-long-long-long-long-long-lon"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Retain the readable deployment prefix while bounding the complete Grove name")
			deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{
				Name: test.name, Namespace: "workloads", UID: "materialization-uid",
			}}
			got := PCSNameForLPX(deployment)
			require.True(t, strings.HasPrefix(got, test.prefix+"-"), got)
			require.Len(t, got, len(test.prefix)+5)
			require.Regexp(t, `-[a-f0-9]{4}$`, got)
			require.LessOrEqual(t, len(got), lpx.MaxPodCliqueSetNameLength)
			require.LessOrEqual(t, len(got)+len("lpx")+len("cond"), commonconsts.MaxCombinedGroveResourceNameLength)
			require.Empty(t, validation.IsDNS1035Label(got+"-serve"), got)
			require.Equal(t, got, PCSNameForLPX(deployment))
		})
	}
}

func TestLPXInputRevision(t *testing.T) {
	const updatedMetadata = "updated"
	t.Log("Create a DGD with independent prefill and LPX engine capacity")
	source := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "hybrid", Namespace: "test", UID: "source", Generation: 1},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			{ComponentName: "decode", ComponentType: v1beta1.ComponentTypeLPX, Replicas: ptr.To(int32(2)), LPX: &v1beta1.LPXConfig{BuildID: "hybrid-build"},
				Roles: []v1beta1.ComponentRoleSpec{
					{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: &corev1.PodTemplateSpec{}},
					{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: &corev1.PodTemplateSpec{}},
				}},
		}},
	}
	deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: source.Name, Namespace: source.Namespace, UID: "materialization"}}
	want, err := LPXInputRevision(source, "")
	require.NoError(t, err)
	require.Regexp(t, `^sha256:[a-f0-9]{64}$`, want)
	pcsName := PCSNameForLPX(deployment)
	require.NotEqual(t, PCSNameForDGD(source.Name, source.Spec.Components), pcsName)

	t.Log("Prefill-only edits and DGD bookkeeping do not alter the LPX revision or PCS name")
	source.Generation++
	source.ResourceVersion = "2"
	source.Labels = map[string]string{"unrelated": "metadata"}
	source.Annotations = map[string]string{"unrelated": "bookkeeping"}
	source.Spec.Components[0].ComponentName = "a-long-independent-prefill"
	source.Spec.Components[0].Replicas = ptr.To(int32(5))
	source.Spec.Restart = &v1beta1.Restart{ID: "not-yet-selected"}
	got, err := LPXInputRevision(source, "")
	require.NoError(t, err)
	require.Equal(t, want, got)
	require.Equal(t, pcsName, PCSNameForLPX(deployment))

	t.Log("Each LPX component or shared-input change invalidates the revision; ordinary edits do not")
	source.Spec.Scheduling = &v1beta1.SchedulingSpec{}
	want, err = LPXInputRevision(source, "")
	require.NoError(t, err)
	for _, test := range []struct {
		name       string
		wantChange bool
		mutate     func(*v1beta1.DynamoGraphDeployment)
	}{
		{"component/name", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).ComponentName = "new-engine" }},
		{"component/replicas", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).Replicas = ptr.To(int32(3)) }},
		{"component/min-available", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).MinAvailable = ptr.To(int32(1)) }},
		{"component/namespace", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).GlobalDynamoNamespace = true }},
		{"component/runtime-version", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).RuntimeVersionOverride = "1.5.0" }},
		{"component/shared-memory", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).SharedMemorySize = ptr.To(resource.MustParse("16Gi"))
		}},
		{"component/model", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ModelRef = &v1beta1.ModelReference{Name: "model", Revision: "2"}
		}},
		{"component/compilation-cache", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).CompilationCache = &v1beta1.CompilationCacheConfig{PVCName: "cache"}
		}},
		{"component/topology", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).TopologyConstraint = &v1beta1.TopologyConstraint{PackDomain: "rack"}
		}},
		{"component/build", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).LPX.BuildID = "next-build" }},
		{"agent/replicas", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = ptr.To(int32(4))
		}},
		{"conductor/replicas", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = ptr.To(int32(2))
		}},
		{"agent/image", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.Containers = []corev1.Container{{Name: "main", Image: "agent:next"}}
		}},
		{"conductor/image", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec.Containers = []corev1.Container{{Name: "main", Image: "conductor:next"}}
		}},
		{"agent/placement", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Spec.NodeSelector = map[string]string{"lpu": "new"}
		}},
		{"conductor/placement", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec.NodeSelector = map[string]string{"gpu": "new"}
		}},
		{"agent/metadata", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXAgent).PodTemplate.Labels = map[string]string{"role": "new"}
		}},
		{"conductor/metadata", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Annotations = map[string]string{"role": "new"}
		}},
		{"scheduling/deadline", true, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Scheduling.AttemptDeadlineSeconds = ptr.To(int64(60))
		}},
		{"shared/labels", true, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.Labels = map[string]string{"workload": "new"} }},
		{"shared/annotations", true, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.Annotations = map[string]string{"workload": "new"} }},
		{"shared/environment", true, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.Env = []corev1.EnvVar{{Name: "SHARED", Value: "new"}} }},
		{"shared/priority", true, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.PriorityClassName = "inference" }},
		{"shared/backend", true, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.BackendFramework = "vllm" }},
		{"shared/topology", true, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.TopologyConstraint = &v1beta1.SpecTopologyConstraint{ClusterTopologyName: "cluster", PackDomain: "rack"}
		}},
		{"shared/provider-override", true, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.ProviderOverride = &v1beta1.ProviderOverride{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Target: "PodCliqueSet", Value: apiextensionsv1.JSON{Raw: []byte(`{"spec":{"template":{"topologyConstraint":{"packDomain":"rack"}}}}`)}}
		}},
		{"restart/selected-lpx", true, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Restart = &v1beta1.Restart{ID: "restart-selected"}
			d.Status.Restart = &v1beta1.RestartStatus{ObservedID: d.Spec.Restart.ID, Phase: v1beta1.RestartPhaseRestarting, InProgress: []string{"decode"}}
		}},
		{"ignored/ordinary-replicas", false, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.Components[0].Replicas = ptr.To(int32(9)) }},
		{"ignored/ordinary-image", false, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Components[0].PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "prefill:next"}}}}
		}},
		{"ignored/ordinary-restart", false, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Restart = &v1beta1.Restart{ID: "restart-ordinary"}
			d.Status.Restart = &v1beta1.RestartStatus{ObservedID: d.Spec.Restart.ID, Phase: v1beta1.RestartPhaseRestarting, InProgress: []string{d.Spec.Components[0].ComponentName}}
		}},
		{"ignored/unselected-restart", false, func(d *v1beta1.DynamoGraphDeployment) { d.Spec.Restart = &v1beta1.Restart{ID: "not-delivered"} }},
		{"ignored/generation", false, func(d *v1beta1.DynamoGraphDeployment) { d.Generation++ }},
		{"ignored/resource-version", false, func(d *v1beta1.DynamoGraphDeployment) { d.ResourceVersion = "999" }},
		{"ignored/status", false, func(d *v1beta1.DynamoGraphDeployment) { d.Status.ObservedGeneration = d.Generation }},
		{"ignored/labels", false, func(d *v1beta1.DynamoGraphDeployment) { d.Labels["unrelated"] = updatedMetadata }},
		{"ignored/annotations", false, func(d *v1beta1.DynamoGraphDeployment) { d.Annotations["unrelated"] = updatedMetadata }},
		{"ignored/component-order", false, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Components[0], d.Spec.Components[1] = d.Spec.Components[1], d.Spec.Components[0]
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			changed := source.DeepCopy()
			test.mutate(changed)
			got, err := LPXInputRevision(changed, LPXRestartToken(changed, ""))
			require.NoError(t, err)
			require.Equal(t, test.wantChange, got != want)
		})
	}

	t.Log("Every consumed metadata key is an input, including both queue APIs and scheduler inheritance")
	for _, key := range []string{
		commonconsts.KubeAnnotationEnableMetrics, commonconsts.KubeAnnotationDynamoDiscoveryBackend,
		commonconsts.KubeAnnotationDynamoKubeDiscoveryMode, commonconsts.KubeAnnotationDynamoOperatorOriginVersion,
		commonconsts.KubeAnnotationVLLMDistributedExecutorBackend, commonconsts.KubeAnnotationLPXSchedulerBackend,
		commonconsts.KubeAnnotationWorkloadProvider, commonconsts.KubeAnnotationGroveUpdateStrategy,
		commonconsts.KubeAnnotationKaiSchedulerQueue, commonconsts.KubeAnnotationVolcanoQueue,
		"kai.scheduler/topology", "priorityClassName", "project", "user",
	} {
		t.Run("metadata/"+key, func(t *testing.T) {
			changed := source.DeepCopy()
			changed.Annotations[key] = updatedMetadata
			got, err := LPXInputRevision(changed, "")
			require.NoError(t, err)
			require.NotEqual(t, want, got)
		})
	}
	for _, key := range []string{"kai.scheduler/preemptibility", "priorityClassName", "project", "user"} {
		t.Run("scheduler-label/"+key, func(t *testing.T) {
			changed := source.DeepCopy()
			changed.Labels[key] = updatedMetadata
			got, err := LPXInputRevision(changed, "")
			require.NoError(t, err)
			require.NotEqual(t, want, got)
		})
	}

	t.Log("Both members independently contribute to the shared LPX revision")
	pair := source.DeepCopy()
	target := lpx.ServingComponent(pair)
	target.Replicas = ptr.To(int32(1))
	draft := target.DeepCopy()
	draft.ComponentName = "small-model"
	draft.Roles = []v1beta1.ComponentRoleSpec{*draft.ComponentRole(v1beta1.ComponentRoleLPXAgent)}
	pair.Spec.Components = append(pair.Spec.Components, *draft)
	pairRevision, err := LPXInputRevision(pair, "")
	require.NoError(t, err)
	for _, name := range []string{draft.ComponentName, target.ComponentName} {
		changed := pair.DeepCopy()
		changed.GetComponentByName(name).LPX.BuildID = "next-build"
		after, err := LPXInputRevision(changed, "")
		require.NoError(t, err)
		require.NotEqual(t, pairRevision, after, name)
	}
	got, err = LPXInputRevision(source, "selected-restart")
	require.NoError(t, err)
	require.NotEqual(t, want, got)
}

func TestLPXInputRevisionTracksIndirectRenderMetadata(t *testing.T) {
	for _, change := range []string{"epp-presence", "alpha-label", "alpha-annotation", "alpha-subtype"} {
		t.Run(change, func(t *testing.T) {
			t.Log("Capture the GPU role metadata derived from the complete source graph")
			payload, err := os.ReadFile("lpx/testdata/from_dgd_yaml/node-local-v2-hybrid.input.yaml")
			require.NoError(t, err)
			source := &v1beta1.DynamoGraphDeployment{}
			require.NoError(t, yaml.Unmarshal(payload, source))
			component := lpx.ServingComponent(source)
			role := lpxRoleComponent(component, component.ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate, source, "")
			role.ComponentType = v1beta1.ComponentTypeDecode
			metadata := generatePodMetadata(role, source, getDGDAlphaComponent(source, component.ComponentName), component.ComponentName, DiscoveryContext{})
			before, err := LPXInputRevision(source, "")
			require.NoError(t, err)

			t.Log("Change an indirect render input without changing the native LPX component")
			changed := source.DeepCopy()
			if change == "epp-presence" {
				changed.Spec.Components = append(changed.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "epp", ComponentType: v1beta1.ComponentTypeEPP})
			} else {
				alpha := &v1alpha1.DynamoGraphDeployment{}
				require.NoError(t, alpha.ConvertFrom(source))
				legacy := alpha.Spec.Services[component.ComponentName]
				switch change {
				case "alpha-label":
					legacy.Labels = map[string]string{"render-input": "new"}
				case "alpha-annotation":
					legacy.Annotations = map[string]string{"render-input": "new"}
				case "alpha-subtype":
					legacy.SubComponentType = "custom-lpx"
				}
				require.NoError(t, alpha.ConvertTo(changed))
			}
			require.Equal(t, component, lpx.ServingComponent(changed))
			nextMetadata := generatePodMetadata(role, changed, getDGDAlphaComponent(changed, component.ComponentName), component.ComponentName, DiscoveryContext{})
			if change == "alpha-annotation" {
				require.NotEqual(t, metadata.Annotations, nextMetadata.Annotations)
			} else {
				require.NotEqual(t, metadata.Labels, nextMetadata.Labels)
			}
			after, err := LPXInputRevision(changed, "")
			require.NoError(t, err)
			require.NotEqual(t, before, after, "different rendered metadata requires a different child revision")
		})
	}
}

func TestLPXRestartTokenPreservesDeliveredTokenOnCancellation(t *testing.T) {
	t.Log("A cancelled sequential restart must not select an LPX component that never started")
	source := &v1beta1.DynamoGraphDeployment{
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend},
			{ComponentName: "lpx", ComponentType: v1beta1.ComponentTypeLPX},
		}},
		Status: v1beta1.DynamoGraphDeploymentStatus{Restart: &v1beta1.RestartStatus{
			ObservedID: "cancelled-restart", Phase: v1beta1.RestartPhaseRestarting, InProgress: []string{"frontend"},
		}},
	}
	for _, request := range []*v1beta1.Restart{nil, {}} {
		source.Spec.Restart = request
		for _, delivered := range []string{"", "previously-delivered"} {
			require.Equal(t, delivered, LPXRestartToken(source, delivered))
		}
	}
}

func TestLPXInputRevisionIgnoresUnrelatedConvertedMetadata(t *testing.T) {
	t.Log("Keep EPP discovery enabled while editing only its independent workload")
	payload, err := os.ReadFile("lpx/testdata/from_dgd_yaml/node-local-v2-hybrid.input.yaml")
	require.NoError(t, err)
	source := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, yaml.Unmarshal(payload, source))
	source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "epp", ComponentType: v1beta1.ComponentTypeEPP, Replicas: ptr.To(int32(1)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "epp:before"}}}},
	})
	before, err := LPXInputRevision(source, "")
	require.NoError(t, err)
	source.Spec.Components[1].Replicas = ptr.To(int32(3))
	source.Spec.Components[1].PodTemplate.Spec.Containers[0].Image = "epp:after"
	after, err := LPXInputRevision(source, "")
	require.NoError(t, err)
	require.Equal(t, before, after)

	t.Log("Ignore conversion bookkeeping and metadata on the ordinary component")
	alpha := &v1alpha1.DynamoGraphDeployment{}
	require.NoError(t, alpha.ConvertFrom(source))
	alpha.Spec.Services["epp"].Labels = map[string]string{"ordinary-label": "changed"}
	alpha.Spec.Services["epp"].Annotations = map[string]string{"ordinary-annotation": "changed"}
	alpha.Spec.Services["epp"].SubComponentType = "ordinary-subtype"
	converted := &v1beta1.DynamoGraphDeployment{}
	require.NoError(t, alpha.ConvertTo(converted))
	require.NotEqual(t, source.Annotations, converted.Annotations)
	after, err = LPXInputRevision(converted, "")
	require.NoError(t, err)
	require.Equal(t, before, after)
}
