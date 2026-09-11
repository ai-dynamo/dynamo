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

func TestRenderSelectedLPXRoleSecurityContext(t *testing.T) {
	tests := []struct {
		name     string
		authored *corev1.PodSecurityContext
		expected *corev1.PodSecurityContext
	}{
		{
			name: "no authored security context",
			expected: &corev1.PodSecurityContext{
				FSGroup:             ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
				FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeOnRootMismatch),
			},
		},
		{
			name:     "authored fsGroup without policy",
			authored: &corev1.PodSecurityContext{FSGroup: ptr.To(int64(2000))},
			expected: &corev1.PodSecurityContext{FSGroup: ptr.To(int64(2000))},
		},
		{
			name: "authored always policy",
			authored: &corev1.PodSecurityContext{
				FSGroup:             ptr.To(int64(2000)),
				FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeAlways),
			},
			expected: &corev1.PodSecurityContext{
				FSGroup:             ptr.To(int64(2000)),
				FSGroupChangePolicy: ptr.To(corev1.FSGroupChangeAlways),
			},
		},
		{
			name:     "authored empty security context",
			authored: &corev1.PodSecurityContext{},
			expected: &corev1.PodSecurityContext{},
		},
		{
			name: "authored root security context",
			authored: &corev1.PodSecurityContext{
				RunAsUser: ptr.To(int64(0)), RunAsGroup: ptr.To(int64(0)), RunAsNonRoot: ptr.To(false),
			},
			expected: &corev1.PodSecurityContext{
				RunAsUser: ptr.To(int64(0)), RunAsGroup: ptr.To(int64(0)), RunAsNonRoot: ptr.To(false),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author a Cyborg role using the shared LPX role base renderer")
			source := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "hybrid", Namespace: "test"},
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "engine", ComponentType: v1beta1.ComponentTypeDecode,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:      []corev1.Container{{Name: "main", Image: "cyborg:test"}},
					SecurityContext: test.authored,
				}},
			}

			t.Log("Render the role before runtime-specific lowering")
			template, err := renderSelectedLPXRole(component, source, nil,
				&configv1alpha1.OperatorConfiguration{}, &mockSecretsRetriever{}, DiscoveryContext{},
				&lpx.SelectedWorkload{}, &imageEntrypointComponentDefaults{ComponentDefaults: NewWorkerDefaults()}, nil)
			require.NoError(t, err)

			t.Log("Retain authored security context and default only an absent one")
			require.Equal(t, test.expected, template.Spec.SecurityContext)
		})
	}
}

func TestLPXPCSNameUsesStableMaterializationIdentity(t *testing.T) {
	source := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "chat", Namespace: "workloads", UID: "chat-uid"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "engine", ComponentType: v1beta1.ComponentTypeLPX,
				Roles: []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXConductor}, {Name: v1beta1.ComponentRoleLPXAgent}}},
		}},
	}
	deployment := &v1alpha1.LPXGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "chat", Namespace: "workloads", UID: "materialization-uid"}}
	name := PCSNameForLPX(deployment, source)
	t.Log("An ordinary DGD named chat-lpx must not collide with the LPX PCS of chat")
	ordinary := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "chat-lpx"}}
	require.NotEqual(t, PCSNameForDGD(ordinary.Name, ordinary.Spec.Components), name)
	require.NotEqual(t, PCSNameForDGD(source.Name, source.Spec.Components), name)

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
			require.NotEqual(t, name, PCSNameForLPX(changed, source))
		})
	}
	t.Log("Source replacement, graph edits, and materialization bookkeeping must not rename the workload")
	source.UID = "replacement-source"
	source.Spec.Components[0].Replicas = ptr.To(int32(3))
	deployment.Generation++
	deployment.Spec.InputRevision = "sha256:changed"
	require.Equal(t, name, PCSNameForLPX(deployment, source))
	t.Log("A draft name cannot change the target-owned resource name budget")
	source.Spec.Components = append(source.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: strings.Repeat("d", 30), ComponentType: v1beta1.ComponentTypeLPX,
		Roles: []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXAgent}},
	})
	require.Equal(t, name, PCSNameForLPX(deployment, source))

	deployment.Name = "chat.example"
	dotted := PCSNameForLPX(deployment, source)
	deployment.Name = "chat-example"
	require.NotEqual(t, dotted, PCSNameForLPX(deployment, source), "normalizing dots must not erase materialization identity")

	t.Log("Both readable and compact names fit the complete Grove name budget")
	for componentLength := 1; componentLength <= 13; componentLength++ {
		source.Spec.Components[0].ComponentName = strings.Repeat("e", componentLength)
		for _, materializationName := range []string{"c", "chat.example", strings.Repeat("long.", 40) + "chat"} {
			deployment.Name = materializationName
			got := PCSNameForLPX(deployment, source)
			require.Empty(t, validation.IsDNS1123Label(got), got)
			require.LessOrEqual(t, len(got)+max(componentLength, 8)+componentLength+len("-engine-gpu"), commonconsts.MaxCombinedGroveResourceNameLength)
			require.Equal(t, got, PCSNameForLPX(deployment, source))
		}
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
	pcsName := PCSNameForLPX(deployment, source)
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
	require.Equal(t, pcsName, PCSNameForLPX(deployment, source))

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
		{"component/merge-strategy", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ExtraPodSpecMergeStrategy = v1beta1.ExtraPodSpecMergeStrategyOverride
		}},
		{"component/shared-memory", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).SharedMemorySize = ptr.To(resource.MustParse("16Gi"))
		}},
		{"component/model", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).ModelRef = &v1beta1.ModelReference{Name: "model", Revision: "2"}
		}},
		{"component/compilation-cache", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).CompilationCache = &v1beta1.CompilationCacheConfig{PVCName: "cache"}
		}},
		{"component/build", true, func(d *v1beta1.DynamoGraphDeployment) { lpx.ServingComponent(d).LPX.BuildID = "next-build" }},
		{"component/settings", true, func(d *v1beta1.DynamoGraphDeployment) {
			lpx.ServingComponent(d).LPX.Settings = &apiextensionsv1.JSON{Raw: []byte(`{"prop_sync":false}`)}
		}},
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
		{"shared/kv-transfer", true, func(d *v1beta1.DynamoGraphDeployment) {
			d.Spec.Experimental = &v1beta1.DynamoGraphDeploymentExperimentalSpec{KvTransferPolicy: &v1beta1.KvTransferPolicy{LabelKey: "topology.kubernetes.io/zone", Domain: "zone"}}
		}},
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
			payload, err := os.ReadFile("testdata/from_dgd_yaml/node-local-v2-hybrid.input.yaml")
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
	payload, err := os.ReadFile("testdata/from_dgd_yaml/node-local-v2-hybrid.input.yaml")
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
