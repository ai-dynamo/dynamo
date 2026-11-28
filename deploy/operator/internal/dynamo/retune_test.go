// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/components/planner"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRetuneRenderingAcrossProviders(t *testing.T) {
	for _, origin := range []string{"1.7.0", "1.8.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Select the Planner and frontend by type, not component names")
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", Spec: v1beta1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Env: []corev1.EnvVar{
					{Name: "DYN_REQUEST_TRACE", Value: "0"},
					{Name: "DYN_PLANNER_RETUNE_CAPABILITY", Value: "wrong-general-binding"},
					{Name: "DYN_ROUTER_MODE", Value: "round-robin"},
				},
				Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: "main", Image: "frontend:1.7.0", Env: []corev1.EnvVar{
							{Name: "DYN_REQUEST_TRACE", Value: "0"},
							{Name: "DYN_REQUEST_TRACE_FILE_PATH", Value: "wrong"},
							{Name: "DYN_ROUTER_MODE", Value: "round-robin"},
							{Name: "UNRELATED", Value: "retained"},
						}}},
					}}},
					{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
						Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"},
							Router: &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(0)), PrefillLoadScale: new(float64(2)), Temperature: new(float64(0.5))}},
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "planner:1.7.0", Env: []corev1.EnvVar{{Name: "DYN_PLANNER_RETUNE_CAPABILITY", Value: "authored-wrong-binding"}}}}}},
					},
				},
			}}
			if origin != "" {
				dgd.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin}
			}
			original := dgd.DeepCopy()
			config := &configv1alpha1.OperatorConfiguration{}

			t.Log("Grove and generated DCDs retain the same operator-owned wiring")
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
			require.NoError(t, err)
			for _, name := range []string{"gateway", "optimizer"} {
				child := children[name]
				child.OwnerReferences = []metav1.OwnerReference{{APIVersion: "nvidia.com/v1beta1", Kind: "DynamoGraphDeployment", Name: dgd.Name, Controller: new(true)}}
				childPod, err := renderStoredEnvironment(child, config, RoleMain)
				require.NoError(t, err)
				clique := cliqueByName(pcs, name)
				require.NotNil(t, clique)
				for _, pod := range []*corev1.PodSpec{childPod, &clique.Spec.PodSpec} {
					main := GetMainContainerFromPodSpec(pod)
					require.NotNil(t, main)
					require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "dynamo-retune-traces", MountPath: "/trace"})
					require.Contains(t, pod.Volumes, corev1.Volume{Name: "dynamo-retune-traces", PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "traces"}})
					if name != "gateway" {
						binding := findEnvVar(main.Env, "DYN_PLANNER_RETUNE_CAPABILITY")
						require.NotNil(t, binding)
						var capability planner.RetuneLaunchConfiguration
						require.NoError(t, json.Unmarshal([]byte(binding.Value), &capability))
						_, expected, err := planner.ResolveRetuneConfiguration(dgd)
						require.NoError(t, err)
						require.Equal(t, *expected, capability)
						continue
					}
					require.Nil(t, findEnvVar(main.Env, "DYN_FRONTEND_RETUNE_CAPABILITY"))

					t.Log("The Pod-name binding expands correctly even in legacy sorted environments")
					env := slices.Clone(main.Env)
					podNameCount := 0
					for i := range env {
						if env[i].Name == "DYN_POD_NAME" {
							podNameCount++
							require.NotNil(t, env[i].ValueFrom)
							require.NotNil(t, env[i].ValueFrom.FieldRef)
							podName := env[i].ValueFrom.FieldRef.FieldPath
							require.Equal(t, "metadata.name", podName)
							// Model the kubelet resolving fieldRef before sequential expansion.
							env[i] = corev1.EnvVar{Name: "DYN_POD_NAME", Value: "gateway-pod"}
						}
					}
					require.Equal(t, 1, podNameCount)
					values := semanticenv.Replay(env)
					require.Equal(t, semanticenv.Value{Known: true, Value: "kv"}, values["DYN_ROUTER_MODE"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "true"}, values["DYN_ROUTER_REPLICA_SYNC"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "0"}, values["DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "2"}, values["DYN_ROUTER_PREFILL_LOAD_SCALE"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "0.5"}, values["DYN_ROUTER_TEMPERATURE"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "/trace/segments/graph/gateway-pod/req"}, values["DYN_REQUEST_TRACE_FILE_PATH"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "1"}, values["DYN_REQUEST_TRACE"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "8002"}, values["DYN_FRONTEND_ADMIN_PORT"])
					require.Equal(t, semanticenv.Value{Known: true, Value: "retained"}, values["UNRELATED"])
					require.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_FRONTEND_ADMIN_HOST", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}}})
				}
				require.Equal(t, childPod.Containers[0].Env, clique.Spec.PodSpec.Containers[0].Env)
			}
			require.Equal(t, original, dgd)
		})
	}
}

func TestRetuneLoweringsPrepareMissingContainers(t *testing.T) {
	for _, test := range []struct {
		name     string
		template *corev1.PodTemplateSpec
	}{
		{name: "omitted template"},
		{name: "empty template", template: &corev1.PodTemplateSpec{}},
		{name: "sidecar only", template: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar", Image: "sidecar:1"}}}}},
		{name: "main after sidecar", template: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar", Image: "sidecar:1"}, {Name: "main", Image: "authored:1"}}}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Retune preserves authored container order while lowerings supply missing structure")
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", Spec: v1beta1.DynamoGraphDeploymentSpec{
				BackendFramework: "vllm",
				Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
					{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend, PodTemplate: test.template.DeepCopy()},
					{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner, PodTemplate: test.template.DeepCopy(),
						Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
				},
			}}
			original := dgd.DeepCopy()
			children, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
			require.NoError(t, err)
			pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
			require.NoError(t, err)
			for _, name := range []string{"gateway", "optimizer"} {
				for index, pod := range []*corev1.PodSpec{&children[name].Spec.PodTemplate.Spec, &cliqueByName(pcs, name).Spec.PodSpec} {
					main := GetMainContainerFromPodSpec(pod)
					require.NotNil(t, main)
					require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "dynamo-retune-traces", MountPath: "/trace"})
					if test.template != nil && len(test.template.Spec.Containers) > 0 {
						require.Contains(t, pod.Containers, test.template.Spec.Containers[0])
						if len(test.template.Spec.Containers) == 2 {
							if index == 0 {
								require.Equal(t, "sidecar", pod.Containers[0].Name)
							} else {
								require.Equal(t, "main", pod.Containers[0].Name)
							}
							require.Equal(t, "authored:1", main.Image)
						} else {
							require.Equal(t, "main", pod.Containers[0].Name)
						}
					}
				}
			}
			require.Equal(t, original, dgd)
		})
	}
}

func TestRetuneCompilationCacheMountOwnership(t *testing.T) {
	for _, componentName := range []string{"gateway", "optimizer"} {
		for _, test := range []struct {
			name      string
			mountPath string
			conflicts bool
		}{
			{name: "trace root", mountPath: "/trace", conflicts: true},
			{name: "trace segments", mountPath: "/trace/segments", conflicts: true},
			{name: "normalized nested trace path", mountPath: "/trace/segments/../windows/replay", conflicts: true},
			{name: "separate cache", mountPath: "/cache"},
			{name: "unrelated prefix", mountPath: "/trace-cache"},
		} {
			t.Run(componentName+"/"+test.name, func(t *testing.T) {
				t.Log("Compilation-cache intent cannot introduce a mount inside the owned trace tree")
				dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
						{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
							Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
					},
				}}
				dgd.GetComponentByName(componentName).CompilationCache = &v1beta1.CompilationCacheConfig{PVCName: "compile-cache", MountPath: test.mountPath}
				original := dgd.DeepCopy()

				t.Run("generated DCD normalization", func(t *testing.T) {
					t.Log("Reject conflicting intent before it is persisted into a child DCD")
					_, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{})
					if test.conflicts {
						require.ErrorContains(t, err, "compilationCache.mountPath")
					} else {
						require.NoError(t, err)
					}
				})

				t.Run("direct Grove rendering", func(t *testing.T) {
					t.Log("Apply the same ownership rule when rendering Pods directly")
					_, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
					if test.conflicts {
						require.ErrorContains(t, err, "compilationCache.mountPath")
					} else {
						require.NoError(t, err)
					}
				})
				require.Equal(t, original, dgd)
			})
		}
	}
}

func TestRetuneLeavesWorkerHashAndDisabledRenderingUnchanged(t *testing.T) {
	t.Log("Only frontend/Planner wiring changes; worker generation must remain stable")
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
		{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner},
		{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}}}}}},
	}}}
	wantHash, err := ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	config := &configv1alpha1.OperatorConfiguration{}
	baseline, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
	require.NoError(t, err)
	for _, test := range []struct {
		name    string
		enabled bool
	}{{name: "disabled"}, {name: "enabled", enabled: true}} {
		t.Run(test.name, func(t *testing.T) {
			candidate := dgd.DeepCopy()
			candidate.Spec.Components[1].Retune = &v1beta1.RetuneSpec{Enabled: test.enabled, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}
			hash, err := ComputeDGDWorkersSpecHash(candidate)
			require.NoError(t, err)
			require.Equal(t, wantHash, hash)
			rendered, err := GenerateGrovePodCliqueSet(t.Context(), candidate, nil, config, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
			require.NoError(t, err)
			require.Equal(t, cliqueByName(baseline, "worker"), cliqueByName(rendered, "worker"))
			plannerContainer := GetMainContainerFromPodSpec(&cliqueByName(rendered, "optimizer").Spec.PodSpec)
			require.Equal(t, test.enabled, findEnvVar(plannerContainer.Env, "DYN_PLANNER_RETUNE_CAPABILITY") != nil)
			if !test.enabled {
				require.Equal(t, baseline, rendered)
			} else {
				require.NotEqual(t, cliqueByName(baseline, "gateway"), cliqueByName(rendered, "gateway"))
				require.NotEqual(t, cliqueByName(baseline, "optimizer"), cliqueByName(rendered, "optimizer"))

				t.Log("A PVC change changes the Planner launch binding as well as its mount")
				oldBinding := findEnvVar(plannerContainer.Env, "DYN_PLANNER_RETUNE_CAPABILITY").Value
				candidate.Spec.Components[1].Retune.TraceStorage.ClaimName = "new-traces"
				updated, err := GenerateGrovePodCliqueSet(t.Context(), candidate, nil, config, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
				require.NoError(t, err)
				updatedPlanner := GetMainContainerFromPodSpec(&cliqueByName(updated, "optimizer").Spec.PodSpec)
				newBinding := findEnvVar(updatedPlanner.Env, "DYN_PLANNER_RETUNE_CAPABILITY")
				require.NotNil(t, newBinding)
				require.NotEqual(t, oldBinding, newBinding.Value)
				var capability planner.RetuneLaunchConfiguration
				require.NoError(t, json.Unmarshal([]byte(newBinding.Value), &capability))
				require.Equal(t, "new-traces", capability.TraceStorage.ClaimName)
			}
		})
	}
}
