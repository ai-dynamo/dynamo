// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"math"
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestRetuneSemanticBinding(t *testing.T) {
	t.Log("Compile one versioned contract from typed inputs and graph identity")
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", UID: "graph-uid", Generation: 7, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
		{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
			Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
	}}}
	_, baseline, err := RetuneConfiguration(dgd)
	require.NoError(t, err)
	require.Len(t, baseline.ConfigHash, 64)
	require.Equal(t, 1, baseline.Version)
	require.Equal(t, RetuneRouterConfiguration{OverlapScoreCredit: 1, PrefillLoadScale: 1}, baseline.Router)

	t.Log("Replica changes and explicit defaults do not alter the immutable process binding")
	unchanged := dgd.DeepCopy()
	unchanged.Generation++
	unchanged.Spec.Components[0].Replicas = new(int32(3))
	unchanged.Spec.Components[1].Replicas = new(int32(2))
	unchanged.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(1)), PrefillLoadScale: new(float64(1)), Temperature: new(float64(0))}
	_, equivalent, err := RetuneConfiguration(unchanged)
	require.NoError(t, err)
	require.Equal(t, baseline, equivalent)

	for _, test := range []struct {
		name   string
		mutate func(*v1beta1.DynamoGraphDeployment)
	}{
		{"graph identity", func(value *v1beta1.DynamoGraphDeployment) { value.UID = "replacement-uid" }},
		{"frontend identity", func(value *v1beta1.DynamoGraphDeployment) { value.Spec.Components[0].ComponentName = "other-gateway" }},
		{"Planner identity", func(value *v1beta1.DynamoGraphDeployment) { value.Spec.Components[1].ComponentName = "other-optimizer" }},
		{"storage", func(value *v1beta1.DynamoGraphDeployment) {
			value.Spec.Components[1].Retune.TraceStorage.ClaimName = "other-traces"
		}},
		{"startup router input", func(value *v1beta1.DynamoGraphDeployment) {
			value.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{Temperature: new(float64(0.5))}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("A semantic binding edit changes the target hash")
			changed := dgd.DeepCopy()
			test.mutate(changed)
			_, configuration, err := RetuneConfiguration(changed)
			require.NoError(t, err)
			require.NotEqual(t, baseline.ConfigHash, configuration.ConfigHash)
		})
	}

	t.Log("Stored or admission-bypassed invalid numbers cannot produce a launch contract")
	for _, invalid := range []float64{-1, math.Inf(1), math.NaN()} {
		changed := dgd.DeepCopy()
		changed.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{Temperature: &invalid}
		_, _, err := RetuneConfiguration(changed)
		require.ErrorContains(t, err, "finite and non-negative")
	}
}

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
					require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: retuneTraceVolumeName, MountPath: "/trace"})
					require.Contains(t, pod.Volumes, corev1.Volume{Name: retuneTraceVolumeName, PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "traces"}})
					if name != "gateway" {
						binding := findEnvVar(main.Env, "DYN_PLANNER_RETUNE_CAPABILITY")
						require.NotNil(t, binding)
						var capability RetuneLaunchConfiguration
						require.NoError(t, json.Unmarshal([]byte(binding.Value), &capability))
						_, expected, err := RetuneConfiguration(dgd)
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

func TestRetuneMountOwnership(t *testing.T) {
	for _, test := range []struct {
		name      string
		mount     corev1.VolumeMount
		volume    corev1.Volume
		wantError string
	}{
		{name: "reuse explicit matching mount", mount: corev1.VolumeMount{Name: "user-traces", MountPath: "/trace"}, volume: corev1.Volume{Name: "user-traces", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "traces"}}}},
		{name: "wrong claim", mount: corev1.VolumeMount{Name: "other", MountPath: "/trace"}, volume: corev1.Volume{Name: "other", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "other"}}}, wantError: "must reference PVC"},
		{name: "read only", mount: corev1.VolumeMount{Name: "other", MountPath: "/trace", ReadOnly: true}, wantError: "direct writable"},
		{name: "subpath", mount: corev1.VolumeMount{Name: "other", MountPath: "/trace", SubPath: "custom"}, wantError: "without subPath"},
		{name: "nested obscuring mount", mount: corev1.VolumeMount{Name: "other", MountPath: "/trace/segments"}, wantError: "obscured"},
		{name: "reserved name mounted elsewhere", mount: corev1.VolumeMount{Name: retuneTraceVolumeName, MountPath: "/other"}, wantError: "already mounted"},
		{name: "reserved volume name collision", volume: corev1.Volume{Name: retuneTraceVolumeName, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}, wantError: "must reference writable PVC"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Respect matching user storage and reject conflicting storage without overwriting it")
			pod := corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}}
			if test.mount.Name != "" {
				pod.Containers[0].VolumeMounts = []corev1.VolumeMount{test.mount}
			}
			if test.volume.Name != "" {
				pod.Volumes = []corev1.Volume{test.volume}
			}
			original := pod.DeepCopy()
			err := addRetuneTraceMount(&pod, &pod.Containers[0], RetuneTraceStorage{ClaimName: "traces", MountPath: "/trace"})
			if test.wantError == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, test.wantError)
			}
			require.Equal(t, original, &pod)
		})
	}
}

func TestRetuneRejectsNestedMountAfterReusableMount(t *testing.T) {
	for _, test := range []struct {
		name   string
		mounts []corev1.VolumeMount
	}{
		{name: "nested mount last", mounts: []corev1.VolumeMount{{Name: "user-traces", MountPath: "/trace"}, {Name: "nested", MountPath: "/trace/segments"}}},
		{name: "nested mount first", mounts: []corev1.VolumeMount{{Name: "nested", MountPath: "/trace/segments"}, {Name: "user-traces", MountPath: "/trace"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("A reusable root mount never hides a later obscuring child mount")
			pod := corev1.PodSpec{Containers: []corev1.Container{{Name: "main", VolumeMounts: test.mounts}}, Volumes: []corev1.Volume{
				{Name: "user-traces", PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "traces"}},
			}}
			original := pod.DeepCopy()
			err := addRetuneTraceMount(&pod, &pod.Containers[0], RetuneTraceStorage{ClaimName: "traces", MountPath: "/trace"})
			require.ErrorContains(t, err, "obscured")
			require.Equal(t, original, &pod)
		})
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
			planner := GetMainContainerFromPodSpec(&cliqueByName(rendered, "optimizer").Spec.PodSpec)
			require.Equal(t, test.enabled, findEnvVar(planner.Env, "DYN_PLANNER_RETUNE_CAPABILITY") != nil)
			if !test.enabled {
				require.Equal(t, baseline, rendered)
			} else {
				require.NotEqual(t, cliqueByName(baseline, "gateway"), cliqueByName(rendered, "gateway"))
				require.NotEqual(t, cliqueByName(baseline, "optimizer"), cliqueByName(rendered, "optimizer"))

				t.Log("A PVC change changes the Planner launch binding as well as its mount")
				oldBinding := findEnvVar(planner.Env, "DYN_PLANNER_RETUNE_CAPABILITY").Value
				candidate.Spec.Components[1].Retune.TraceStorage.ClaimName = "new-traces"
				updated, err := GenerateGrovePodCliqueSet(t.Context(), candidate, nil, config, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
				require.NoError(t, err)
				updatedPlanner := GetMainContainerFromPodSpec(&cliqueByName(updated, "optimizer").Spec.PodSpec)
				newBinding := findEnvVar(updatedPlanner.Env, "DYN_PLANNER_RETUNE_CAPABILITY")
				require.NotNil(t, newBinding)
				require.NotEqual(t, oldBinding, newBinding.Value)
				var capability RetuneLaunchConfiguration
				require.NoError(t, json.Unmarshal([]byte(newBinding.Value), &capability))
				require.Equal(t, "new-traces", capability.TraceStorage.ClaimName)
			}
		})
	}
}
