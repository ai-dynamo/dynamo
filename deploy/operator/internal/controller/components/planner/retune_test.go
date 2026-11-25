// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package planner

import (
	"math"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestRetuneSemanticBinding(t *testing.T) {
	t.Log("Compile one versioned contract from typed inputs and graph identity")
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", UID: "graph-uid", Generation: 7, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
		{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
		{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
			Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
	}}}
	_, baseline, err := ResolveRetuneConfiguration(dgd)
	require.NoError(t, err)
	require.Equal(t, "9c71d5750148c53a4cf5009d87c1a750cde7679bdeded10479dbaf295e099096", baseline.ConfigHash)
	require.Equal(t, 1, baseline.Version)
	require.Equal(t, RetuneRouterConfiguration{Profile: "kv-v1", OverlapScoreCredit: 1, PrefillLoadScale: 1}, baseline.Router)

	t.Log("Replica changes and explicit defaults do not alter the immutable process binding")
	unchanged := dgd.DeepCopy()
	unchanged.Generation++
	unchanged.Spec.Components[0].Replicas = new(int32(3))
	unchanged.Spec.Components[1].Replicas = new(int32(2))
	unchanged.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(1)), PrefillLoadScale: new(float64(1)), Temperature: new(float64(0))}
	_, equivalent, err := ResolveRetuneConfiguration(unchanged)
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
			_, configuration, err := ResolveRetuneConfiguration(changed)
			require.NoError(t, err)
			require.NotEqual(t, baseline.ConfigHash, configuration.ConfigHash)
		})
	}

	t.Log("Stored or admission-bypassed invalid numbers cannot produce a launch contract")
	for _, invalid := range []float64{-1, math.Inf(1), math.NaN()} {
		changed := dgd.DeepCopy()
		changed.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{Temperature: &invalid}
		_, _, err := ResolveRetuneConfiguration(changed)
		require.ErrorContains(t, err, "finite and non-negative")
	}
}

func TestRetuneAppliesOnlyToSuppliedPod(t *testing.T) {
	t.Log("Feature wiring does not select or create a template or container")
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "gateway", PodTemplate: &corev1.PodTemplateSpec{}}
	original := component.DeepCopy()
	configuration := &RetuneLaunchConfiguration{Version: 1, FrontendComponentName: "gateway", AdminPort: 8002,
		TraceStorage: RetuneTraceStorage{ClaimName: "traces", MountPath: "/trace", SegmentsDir: "segments", WindowsDir: "windows"},
	}
	pod := corev1.PodSpec{Containers: []corev1.Container{{Name: "sidecar", Image: "sidecar:1"}, {Name: "main", Image: "frontend:1"}}}
	sidecar := pod.Containers[0].DeepCopy()
	require.NoError(t, configuration.Apply(component, "graph", &pod, &pod.Containers[1]))
	require.Len(t, pod.Containers, 2)
	require.Equal(t, sidecar, &pod.Containers[0])
	require.Equal(t, "main", pod.Containers[1].Name)
	require.Equal(t, "frontend:1", pod.Containers[1].Image)
	require.NotEmpty(t, pod.Containers[1].Env)
	require.Equal(t, original, component)

	t.Log("Applying the feature twice preserves the exact Pod")
	configured := pod.DeepCopy()
	require.NoError(t, configuration.Apply(component, "graph", &pod, &pod.Containers[1]))
	require.Equal(t, configured, &pod)
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
