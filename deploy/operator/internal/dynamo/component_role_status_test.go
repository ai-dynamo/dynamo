// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestResolveComponentRoleStatus(t *testing.T) {
	t.Log("Build final declarations with ordered aliases, duplicate names, and opaque inputs")
	pod := &corev1.PodSpec{
		NodeName: "not-exported",
		Containers: []corev1.Container{{
			Name: "main", Image: "engine:test", Command: []string{"/bin/sh", "-c"}, Args: []string{"exec engine $(MODE)"},
			Env: []corev1.EnvVar{
				{Name: "Z_MODE", Value: "k"}, {Name: "Z_MODE", Value: "$(Z_MODE)v"},
				{Name: "MODE", Value: "$(Z_MODE)"}, {Name: "EMPTY"},
				{Name: "ESCAPED", Value: "$$(MODE)"},
				{Name: "FORWARD", Value: "$(LATER)"}, {Name: "LATER", Value: "later"},
				{Name: "TOKEN", Value: "stale"},
				{Name: "TOKEN", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Key: "token"}}},
				{Name: "DERIVED", Value: "prefix-$(TOKEN)"},
				{Name: "CONFIG", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Key: "config"}}},
				{Name: "POD", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
				{Name: "RANK", Value: "$(GROVE_PCSG_POD_INDEX)"},
			},
		}},
		InitContainers: []corev1.Container{{Name: "runtime", Image: "runtime:test", Env: []corev1.EnvVar{{Name: "MODE", Value: "round-robin"}}}},
	}
	original := pod.DeepCopy()

	t.Log("Project known final bindings without guessing provider, Secret, or Pod values")
	status := ResolveComponentRoleStatus("leader", pod)
	require.Equal(t, "leader", status.Name)
	require.Len(t, status.PodSpec.Containers, 1)
	require.Len(t, status.PodSpec.InitContainers, 1)
	require.Empty(t, status.PodSpec.NodeName)
	main := status.PodSpec.Containers[0]
	require.Equal(t, pod.Containers[0].Command, main.Command)
	require.Equal(t, pod.Containers[0].Args, main.Args)
	env := make(map[string]corev1.EnvVar, len(main.Env))
	for _, variable := range main.Env {
		require.NotContains(t, env, variable.Name)
		env[variable.Name] = variable
	}
	for name, value := range map[string]string{"Z_MODE": "kv", "MODE": "kv", "EMPTY": "", "ESCAPED": "$(MODE)", "LATER": "later"} {
		require.Equal(t, corev1.EnvVar{Name: name, Value: value}, env[name])
	}
	for _, index := range []int{5, 8, 9, 10, 11, 12} {
		variable := pod.Containers[0].Env[index]
		require.Equal(t, variable, env[variable.Name])
	}
	require.NotContains(t, env, "GROVE_PCSG_POD_INDEX")
	require.Equal(t, "round-robin", status.PodSpec.InitContainers[0].Env[0].Value)
	require.Equal(t, original, pod)

	t.Log("Editing the exported declarations must not mutate the workload")
	status.PodSpec.Containers[0].Command[0] = "changed"
	status.PodSpec.Containers[0].Args[0] = "changed"
	env["TOKEN"].ValueFrom.SecretKeyRef.Key = "changed"
	require.Equal(t, original, pod)
}

func TestComponentRoleStatusBoundsRetainedExpansions(t *testing.T) {
	t.Log("Many compact containers must not retain a replay budget's worth of aliases each")
	env := []corev1.EnvVar{{Name: "V0", Value: "x"}}
	for i := 1; i < 25; i++ {
		previous := "$(V" + strconv.Itoa(i-1) + ")"
		env = append(env, corev1.EnvVar{Name: "V" + strconv.Itoa(i), Value: previous + previous})
	}
	pod := &corev1.PodSpec{}
	for i := range 20 {
		pod.Containers = append(pod.Containers, corev1.Container{Name: "container-" + strconv.Itoa(i), Env: env})
	}
	pod.InitContainers = pod.Containers[10:]
	pod.Containers = pod.Containers[:10]
	original := pod.DeepCopy()
	status := ResolveComponentRoleStatus("main", pod)
	data, err := json.Marshal(status)
	require.NoError(t, err)
	require.Less(t, len(data), maxComponentRoleStatusBytes)
	require.Equal(t, env[18], status.PodSpec.InitContainers[9].Env[18])
	require.Equal(t, original, pod)
}

func TestBoundComponentRoleStatuses(t *testing.T) {
	t.Log("An expanded environment must not make observational status unbounded")
	large := strings.Repeat("x", maxComponentRoleStatusBytes)
	statuses := map[string]v1beta1.ComponentReplicaStatus{
		"small": {Replicas: 2, Roles: []v1beta1.ComponentRoleStatus{{Name: "main"}}},
		"large": {Replicas: 1, Roles: []v1beta1.ComponentRoleStatus{{Name: "main", PodSpec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{{Name: "LARGE", Value: large}}}}}}}},
	}
	BoundComponentRoleStatuses(statuses)
	require.Nil(t, statuses["large"].Roles)
	require.NotEmpty(t, statuses["small"].Roles)
	require.EqualValues(t, 1, statuses["large"].Replicas)
	require.EqualValues(t, 2, statuses["small"].Replicas)

	t.Log("A missing projection stays missing")
	BoundComponentRoleStatuses(nil)
}
