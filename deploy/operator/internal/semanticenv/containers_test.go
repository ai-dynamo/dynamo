/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestContainerLayersSurvivePhysicalFiltering(t *testing.T) {
	t.Log("Capture sources before an in-place Pod mutation reuses their backing array")
	pod := &corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{
		{Name: "REMOVED", Value: "stale"}, {Name: "PATH", Value: "$(ROOT)/model"},
	}}}}
	layers := make(ContainerLayers)
	layers.Capture(pod)
	pod.Containers[0].Env[0] = pod.Containers[0].Env[1]
	pod.Containers[0].Env = pod.Containers[0].Env[:1]
	layers.Omit(&pod.Containers[0], "REMOVED")
	layers.Bind(&pod.Containers[0], "operator.storage", []corev1.EnvVar{{Name: "ROOT", Value: "/models"}})

	t.Log("Finalize surviving values from the immutable sources, including their original order")
	require.NoError(t, layers.Apply(pod))
	require.Equal(t, []corev1.EnvVar{{Name: "ROOT", Value: "/models"}, {Name: "PATH", Value: "$(ROOT)/model"}}, pod.Containers[0].Env)
}

func TestContainerLayerMutations(t *testing.T) {
	for _, semantic := range []bool{false, true} {
		t.Run(fmt.Sprintf("semantic=%t", semantic), func(t *testing.T) {
			t.Log("Mutate an uncaptured container through the same boundary in both modes")
			var layers ContainerLayers
			if semantic {
				layers = make(ContainerLayers)
			}
			pod := &corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Env: []corev1.EnvVar{
				{Name: "REMOVED", Value: "stale"}, {Name: "PATH", Value: "$(ROOT)/model"},
			}}}}
			main := &pod.Containers[0]
			layers.Bind(main, "operator.storage", []corev1.EnvVar{{Name: "ROOT", Value: "/models"}})
			layers.Omit(main, "REMOVED")
			require.Equal(t, []corev1.EnvVar{{Name: "PATH", Value: "$(ROOT)/model"}, {Name: "ROOT", Value: "/models"}}, main.Env)

			t.Log("Finalization retains authored values and omissions without changing legacy order")
			require.NoError(t, layers.Apply(pod))
			want := []corev1.EnvVar{{Name: "PATH", Value: "$(ROOT)/model"}, {Name: "ROOT", Value: "/models"}}
			if semantic {
				want = []corev1.EnvVar{want[1], want[0]}
			}
			require.Equal(t, want, main.Env)
		})
	}
}

func TestContainerLayerPrependAndClone(t *testing.T) {
	for _, semantic := range []bool{false, true} {
		t.Run(fmt.Sprintf("semantic=%t", semantic), func(t *testing.T) {
			t.Log("Replace an authored binding before cloning a failover container")
			var layers ContainerLayers
			if semantic {
				layers = make(ContainerLayers)
			}
			base := corev1.Container{Name: "main", Env: []corev1.EnvVar{
				{Name: "ROOT", Value: "stale"}, {Name: "REMOVE", Value: "stale"}, {Name: "PATH", Value: "$(ROOT)/model"},
			}}
			layers.Prepend(&base, "operator.storage", []corev1.EnvVar{{Name: "ROOT", Value: "/models"}})
			require.Equal(t, []corev1.EnvVar{{Name: "ROOT", Value: "/models"}, {Name: "REMOVE", Value: "stale"}, {Name: "PATH", Value: "$(ROOT)/model"}}, base.Env)
			engine := corev1.Container{Name: "engine-0"}
			layers.Clone(&base, &engine, []string{"REMOVE"})
			layers.Bind(&engine, "operator.engine", []corev1.EnvVar{{Name: "ENGINE_ID", Value: "0"}})

			t.Log("The clone keeps source layers but does not retain omitted or stale values")
			pod := &corev1.PodSpec{Containers: []corev1.Container{engine}}
			require.NoError(t, layers.Apply(pod))
			require.Equal(t, []corev1.EnvVar{{Name: "ROOT", Value: "/models"}, {Name: "PATH", Value: "$(ROOT)/model"}, {Name: "ENGINE_ID", Value: "0"}}, pod.Containers[0].Env)
			require.Len(t, base.Env, 3)
		})
	}
}

func TestContainerLayerPodBindings(t *testing.T) {
	t.Log("Apply Pod-level bindings to both regular and init containers before capture")
	pod := &corev1.PodSpec{
		Containers:     []corev1.Container{{Name: "main", Env: []corev1.EnvVar{{Name: "PATH", Value: "$(ROOT)/main"}}}},
		InitContainers: []corev1.Container{{Name: "prepare", Env: []corev1.EnvVar{{Name: "PATH", Value: "$(ROOT)/prepare"}}}},
	}
	layers := make(ContainerLayers)
	layers.PrependPod(pod, "operator.storage", []corev1.EnvVar{{Name: "ROOT", Value: "/models"}})
	layers.BindPod(pod, "operator.cluster", []corev1.EnvVar{{Name: "CLUSTER", Value: "serving"}})

	t.Log("Finalization preserves both authored lists and their precise Pod field paths")
	require.NoError(t, layers.Apply(pod))
	for _, group := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for _, container := range group {
			require.Equal(t, "ROOT", container.Env[0].Name)
			require.Equal(t, "PATH", container.Env[1].Name)
			require.Equal(t, "CLUSTER", container.Env[2].Name)
		}
	}
	require.Equal(t, "podTemplate.spec.initContainers[name=prepare].env", layers["prepare"][0].Source)
}
