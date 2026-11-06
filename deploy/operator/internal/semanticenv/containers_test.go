/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
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
	layers.Omit("main", "REMOVED")
	layers.Bind("main", "operator.storage", []corev1.EnvVar{{Name: "ROOT", Value: "/models"}})

	t.Log("Finalize surviving values from the immutable sources, including their original order")
	require.NoError(t, layers.Apply(pod))
	require.Equal(t, []corev1.EnvVar{{Name: "ROOT", Value: "/models"}, {Name: "PATH", Value: "$(ROOT)/model"}}, pod.Containers[0].Env)
}
