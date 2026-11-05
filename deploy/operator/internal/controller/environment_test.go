/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestSnapshotEnvironmentRetainsFourSourceLayers(t *testing.T) {
	t.Log("A job-specific override references the component while a graph value references the override")
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
			Env: []corev1.EnvVar{{Name: "STORAGE", Value: "nats://custom:4222"}},
		}}}},
		Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true,
			Job: &v1beta1.ComponentCheckpointJobConfig{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "main", Env: []corev1.EnvVar{{Name: "NATS_SERVER", Value: "$(STORAGE)"}},
			}}}}},
		}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"},
		Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: "MODEL", Value: "$(NATS_SERVER)/model"}}},
	}
	reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{Infrastructure: configv1alpha1.InfrastructureConfiguration{NATSAddress: "default"}}}
	pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkVLLM)
	require.NoError(t, err)
	var relevant []string
	for _, env := range pod.Spec.Containers[0].Env {
		if env.Name == "STORAGE" || env.Name == "NATS_SERVER" || env.Name == "MODEL" {
			relevant = append(relevant, env.Name)
		}
	}
	require.Equal(t, []string{"STORAGE", "NATS_SERVER", "MODEL"}, relevant)
}

func TestSnapshotGMSBindingsPrecedeReferences(t *testing.T) {
	t.Log("The target and an extra capture client reference GMS values wired after base rendering")
	pod := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{
		{Name: "main", Env: []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}, {Name: gms.EnvSocketDir, Value: "ignored"}}},
		{Name: "saver", Env: []corev1.EnvVar{{Name: "SAVE_PATH", Value: "$(GMS_SOCKET_DIR)/snapshot"}}},
	}}}
	err := prepareCheckpointGMSPodTemplate(pod, "main", "checkpoint", &v1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: v1alpha1.GMSModeIntraPod, ExtraClientContainers: []string{"saver"}},
		map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"})
	require.NoError(t, err)
	for _, container := range pod.Spec.Containers {
		t.Log("Verify authoritative GMS values precede each dependent authored value")
		var names []string
		for _, env := range container.Env {
			names = append(names, env.Name)
			if env.Name == gms.EnvSocketDir {
				require.Equal(t, gms.SharedMountPath, env.Value)
			}
		}
		require.Equal(t, gms.EnvSocketDir, names[0])
	}
}
