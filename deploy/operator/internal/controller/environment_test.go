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
	"k8s.io/apimachinery/pkg/api/resource"
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
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
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

func TestSnapshotGMSPhysicalSetupPreservesComposedEnvironment(t *testing.T) {
	t.Log("The target and extra capture client have already composed GMS bindings")
	pod := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{
		{Name: "main", Env: append(gms.ClientEnvironment(true), corev1.EnvVar{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"})},
		{Name: "saver", Env: append(gms.ClientEnvironment(true), corev1.EnvVar{Name: "SAVE_PATH", Value: "$(GMS_SOCKET_DIR)/snapshot"})},
	}}}
	original := pod.DeepCopy()
	err := prepareCheckpointGMSPodTemplate(pod, "main", "checkpoint", &v1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: v1alpha1.GMSModeIntraPod, ExtraClientContainers: []string{"saver"}})
	require.NoError(t, err)
	for i := range pod.Spec.Containers {
		require.Equal(t, original.Spec.Containers[i].Env, pod.Spec.Containers[i].Env)
	}
}

func TestSnapshotGMSBindingsAvailableDuringRendering(t *testing.T) {
	for _, targetName := range []string{"main", "writer"} {
		t.Run(targetName, func(t *testing.T) {
			t.Log("A GMS-owned binding replaces a cyclic authored definition before capture rendering")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env:       []corev1.EnvVar{{Name: gms.EnvSocketDir, Value: "$(WEIGHTS)"}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				}}}},
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod},
					Checkpoint:       &v1beta1.ComponentCheckpointConfig{Enabled: true, TargetContainerName: targetName},
				},
			}
			if targetName != "main" {
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{
					Name: targetName, Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env:       []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				})
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}}},
			}
			original := component.DeepCopy()
			reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
			pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkVLLM)
			require.NoError(t, err)

			t.Log("Physical GMS setup retains the already resolved ownership and ordering")
			require.Equal(t, targetName, pod.Spec.Containers[0].Name)
			err = prepareCheckpointGMSPodTemplate(&pod, targetName, "checkpoint", &v1alpha1.GPUMemoryServiceSpec{Enabled: true, Mode: v1alpha1.GMSModeIntraPod})
			require.NoError(t, err)
			var names []string
			for _, env := range pod.Spec.Containers[0].Env {
				if env.Name == gms.EnvSocketDir || env.Name == "WEIGHTS" {
					names = append(names, env.Name)
				}
				if env.Name == gms.EnvSocketDir {
					require.Equal(t, gms.SharedMountPath, env.Value)
				}
			}
			require.Equal(t, []string{gms.EnvSocketDir, "WEIGHTS"}, names)
			require.Equal(t, original, component)
		})
	}
}

func TestSnapshotFailoverOmissionPrecedesEnvironmentComposition(t *testing.T) {
	for _, targetName := range []string{"main", "writer"} {
		t.Run(targetName, func(t *testing.T) {
			t.Log("Capture omits the serving-only metric port before validating dependencies")
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:test", Command: []string{"python3", "-m", "dynamo.vllm"},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				}}}},
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod},
					Failover:         &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeIntraPod},
					Checkpoint:       &v1beta1.ComponentCheckpointConfig{Enabled: true, TargetContainerName: targetName},
				},
			}
			target := component.PodTemplate.Spec.Containers[0].DeepCopy()
			target.Name = targetName
			target.Env = []corev1.EnvVar{{Name: "DYN_FORWARDPASS_METRIC_PORT", Value: "$(LABEL_PORT)"}, {Name: "AFTER", Value: "retained"}}
			if targetName == "main" {
				component.PodTemplate.Spec.Containers[0] = *target
			} else {
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, *target)
			}
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
				Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
				Spec:        v1beta1.DynamoGraphDeploymentSpec{Env: []corev1.EnvVar{{Name: "LABEL_PORT", Value: "$(DYN_FORWARDPASS_METRIC_PORT)"}}},
			}
			reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
			pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkVLLM)
			require.NoError(t, err)
			require.Equal(t, targetName, pod.Spec.Containers[0].Name)
			require.Contains(t, pod.Spec.Containers[0].Env, corev1.EnvVar{Name: "AFTER", Value: "retained"})
			for _, env := range pod.Spec.Containers[0].Env {
				require.NotEqual(t, "DYN_FORWARDPASS_METRIC_PORT", env.Name)
			}
		})
	}
}

func TestSnapshotJobSidecarDoesNotInheritDiscardedComponentEnvironment(t *testing.T) {
	t.Log("A job-only metrics sidecar replaces rather than inherits the discarded component sidecar")
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{
			{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"}},
			{Name: "metrics", Image: "metrics:component", Env: []corev1.EnvVar{{Name: "X", Value: "$(Y)"}},
				EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "component-metrics"}}}},
		}}},
		Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true,
			Job: &v1beta1.ComponentCheckpointJobConfig{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "metrics", Image: "metrics:job", Env: []corev1.EnvVar{{Name: "Y", Value: "$(X)"}},
				EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "job-metrics"}}},
			}}}}},
		}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
	}
	reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
	pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkVLLM)
	require.NoError(t, err)
	require.Len(t, pod.Spec.Containers, 2)
	require.Equal(t, component.Experimental.Checkpoint.Job.PodTemplate.Spec.Containers[0], pod.Spec.Containers[1])
}
