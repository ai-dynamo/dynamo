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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// snapshotEnvironmentFixture gives each test its own graph and checkpoint worker.
func snapshotEnvironmentFixture() (*v1beta1.DynamoComponentDeploymentSharedSpec, *v1beta1.DynamoGraphDeployment) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
			Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
		}}}},
		Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true}},
	}
	dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns",
		Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"},
	}
	return component, dgd
}

func TestSnapshotEnvironmentRetainsFourSourceLayers(t *testing.T) {
	t.Log("A job-specific override references the component while a graph value references the override")
	component, dgd := snapshotEnvironmentFixture()
	component.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "STORAGE", Value: "nats://custom:4222"}}
	component.Experimental.Checkpoint.Job = &v1beta1.ComponentCheckpointJobConfig{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main", Env: []corev1.EnvVar{{Name: "NATS_SERVER", Value: "$(STORAGE)"}},
	}}}}}
	dgd.Spec.Env = []corev1.EnvVar{{Name: "MODEL", Value: "$(NATS_SERVER)/model"}}
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

func TestSnapshotSGLangLookupUsesFinalJobEnvironment(t *testing.T) {
	for _, targetName := range []string{"main", "writer"} {
		for _, test := range []struct {
			name      string
			base      corev1.EnvVar
			port      int32
			wantError string
		}{
			{name: "known job override", base: corev1.EnvVar{Name: "BASE", Value: "30500"}, port: 30500},
			{name: "opaque job override", base: corev1.EnvVar{Name: "BASE", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "settings", Key: "port"}}}, port: 19090},
			{name: "known malformed job override", base: corev1.EnvVar{Name: "BASE", Value: "invalid"}, port: 19090, wantError: "not a number"},
		} {
			t.Run(targetName+"/"+test.name, func(t *testing.T) {
				t.Log("The port reference is resolved only after the selected capture container's job override")
				main := corev1.Container{Name: "main", Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.sglang"},
					Env:   []corev1.EnvVar{{Name: "NIXL_TELEMETRY_ENABLE", Value: "y"}, {Name: "NIXL_TELEMETRY_EXPORTER", Value: "prometheus"}, {Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", Value: "$(BASE)"}},
					Ports: []corev1.ContainerPort{{Protocol: corev1.ProtocolTCP, Name: commonconsts.DynamoNixlPortName, ContainerPort: 19090}},
				}
				component, dgd := snapshotEnvironmentFixture()
				component.PodTemplate.Spec.Containers[0] = main
				component.Experimental.Checkpoint.TargetContainerName = targetName
				component.Experimental.Checkpoint.Job = &v1beta1.ComponentCheckpointJobConfig{PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: targetName, Env: []corev1.EnvVar{test.base},
				}}}}}
				if targetName != "main" {
					writer := *main.DeepCopy()
					writer.Name = targetName
					component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, writer)
				}
				dgd.Spec.Env = []corev1.EnvVar{{Name: "BASE", Value: "30000"}}
				original := component.DeepCopy()
				reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
				pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkSGLang)
				if targetName == "main" && test.wantError != "" {
					require.ErrorContains(t, err, test.wantError)
					require.Equal(t, original, component)
					return
				}
				require.NoError(t, err)
				require.Equal(t, targetName, pod.Spec.Containers[0].Name)
				port := test.port
				if targetName != "main" {
					t.Log("A selected helper keeps its own scrape port and is not validated as the engine")
					port = 19090
				}
				require.Contains(t, pod.Spec.Containers[0].Ports, corev1.ContainerPort{Protocol: corev1.ProtocolTCP, Name: commonconsts.DynamoNixlPortName, ContainerPort: port})
				require.Contains(t, pod.Spec.Containers[0].Env, corev1.EnvVar{Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", Value: "$(BASE)"})
				require.Equal(t, original, component)
			})
		}
	}
}

func TestSnapshotCompatibilityEnvironmentUsesAuthoritativeOrigin(t *testing.T) {
	for _, origin := range []string{"", "1.7.0", "1.8.0"} {
		t.Run(origin, func(t *testing.T) {
			t.Log("An alias around a restore-time binding is part of new-origin process identity")
			component, dgd := snapshotEnvironmentFixture()
			component.PodTemplate.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"}
			component.PodTemplate.Spec.Containers[0].Image = "engine:1.8.0"
			component.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "DYN_SERVED_MODEL_NAME", Value: "model-a"}, {Name: "MODEL_ID", Value: "$(DYN_SERVED_MODEL_NAME)"}}
			if origin == "1.8.0" {
				component.PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = "1.7.0"
			}
			dgd.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = origin
			dgd.Spec.BackendFramework = "vllm"
			reordered := component.DeepCopy()
			reordered.PodTemplate.Spec.Containers[0].Env[0], reordered.PodTemplate.Spec.Containers[0].Env[1] =
				reordered.PodTemplate.Spec.Containers[0].Env[1], reordered.PodTemplate.Spec.Containers[0].Env[0]
			reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
			beforeHash, err := reconciler.snapshotCompatibilityHashForComponent(dgd, "worker", component)
			require.NoError(t, err)
			afterHash, err := reconciler.snapshotCompatibilityHashForComponent(dgd, "worker", reordered)
			require.NoError(t, err)
			if origin == "1.8.0" {
				require.NotEqual(t, beforeHash, afterHash)
			} else {
				require.Equal(t, beforeHash, afterHash)
			}
		})
	}
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
			component, dgd := snapshotEnvironmentFixture()
			component.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: gms.EnvSocketDir, Value: "$(WEIGHTS)"}}
			component.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
			component.Experimental.GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod}
			component.Experimental.Checkpoint.TargetContainerName = targetName
			if targetName != "main" {
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{
					Name: targetName, Image: "engine:1.7.0", Command: []string{"python3", "-m", "dynamo.vllm"},
					Env:       []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}},
					Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}},
				})
			}
			dgd.Spec.Env = []corev1.EnvVar{{Name: "WEIGHTS", Value: "$(GMS_SOCKET_DIR)/weights"}}
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
			component, dgd := snapshotEnvironmentFixture()
			component.PodTemplate.Spec.Containers[0].Image = "engine:test"
			component.PodTemplate.Spec.Containers[0].Resources = corev1.ResourceRequirements{Limits: corev1.ResourceList{"nvidia.com/gpu": resource.MustParse("1")}}
			component.Experimental.GPUMemoryService = &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod}
			component.Experimental.Failover = &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeIntraPod}
			component.Experimental.Checkpoint.TargetContainerName = targetName
			target := component.PodTemplate.Spec.Containers[0].DeepCopy()
			target.Name = targetName
			target.Env = []corev1.EnvVar{{Name: "DYN_FORWARDPASS_METRIC_PORT", Value: "$(LABEL_PORT)"}, {Name: "AFTER", Value: "retained"}}
			if targetName == "main" {
				component.PodTemplate.Spec.Containers[0] = *target
			} else {
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, *target)
			}
			dgd.Spec.Env = []corev1.EnvVar{{Name: "LABEL_PORT", Value: "$(DYN_FORWARDPASS_METRIC_PORT)"}}
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
	for _, kind := range []string{"containers", "initContainers"} {
		t.Run(kind, func(t *testing.T) {
			t.Log("A job replacement does not inherit the discarded container's sources")
			component, dgd := snapshotEnvironmentFixture()
			authored := corev1.Container{Name: "metrics", Image: "metrics:component", Env: []corev1.EnvVar{{Name: "X", Value: "$(Y)"}},
				EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "component-metrics"}}}}
			replacement := corev1.Container{Name: "metrics", Image: "metrics:job", Env: []corev1.EnvVar{{Name: "Y", Value: "$(X)"}},
				EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "job-metrics"}}}}
			jobTemplate := &corev1.PodTemplateSpec{}
			if kind == "containers" {
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, authored)
				jobTemplate.Spec.Containers = []corev1.Container{replacement}
			} else {
				component.PodTemplate.Spec.InitContainers = []corev1.Container{authored}
				jobTemplate.Spec.InitContainers = []corev1.Container{replacement}
			}
			component.Experimental.Checkpoint.Job = &v1beta1.ComponentCheckpointJobConfig{PodTemplate: jobTemplate}
			original := component.DeepCopy()
			reconciler := &dgdCheckpointsReconciler{config: &configv1alpha1.OperatorConfiguration{}}
			pod, err := reconciler.buildCheckpointJobPodTemplate(dgd, component, "worker", dynamo.BackendFrameworkVLLM)
			require.NoError(t, err)
			containers := pod.Spec.Containers
			if kind == "containers" {
				require.Len(t, containers, 2)
			} else {
				containers = pod.Spec.InitContainers
				require.Len(t, containers, 1)
			}
			require.Equal(t, &replacement, common.FindContainerByName(containers, "metrics"))
			require.Equal(t, original, component)
		})
	}
}
