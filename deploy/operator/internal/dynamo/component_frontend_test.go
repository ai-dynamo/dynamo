/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestFrontendDefaultsStartupProbeVersionGate(t *testing.T) {
	tests := []struct {
		name           string
		origin         string
		runtimeVersion *runtimeversion.Version
		wantEnabled    bool
	}{
		{name: "unknown runtime on new graph", origin: "1.7.0"},
		{name: "older runtime on new graph", origin: "1.7.0", runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 5, Patch: 9}},
		{name: "released 1.6 graph preserves existing defaults", origin: "1.6.0", runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 6}},
		{name: "new graph with minimum supported runtime", origin: "1.7.0", runtimeVersion: &runtimeversion.Version{Major: 1, Minor: 6}, wantEnabled: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render frontend defaults for the resolved runtime version")
			container, err := NewFrontendDefaults().GetBaseContainer(ComponentContext{RuntimeVersion: tt.runtimeVersion, Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: tt.origin}})
			require.NoError(t, err)

			t.Log("Check the version-gated startup allowance")
			if tt.wantEnabled {
				require.Equal(t, &corev1.Probe{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/live", Port: intstr.FromString("http"),
					},
					PeriodSeconds: 10, TimeoutSeconds: 1, FailureThreshold: 30,
				}, container.StartupProbe)
			} else {
				require.Nil(t, container.StartupProbe)
			}
		})
	}
}

func TestFrontendSidecarProbeOverride(t *testing.T) {
	tests := []struct {
		name    string
		handler corev1.ProbeHandler
	}{
		{name: "exec", handler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{"true"}}}},
		{name: "http", handler: corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/ready", Port: intstr.FromInt(8000)}}},
		{name: "tcp", handler: corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(8000)}}},
		{name: "grpc", handler: corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: 8000}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Provide complete frontend sidecar probes on runtime 1.6.0")
			override := &corev1.Probe{ProbeHandler: tt.handler, PeriodSeconds: 5, FailureThreshold: 90}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				RuntimeVersionOverride: "1.6.0", FrontendSidecar: new("frontend"),
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Image: "worker:1.6.0"},
					{
						Name: "frontend", Image: "frontend:1.6.0",
						StartupProbe: override.DeepCopy(), LivenessProbe: override.DeepCopy(), ReadinessProbe: override.DeepCopy(),
					},
				}}},
			}

			t.Log("Render the complete pod specification through the production entry point")
			pod, err := GenerateBasePodSpec(
				component, BackendFrameworkVLLM, nil, "test-dgd", "test-ns", RoleMain, 1,
				&configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove,
				"worker", nil, staticContainerGPUCount(0),
			)

			t.Log("Preserve each complete probe override")
			require.NoError(t, err)
			require.Len(t, pod.Containers, 2)
			require.Equal(t, "frontend", pod.Containers[1].Name)
			require.Equal(t, override, pod.Containers[1].StartupProbe)
			require.Equal(t, override, pod.Containers[1].LivenessProbe)
			require.Equal(t, override, pod.Containers[1].ReadinessProbe)
		})
	}
}

func TestFrontendStartupProbeOriginPropagation(t *testing.T) {
	tests := []struct {
		name        string
		origin      string
		sidecar     bool
		wantEnabled bool
	}{
		{name: "missing graph origin preserves frontend defaults"},
		{name: "1.7 graph enables frontend default", origin: "1.7.0", wantEnabled: true},
		{name: "1.6 graph preserves frontend sidecar", origin: "1.6.0", sidecar: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a graph whose immutable origin overrides component metadata")
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "default"}
			if tt.origin != "" {
				dgd.Annotations = map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: tt.origin}
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{
					Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "2.0.0"},
					Spec:        corev1.PodSpec{Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "frontend:1.6.0"}}},
				},
			}
			frontendIndex := 0
			if tt.sidecar {
				component.ComponentName, component.ComponentType = "worker", v1beta1.ComponentTypeWorker
				component.PodTemplate.Spec.Containers[0].Image = "worker:1.5.0"
				component.FrontendSidecar = new("frontend")
				component.PodTemplate.Spec.Containers = append(component.PodTemplate.Spec.Containers, corev1.Container{Name: "frontend", Image: "frontend:1.6.0"})
				frontendIndex = 1
			}
			if tt.wantEnabled {
				component.PodTemplate.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = "1.6.0"
			}

			t.Log("Render through the production graph path, including frontend sidecar lowering")
			pod, err := GeneratePodSpecForComponent(component, BackendFrameworkVLLM, nil, dgd, RoleMain, 1,
				&configv1alpha1.OperatorConfiguration{}, commonconsts.MultinodeDeploymentTypeGrove, component.ComponentName, nil, staticContainerGPUCount(0))
			require.NoError(t, err)

			t.Log("Preserve old graph defaults and propagate the authoritative new origin")
			probe := pod.Containers[frontendIndex].StartupProbe
			require.Equal(t, tt.wantEnabled, probe != nil)
		})
	}
}

func TestApplyFrontendDrainBudget(t *testing.T) {
	const propagationEnv = "DYN_FRONTEND_PROPAGATION_DELAY_SECS"
	const drainEnv = "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS"

	drainValue := func(container corev1.Container) (string, bool) {
		for _, env := range container.Env {
			if env.Name == drainEnv {
				return env.Value, true
			}
		}
		return "", false
	}

	tests := []struct {
		name          string
		componentType v1beta1.ComponentType
		grace         *int64
		env           []corev1.EnvVar
		wantDrain     string // expected derived drain value; "" means no drain env expected
		wantInjected  bool
	}{
		{
			name:          "O1 default grace and propagation derive 50",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         nil, // default 60
			wantDrain:     "50",
			wantInjected:  true,
		},
		{
			name:          "O2 larger grace period derives 110",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(120)),
			wantDrain:     "110",
			wantInjected:  true,
		},
		{
			name:          "O3 user-set propagation shrinks the derived drain",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(60)),
			env:           []corev1.EnvVar{{Name: propagationEnv, Value: "15"}},
			wantDrain:     "40",
			wantInjected:  true,
		},
		{
			name:          "O4 user-set drain is never overridden",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(60)),
			env:           []corev1.EnvVar{{Name: drainEnv, Value: "200"}},
			wantDrain:     "200",
			wantInjected:  true,
		},
		{
			name:          "O5 grace period smaller than the derived minimum leaves the frontend default",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(8)),
			wantInjected:  false,
		},
		{
			name:          "grace period just large enough derives a 1s drain",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(11)),
			wantDrain:     "1",
			wantInjected:  true,
		},
		{
			name:          "Grove-sized pod grace period leaves the frontend default",
			componentType: v1beta1.ComponentTypeFrontend,
			grace:         new(int64(10)),
			wantInjected:  false,
		},
		{
			name:          "non-frontend components are unchanged",
			componentType: v1beta1.ComponentTypeWorker,
			grace:         new(int64(60)),
			wantInjected:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render a single-container pod spec with the test's grace period and env")
			podSpec := &corev1.PodSpec{
				TerminationGracePeriodSeconds: tt.grace,
				Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Env: tt.env},
				},
			}

			t.Log("Apply the frontend drain budget")
			applyFrontendDrainBudget(podSpec, tt.componentType)

			t.Log("Check the derived or preserved drain timeout")
			value, present := drainValue(podSpec.Containers[0])
			require.Equal(t, tt.wantInjected, present)
			if tt.wantInjected {
				require.Equal(t, tt.wantDrain, value)
			}
		})
	}
}

func TestApplyFrontendDrainBudgetIgnoresValueFrom(t *testing.T) {
	t.Log("A drain timeout sourced from valueFrom is left untouched")
	podSpec := &corev1.PodSpec{
		TerminationGracePeriodSeconds: new(int64(60)),
		Containers: []corev1.Container{
			{
				Name: commonconsts.MainContainerName,
				Env: []corev1.EnvVar{
					{
						Name:      "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS",
						ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}},
					},
				},
			},
		},
	}

	applyFrontendDrainBudget(podSpec, v1beta1.ComponentTypeFrontend)

	t.Log("The env keeps its single valueFrom entry; no literal drain is appended")
	require.Len(t, podSpec.Containers[0].Env, 1)
	require.NotNil(t, podSpec.Containers[0].Env[0].ValueFrom)
}
