// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestSnapshotFailoverSGLangValidatesFinalEnvironmentWithoutScrapePorts(t *testing.T) {
	for _, origin := range []string{"1.7.0", "1.8.0"} {
		for _, test := range []struct {
			name      string
			port      corev1.EnvVar
			wantError string
		}{
			{name: "malformed literal", port: corev1.EnvVar{Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", Value: "invalid"}, wantError: "not a number"},
			{name: "out of range literal", port: corev1.EnvVar{Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", Value: "70000"}, wantError: "outside the port range"},
			{name: "known literal", port: corev1.EnvVar{Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", Value: "30500"}},
			{name: "unknown runtime source", port: corev1.EnvVar{Name: "NIXL_TELEMETRY_PROMETHEUS_PORT", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "settings", Key: "port"}}}},
		} {
			t.Run(origin+"/"+test.name, func(t *testing.T) {
				t.Log("Render a snapshot-backed SGLang failover pair through the production entrypoint")
				component := &v1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentType: v1beta1.ComponentTypeWorker,
					Experimental: &v1beta1.ExperimentalSpec{
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod},
						Failover:         &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeIntraPod},
						Checkpoint:       &v1beta1.ComponentCheckpointConfig{Enabled: true},
					},
					PodTemplate: &corev1.PodTemplateSpec{Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin},
						Spec: corev1.PodSpec{Containers: []corev1.Container{{
							Name: "main", Image: "runtime:1.8.0", Command: []string{"python3", "-m", "dynamo.sglang"},
							Env:       []corev1.EnvVar{{Name: "NIXL_TELEMETRY_ENABLE", Value: "y"}, {Name: "NIXL_TELEMETRY_EXPORTER", Value: "prometheus"}, test.port},
							Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{consts.KubeResourceGPUNvidia: resource.MustParse("2")}},
						}}}},
				}
				original := component.DeepCopy()
				pod, err := GenerateBasePodSpec(component, BackendFrameworkSGLang, nil, "graph", "ns", RoleMain, 1,
					&configv1alpha1.OperatorConfiguration{}, consts.MultinodeDeploymentTypeGrove, "worker", nil, staticContainerGPUCount(2))
				wantError := test.wantError
				if origin == "1.7.0" && test.port.ValueFrom != nil {
					wantError = "is set through valueFrom"
				}
				if wantError != "" {
					require.ErrorContains(t, err, wantError)
				} else {
					require.NoError(t, err)
					require.Len(t, pod.Containers, 2)
					for _, engine := range pod.Containers {
						require.Nil(t, findContainerPort(&engine, consts.DynamoNixlPortName), "validation must not recreate deliberately omitted scrape ports")
					}
				}
				require.Equal(t, original, component)
			})
		}
	}
}
