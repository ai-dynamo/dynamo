/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestDCDEnvironmentLayerConversion(t *testing.T) {
	t.Log("Keep general defaults distinct from the specific container environment")
	hub := &v1beta1.DynamoComponentDeployment{
		Name: "worker",
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			Env: []corev1.EnvVar{{Name: "BASE", Value: "/models"}, {Name: "MODEL", Value: "$(BASE)/qwen"}},
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name: "main", Image: "engine:1.7.0", Env: []corev1.EnvVar{{Name: "BASE", Value: "/fast"}},
				}}}},
			},
		},
	}
	var spoke DynamoComponentDeployment
	require.NoError(t, spoke.ConvertFrom(hub))
	require.Equal(t, hub.Spec.Env, spoke.Spec.Env)
	roundTrip := &v1beta1.DynamoComponentDeployment{}
	require.NoError(t, spoke.ConvertTo(roundTrip))
	require.Equal(t, hub.Spec.Env, roundTrip.Spec.Env)
	require.Equal(t, hub.Spec.PodTemplate, roundTrip.Spec.PodTemplate)

	t.Log("A live alpha edit wins without flattening or restoring a stale environment")
	spoke.Spec.Env = []corev1.EnvVar{{Name: "BASE", ValueFrom: &corev1.EnvVarSource{
		ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "paths", Key: "base"},
	}}}
	require.NoError(t, spoke.ConvertTo(roundTrip))
	require.Equal(t, spoke.Spec.Env, roundTrip.Spec.Env)
	require.Equal(t, hub.Spec.PodTemplate, roundTrip.Spec.PodTemplate)

	t.Log("Clearing the represented alpha field must not revive the old hub value")
	spoke.Spec.Env = nil
	require.NoError(t, spoke.ConvertTo(roundTrip))
	require.Nil(t, roundTrip.Spec.Env)
	require.Equal(t, "/models", hub.Spec.Env[0].Value)
}
