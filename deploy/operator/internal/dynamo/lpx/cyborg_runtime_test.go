/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestConfiguredCyborgBatchSizeRejectsInvalidEnvironment(t *testing.T) {
	t.Log("Define invalid literal and field-sourced Cyborg batch-size variables")
	tests := []corev1.EnvVar{
		{Name: CyborgBatchSizeEnv, Value: "0"},
		{Name: CyborgBatchSizeEnv, Value: "invalid"},
		{Name: CyborgBatchSizeEnv, ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	for _, variable := range tests {
		t.Logf("Reject invalid Cyborg batch-size environment %+v", variable)
		_, err := configuredCyborgBatchSize(&corev1.Container{Env: []corev1.EnvVar{variable}}, 1)
		require.Error(t, err)
	}
}
