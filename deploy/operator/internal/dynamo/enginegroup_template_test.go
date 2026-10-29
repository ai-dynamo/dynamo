/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestEngineGroupTopologyAliasesAreOperatorOwned(t *testing.T) {
	for _, name := range []string{consts.DynamoRankEnvVar, consts.DynamoLeaderAddressEnvVar} {
		t.Run(name, func(t *testing.T) {
			t.Log("attempt to shadow an allocation topology alias in the explicit template")
			template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				RestartPolicy: corev1.RestartPolicyNever,
				Containers:    []corev1.Container{{Name: consts.MainContainerName, Env: []corev1.EnvVar{{Name: name, Value: "wrong"}}}},
			}}
			before := template.DeepCopy()

			t.Log("reject the conflicting declaration rather than silently overwriting it")
			require.ErrorContains(t, validateEngineGroupPodTemplate(template), "operator-owned topology")
			assert.Equal(t, before, template)
		})
	}
}

func TestEngineGroupIncarnationRequiresDownwardIdentity(t *testing.T) {
	for _, tc := range []struct {
		name      string
		value     string
		field     string
		version   string
		wantError bool
	}{
		{"implicit API version", "", "metadata.uid", "", false},
		{"explicit API version", "", "metadata.uid", "v1", false},
		{"caller supplied incarnation", "stale", "metadata.uid", "", true},
		{"name is not lifetime identity", "", "metadata.name", "", true},
		{"wrong API version", "", "metadata.uid", "v2", true},
		{"missing field", "", "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("declare native process identity through Kubernetes metadata")
			main := &corev1.Container{Name: consts.MainContainerName, Env: []corev1.EnvVar{{
				Name: "PROCESS_UID", Value: tc.value,
				ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: tc.version, FieldPath: tc.field}},
			}}}
			before := main.DeepCopy()

			t.Log("accept only the exact Downward API identity without modifying the launch")
			err := requireEngineGroupFieldEnv(main, "PROCESS_UID", "metadata.uid")
			if tc.wantError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, before, main)
		})
	}
}
