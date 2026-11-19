/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestGroveEngineGroupReservedEnvironment(t *testing.T) {
	for _, owner := range []string{"main", "sidecar", "parent"} {
		t.Run(owner, func(t *testing.T) {
			t.Log("declare provider-owned environment independently of engine geometry")
			variable := corev1.EnvVar{Name: "GROVE_PCLQ_POD_INDEX", Value: "0"}
			template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever,
				Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "sidecar"}}}}
			var parent []corev1.EnvVar
			switch owner {
			case "main":
				template.Spec.Containers[0].Env = []corev1.EnvVar{variable}
			case "sidecar":
				template.Spec.Containers[1].Env = []corev1.EnvVar{variable}
			case "parent":
				parent = []corev1.EnvVar{variable}
			}
			before := template.DeepCopy()

			t.Log("keep common Kubernetes validation independent, and reject the conflict at the Grove boundary")
			require.NoError(t, validateEngineGroupPodTemplate(template))
			require.ErrorContains(t, ValidateGroveEngineGroupTemplate(template, parent), "supplied by Grove")
			assert.Equal(t, before, template)
		})
	}
}

func TestGroveEngineGroupRetentionGate(t *testing.T) {
	for _, count := range []int{0, 1, 2} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			t.Log("declare readiness-steered release independently of engine flags")
			template := &corev1.PodTemplateSpec{}
			for range count {
				template.Spec.ReadinessGates = append(template.Spec.ReadinessGates, corev1.PodReadinessGate{
					ConditionType: corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate),
				})
			}
			before := template.DeepCopy()

			t.Log("require exactly one gate without modifying the explicit template")
			err := RequireGroveEngineGroupRetentionGate(template)
			if count == 1 {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, "exactly one")
			}
			assert.Equal(t, before, template)
		})
	}
}
