/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// ValidateGroveEngineGroupTemplate checks provider-owned environment before Grove rendering.
// template must not be nil. parentEnv may be nil. Inputs are not mutated.
func ValidateGroveEngineGroupTemplate(template *corev1.PodTemplateSpec, parentEnv []corev1.EnvVar) error {
	for _, container := range template.Spec.Containers {
		environment := container.Env
		if container.Name == consts.MainContainerName {
			environment = MergeEnvs(parentEnv, environment)
		}
		for _, variable := range environment {
			if strings.HasPrefix(variable.Name, "GROVE_") {
				return fmt.Errorf("environment %s is supplied by Grove and cannot be shadowed in the Pod template", variable.Name)
			}
		}
	}
	return nil
}

// RequireGroveEngineGroupRetentionGate checks the explicit readiness-steered release prerequisite.
// template must not be nil.
func RequireGroveEngineGroupRetentionGate(template *corev1.PodTemplateSpec) error {
	count := 0
	for _, gate := range template.Spec.ReadinessGates {
		if gate.ConditionType == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate) {
			count++
		}
	}
	if count != 1 {
		return fmt.Errorf("Engine Group podTemplate must declare exactly one %s readiness gate", consts.EngineGroupRetainedReadinessGate)
	}
	return nil
}
