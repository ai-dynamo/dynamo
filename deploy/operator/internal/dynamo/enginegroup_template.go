/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// validateEngineGroupPodTemplate checks the explicit allocation lifecycle without supplying defaults.
// template must not be nil.
func validateEngineGroupPodTemplate(template *corev1.PodTemplateSpec) error {
	if template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		return fmt.Errorf("Engine Group podTemplate.spec.restartPolicy must explicitly be Never")
	}
	for _, container := range template.Spec.Containers {
		for _, variable := range container.Env {
			if container.Name == consts.MainContainerName &&
				(variable.Name == consts.DynamoRankEnvVar || variable.Name == consts.DynamoLeaderAddressEnvVar) {
				return fmt.Errorf("environment %s is operator-owned topology and cannot be shadowed in the main container", variable.Name)
			}
		}
	}
	return nil
}

// requireEngineGroupFieldEnv binds an identity to Kubernetes metadata, never to a caller-supplied incarnation.
// main must not be nil.
func requireEngineGroupFieldEnv(main *corev1.Container, name, field string) error {
	for _, variable := range main.Env {
		if variable.Name == name {
			if variable.Value != "" || variable.ValueFrom == nil || variable.ValueFrom.FieldRef == nil ||
				variable.ValueFrom.FieldRef.FieldPath != field ||
				(variable.ValueFrom.FieldRef.APIVersion != "" && variable.ValueFrom.FieldRef.APIVersion != "v1") {
				return fmt.Errorf("Engine Group main environment %s must use the Downward API field %s", name, field)
			}
			return nil
		}
	}
	return fmt.Errorf("Engine Group main environment %s must be explicit using the Downward API field %s", name, field)
}
