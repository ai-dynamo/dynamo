/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

const (
	dynHealthCheckEnabledEnv = "DYN_HEALTH_CHECK_ENABLED"
	envValueFalse            = "false"
	envValueTrue             = "true"
)

func setWorkerHealthCheckDefault(container *corev1.Container, component *v1beta1.DynamoComponentDeploymentSharedSpec) {
	if container == nil || component == nil || !IsWorkerComponent(string(component.ComponentType)) {
		return
	}
	if main := GetMainContainer(component); main != nil && hasEnvVarNamed(main.Env, dynHealthCheckEnabledEnv) {
		return
	}
	setEnvVar(container, dynHealthCheckEnabledEnv, envValueTrue)
}

func setEnvVar(container *corev1.Container, name, value string) {
	for i := range container.Env {
		if container.Env[i].Name == name {
			container.Env[i].Value = value
			container.Env[i].ValueFrom = nil
			return
		}
	}
	container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: value})
}

func hasEnvVarNamed(envs []corev1.EnvVar, name string) bool {
	for _, env := range envs {
		if env.Name == name {
			return true
		}
	}
	return false
}
