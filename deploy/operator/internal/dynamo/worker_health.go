/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

const dynHealthCheckEnabledEnv = "DYN_HEALTH_CHECK_ENABLED"

func setWorkerHealthCheckDefault(container *corev1.Container, component *v1beta1.DynamoComponentDeploymentSharedSpec, enabled bool) {
	if container == nil || component == nil || !IsWorkerComponent(string(component.ComponentType)) {
		return
	}
	if main := GetMainContainer(component); main != nil && hasEnvVarNamed(main.Env, dynHealthCheckEnabledEnv) {
		return
	}
	setEnvVar(container, dynHealthCheckEnabledEnv, boolString(enabled))
}

func boolString(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func isDecodeWorkerMode(container *corev1.Container, component *v1beta1.DynamoComponentDeploymentSharedSpec, modeEnvName, legacyDecodeEnvName string) bool {
	if component != nil && component.ComponentType == v1beta1.ComponentTypeDecode {
		return true
	}
	if hasFlagValue(container, "--disaggregation-mode", "decode") || hasCLIFlag(container, "--is-decode-worker") {
		return true
	}
	if modeEnvName != "" && envValueEquals(container.Env, modeEnvName, "decode") {
		return true
	}
	return legacyDecodeEnvName != "" && hasTruthyEnv(container.Env, legacyDecodeEnvName)
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

func hasTruthyEnv(envs []corev1.EnvVar, name string) bool {
	for _, env := range envs {
		if env.Name == name {
			switch strings.ToLower(env.Value) {
			case "1", "true", "yes", "y", "on":
				return true
			}
		}
	}
	return false
}

func envValueEquals(envs []corev1.EnvVar, name, value string) bool {
	for _, env := range envs {
		if env.Name == name && strings.EqualFold(env.Value, value) {
			return true
		}
	}
	return false
}

func hasCLIFlag(container *corev1.Container, flag string) bool {
	for _, token := range containerCommandTokens(container) {
		if token == flag {
			return true
		}
	}
	return false
}

func hasFlagValue(container *corev1.Container, flag, value string) bool {
	tokens := containerCommandTokens(container)
	for i, token := range tokens {
		if token == flag && i+1 < len(tokens) && strings.EqualFold(tokens[i+1], value) {
			return true
		}
		if strings.HasPrefix(token, flag+"=") && strings.EqualFold(strings.TrimPrefix(token, flag+"="), value) {
			return true
		}
	}
	return false
}

func containerCommandTokens(container *corev1.Container) []string {
	if container == nil {
		return nil
	}
	tokens := make([]string, 0, len(container.Command)+len(container.Args))
	for _, part := range append(append([]string{}, container.Command...), container.Args...) {
		tokens = append(tokens, strings.Fields(part)...)
	}
	return tokens
}
