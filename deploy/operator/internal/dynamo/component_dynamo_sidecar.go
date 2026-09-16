// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"fmt"
	"strconv"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

// DynamoSidecarDefaults configures the runtime independently of engine health,
// model loading, inference canaries, and engine-owned NIXL telemetry.
type DynamoSidecarDefaults struct {
	*BaseComponentDefaults
}

func (d *DynamoSidecarDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	// Preserve the sidecar image entrypoint unless the user provides a command.
	container := d.getCommonContainer(context)
	container.Command = nil
	container.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
	container.Ports = []corev1.ContainerPort{{
		Name: commonconsts.DynamoSystemPortName, ContainerPort: int32(commonconsts.DynamoSystemPort), Protocol: corev1.ProtocolTCP,
	}}
	container.Env = append(container.Env,
		corev1.EnvVar{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(commonconsts.DynamoSystemPort)},
	)

	// Rollout isolation belongs to the worker runtime even though the engine is main.
	if context.WorkerHashSuffix != "" {
		container.Env = append(container.Env, corev1.EnvVar{Name: commonconsts.DynamoNamespaceWorkerSuffixEnvVar, Value: context.WorkerHashSuffix})
	}

	// Startup covers only the independent HTTP listener, not engine model loading.
	container.StartupProbe = &corev1.Probe{
		ProbeHandler:  corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromString(commonconsts.DynamoSystemPortName)}},
		PeriodSeconds: 2, TimeoutSeconds: 1, FailureThreshold: 30,
	}
	container.LivenessProbe = &corev1.Probe{
		ProbeHandler:  corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromString(commonconsts.DynamoSystemPortName)}},
		PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
	}
	container.ReadinessProbe = &corev1.Probe{
		ProbeHandler:  corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromString(commonconsts.DynamoSystemPortName)}},
		PeriodSeconds: 5, TimeoutSeconds: 4, FailureThreshold: 3,
	}
	return container, nil
}

// mergeDynamoSidecarDefaults merges defaults into the named native sidecar.
// podSpec and operatorConfig must not be nil.
func mergeDynamoSidecarDefaults(podSpec *corev1.PodSpec, name string, context ComponentContext, operatorConfig *configv1alpha1.OperatorConfiguration) error {
	// Resolve the exact init container while preserving all other pod-template entries.
	for i := range podSpec.InitContainers {
		user := &podSpec.InitContainers[i]
		if user.Name != name {
			continue
		}
		if user.RestartPolicy == nil || *user.RestartPolicy != corev1.ContainerRestartPolicyAlways {
			return fmt.Errorf("dynamoSidecar %q requires restartPolicy Always", name)
		}

		// User configuration overrides defaults, including entire probe handlers.
		defaults := &DynamoSidecarDefaults{&BaseComponentDefaults{}}
		base, err := defaults.GetBaseContainer(context)
		if err != nil {
			return err
		}
		if err := mergeContainerByName(&base, user); err != nil {
			return fmt.Errorf("merge dynamoSidecar %q: %w", name, err)
		}
		AddStandardEnvVars(&base, operatorConfig)
		AddTransportTLSEnvVars(&base, operatorConfig)
		podSpec.InitContainers[i] = base
		return nil
	}
	return fmt.Errorf("dynamoSidecar %q does not match any podTemplate init container", name)
}
