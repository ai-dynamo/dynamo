/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strconv"
	"strings"
	"unicode"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// ComponentRuntimeStatus is the provider-neutral runtime identity projected
// from the Pod that exposes a logical Dynamo component.
type ComponentRuntimeStatus struct {
	ServedModelName      string
	RuntimeComponentName string
	GPUPowerLimitWatts   *int64
}

// ResolveComponentRuntimeStatus projects Planner-relevant runtime facts from
// the fully rendered serving Pod without exposing PodTemplate structure to the
// Planner.
func ResolveComponentRuntimeStatus(
	podSpec *corev1.PodSpec,
	annotations map[string]string,
) ComponentRuntimeStatus {
	status := ComponentRuntimeStatus{}
	main := mainContainerFromPodSpec(podSpec)
	if main != nil {
		tokens := shellCommandLineTokens(main.Command, main.Args)
		status.ServedModelName = firstFlagValue(
			tokens,
			"--served-model-name",
			"--model-name",
			"--model",
			"--model-path",
		)
		if endpoint := firstFlagValue(tokens, "--endpoint"); endpoint != "" {
			if componentName, ok := runtimeComponentFromEndpoint(endpoint); ok {
				status.RuntimeComponentName = componentName
			}
		}
	}

	if raw := annotations[commonconsts.KubeAnnotationGPUPowerLimit]; raw != "" {
		if watts, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64); err == nil && watts > 0 {
			status.GPUPowerLimitWatts = ptr.To(watts)
		}
	}
	return status
}

// ResolveGroveComponentRuntimeStatuses selects each component's semantic
// serving role through the same role expansion used by Grove lowering.
func ResolveGroveComponentRuntimeStatuses(
	dgd *v1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
) map[string]ComponentRuntimeStatus {
	statuses := make(map[string]ComponentRuntimeStatus)
	if dgd == nil || pcs == nil {
		return statuses
	}

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		roles := expandRolesForComponent(
			component.ComponentName,
			component.Replicas,
			component.GetNumberOfNodes(),
			component,
		)
		servingRole, ok := servingRole(roles)
		if !ok {
			continue
		}
		clique := cliqueByName(pcs, strings.ToLower(servingRole.Name))
		if clique == nil {
			continue
		}
		statuses[component.ComponentName] = ResolveComponentRuntimeStatus(
			&clique.Spec.PodSpec,
			clique.Annotations,
		)
	}
	return statuses
}

func mainContainerFromPodSpec(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i]
		}
	}
	return nil
}

func shellCommandLineTokens(command, args []string) []string {
	parts := make([]string, 0, len(command)+len(args))
	parts = append(parts, command...)
	parts = append(parts, args...)
	tokens := make([]string, 0, len(parts))
	for _, part := range parts {
		tokens = append(tokens, splitShellWords(part)...)
	}
	return tokens
}

func splitShellWords(value string) []string {
	var tokens []string
	var current strings.Builder
	var quote rune
	escaped := false
	flush := func() {
		if current.Len() == 0 {
			return
		}
		tokens = append(tokens, current.String())
		current.Reset()
	}
	for _, r := range value {
		if escaped {
			current.WriteRune(r)
			escaped = false
			continue
		}
		if r == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			} else {
				current.WriteRune(r)
			}
			continue
		}
		if r == '\'' || r == '"' {
			quote = r
			continue
		}
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		current.WriteRune(r)
	}
	if escaped {
		current.WriteRune('\\')
	}
	flush()
	return tokens
}

func firstFlagValue(tokens []string, flags ...string) string {
	for _, flag := range flags {
		for i, token := range tokens {
			if token == flag && i+1 < len(tokens) {
				return tokens[i+1]
			}
			if value, ok := strings.CutPrefix(token, flag+"="); ok {
				return value
			}
		}
	}
	return ""
}

func runtimeComponentFromEndpoint(endpoint string) (string, bool) {
	endpoint = strings.TrimPrefix(endpoint, "dyn://")
	parts := strings.Split(endpoint, ".")
	if len(parts) != 3 || parts[1] == "" {
		return "", false
	}
	return parts[1], true
}

func servingRole(roles []ServiceRole) (ServiceRole, bool) {
	for _, role := range roles {
		if role.Role == RoleLeader {
			return role, true
		}
	}
	for _, role := range roles {
		if role.Role == RoleMain {
			return role, true
		}
	}
	return ServiceRole{}, false
}

func cliqueByName(pcs *grovev1alpha1.PodCliqueSet, name string) *grovev1alpha1.PodCliqueTemplateSpec {
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Name == name {
			return clique
		}
	}
	return nil
}
