/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	corev1 "k8s.io/api/core/v1"
)

// podEnvironment retains source layers through backend, GMS and failover
// transformations. A nil collector selects the unchanged legacy render path.
type podEnvironment struct {
	general        []corev1.EnvVar
	containers     semanticenv.ContainerLayers
	overrides      []map[string]semanticenv.Layer
	policyRuntime  string
	policyBindings []corev1.EnvVar
}

func newPodEnvironment(annotations map[string]string, general []corev1.EnvVar) *podEnvironment {
	if !compatibility.OrderedEnvironmentVariables.Enabled(annotations) {
		return nil
	}
	return &podEnvironment{general: general, containers: make(semanticenv.ContainerLayers)}
}

func (p *podEnvironment) register(name, source string, defaults, general, authored []corev1.EnvVar) {
	if p == nil {
		return
	}
	p.containers[name] = []semanticenv.Layer{
		{Source: "operator.defaults." + name, Env: defaults},
		{Source: "spec.env", Env: general},
		{Source: source, Env: slices.Clone(authored)},
	}

	// Only the compiled policy's runtime omits graph-level policy inputs.
	if name == p.policyRuntime {
		p.containers[name][1] = p.containers[name][1].Without(workerKvTransferPolicyEnvNames...)
		original := p.containers[name][2].Without(workerKvTransferPolicyEnvNames...)
		p.containers[name] = append(p.containers[name][:2],
			semanticenv.Layer{Source: "operator.kvTransferPolicy", Env: p.policyBindings}, original)
	}
	for _, overrides := range p.overrides {
		if layer, exists := overrides[name]; exists {
			p.containers[name] = append(p.containers[name], layer)
		}
	}
}

// DCDHasKVTransferPolicy recognizes a graph-compiled policy from its existing
// topology-source snapshot, not from standalone DCD topology annotations.
// dcd must not be nil.
func DCDHasKVTransferPolicy(dcd *v1beta1.DynamoComponentDeployment) bool {
	return dcd.GetParentGraphDeploymentName() != "" &&
		(dcd.Annotations[commonconsts.KubeAnnotationTopologyLabelKey] != "" ||
			dcd.Annotations[commonconsts.KubeAnnotationTopologyClusterTopologyName] != "")
}

// ownKVTransferPolicy scopes graph-default removal to the container publishing
// the compiled policy. Compute-only roles have no such runtime container.
func (p *podEnvironment) ownKVTransferPolicy(component *v1beta1.DynamoComponentDeploymentSharedSpec, nativeSidecar bool, bindings []corev1.EnvVar) {
	if p == nil || len(bindings) == 0 {
		return
	}
	if GetDynamoSidecar(component) != nil {
		p.policyRuntime = commonconsts.RuntimeContainerName
	} else if !nativeSidecar {
		p.policyRuntime = commonconsts.MainContainerName
	} else {
		return
	}
	p.policyBindings = bindings
}

// DCDKVTransferPolicyEnvironment reads the policy snapshot stored in a child
// DCD's selected runtime. Standalone DCDs do not own graph policy bindings.
// dcd and component must not be nil.
func DCDKVTransferPolicyEnvironment(dcd *v1beta1.DynamoComponentDeployment, component *v1beta1.DynamoComponentDeploymentSharedSpec) []corev1.EnvVar {
	if !DCDHasKVTransferPolicy(dcd) {
		return nil
	}
	runtime := GetDynamoSidecar(component)
	if runtime == nil {
		runtime = GetMainContainer(component)
	}
	if runtime == nil {
		return nil
	}
	var bindings []corev1.EnvVar
	for _, value := range runtime.Env {
		if slices.Contains(workerKvTransferPolicyEnvNames, value.Name) {
			bindings = append(bindings, value)
		}
	}
	return bindings
}

func (p *podEnvironment) appendDefaults(container *corev1.Container, values []corev1.EnvVar) {
	if p != nil {
		p.containers[container.Name][0].Env = slices.Concat(p.containers[container.Name][0].Env, values)
	}
	container.Env = append(container.Env, values...)
}

// staged exposes raw, last-definition-wins inputs to backend readers without
// prematurely sorting or expanding them. p must not be nil.
func (p *podEnvironment) staged(name string) []corev1.EnvVar {
	var result []corev1.EnvVar
	for _, layer := range p.containers[name] {
		result = append(result, layer.Env...)
	}
	return result
}

func (p *podEnvironment) bind(container *corev1.Container, source string, values []corev1.EnvVar) {
	var layers semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Bind(container, source, values)
}

func (p *podEnvironment) clone(from, to *corev1.Container, omitted []string) {
	var layers semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Clone(from, to, omitted)
}

func (p *podEnvironment) omit(container *corev1.Container, omitted ...string) {
	var layers semanticenv.ContainerLayers
	if p != nil {
		layers = p.containers
	}
	layers.Omit(container, omitted...)
}

// capture registers remaining authored containers before provider mutations.
// pod must not be nil. A nil collector selects legacy rendering.
func (p *podEnvironment) capture(pod *corev1.PodSpec) semanticenv.ContainerLayers {
	if p == nil {
		return nil
	}
	for _, group := range []struct {
		name       string
		containers []corev1.Container
	}{{"containers", pod.Containers}, {"initContainers", pod.InitContainers}} {
		for i := range group.containers {
			container := &group.containers[i]
			_, registered := p.containers[container.Name]
			if !registered {
				p.register(container.Name, "podTemplate.spec."+group.name+"[name="+container.Name+"].env", nil, nil, container.Env)
			}
		}
	}
	return p.containers
}

// apply finalizes containers when there are no remaining provider mutations.
// pod must not be nil. A nil collector leaves legacy rendering unchanged.
func (p *podEnvironment) apply(pod *corev1.PodSpec) error {
	return p.capture(pod).Apply(pod)
}
