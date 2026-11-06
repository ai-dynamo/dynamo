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
	policyAuthored *semanticenv.Layer
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
		var bindings []corev1.EnvVar
		for _, value := range authored {
			if slices.Contains(workerKvTransferPolicyEnvNames, value.Name) {
				bindings = append(bindings, value)
			}
		}
		original := *p.policyAuthored
		original.Source = source
		p.containers[name] = append(p.containers[name][:2],
			semanticenv.Layer{Source: "operator.kvTransferPolicy", Env: bindings}, original)
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
func (p *podEnvironment) ownKVTransferPolicy(component *v1beta1.DynamoComponentDeploymentSharedSpec, nativeSidecar bool) {
	if p == nil {
		return
	}
	if GetDynamoSidecar(component) != nil {
		p.policyRuntime = commonconsts.RuntimeContainerName
	} else if !nativeSidecar {
		p.policyRuntime = commonconsts.MainContainerName
	} else {
		return
	}
	layer := AuthoredRuntimeEnvironment(component).Without(workerKvTransferPolicyEnvNames...)
	p.policyAuthored = &layer
}

// AuthoredRuntimeEnvironment captures the selected runtime's source before
// graph defaults or compiled policy values are inserted. component must not be nil.
func AuthoredRuntimeEnvironment(component *v1beta1.DynamoComponentDeploymentSharedSpec) semanticenv.Layer {
	runtime := GetDynamoSidecar(component)
	if runtime == nil {
		runtime = GetMainContainer(component)
	}
	if runtime == nil {
		return semanticenv.Layer{}
	}
	return semanticenv.Layer{Env: runtime.Env}
}

func (p *podEnvironment) appendDefaults(name string, values []corev1.EnvVar) {
	if p != nil {
		p.containers[name][0].Env = slices.Concat(p.containers[name][0].Env, values)
	}
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

func (p *podEnvironment) bind(name, source string, values []corev1.EnvVar) {
	if p != nil {
		p.containers.Bind(name, source, values)
	}
}

func (p *podEnvironment) clone(from, to string, omitted []string) {
	if p == nil {
		return
	}
	p.containers.Clone(from, to, omitted)
}

func (p *podEnvironment) omit(name string, omitted ...string) {
	if p == nil {
		return
	}
	p.containers.Omit(name, omitted...)
}

// capture registers remaining authored containers before provider mutations.
// pod must not be nil. A nil collector selects legacy rendering.
func (p *podEnvironment) capture(pod *corev1.PodSpec) semanticenv.ContainerLayers {
	if p == nil {
		return nil
	}
	for _, containers := range [][]corev1.Container{pod.Containers, pod.InitContainers} {
		for i := range containers {
			container := &containers[i]
			_, registered := p.containers[container.Name]
			if !registered {
				p.register(container.Name, "podTemplate.containers[name="+container.Name+"].env", nil, nil, container.Env)
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
