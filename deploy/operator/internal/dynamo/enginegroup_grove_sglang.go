/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/utils/ptr"
)

func (r groveEngineGroupRenderer) renderSGLang(
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Validate the explicit native template before constructing Grove metadata.
	profile, err := ResolveSGLangComponentEngineGroup(component, dgd.Spec.Env)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	component = component.DeepCopy()
	propagateDGDAnnotations(dgd.Annotations, component)
	propagateDGDSpecMetadata(dgd.Spec.Annotations, dgd.Spec.Labels, component)
	podTemplate := ensurePodTemplate(component)
	podTemplate.Labels[consts.KubeLabelDynamoNamespace] = GetDynamoNamespace(dgd, component)
	discovery := controller_common.GetDiscoveryBackend(r.operatorConfig.Discovery.Backend, dgd.Annotations)
	if discovery != "" {
		podTemplate.Annotations[consts.KubeAnnotationDynamoDiscoveryBackend] = string(discovery)
	}

	// Reuse standard Pod construction; native bootstrap arguments remain user-authored.
	role := ServiceRole{Name: component.ComponentName, Role: RoleMain, Replicas: profile.InitialReplicas}
	clique, err := buildCliqueForRole(cliqueParams{
		r: role, component: component, dynamoDeployment: dgd,
		backendFramework: BackendFrameworkSGLang, secretsRetriever: r.secretsRetriever, numberOfNodes: 1,
		operatorConfig: r.operatorConfig, runtimeConfig: r.runtimeConfig, componentName: component.ComponentName,
		usesPCSG: true, discoveryBackend: discovery,
		discoveryContext: NewDiscoveryContext(r.operatorConfig.Discovery.Backend, dgd.Annotations),
		restartState:     restart, existingRestartAnnotations: restartAnnotations,
		validatedQueueName: queueName, groveClusterTopologyDomains: topologyDomains,
		containerGPUs: func() (int64, error) { return profile.Geometry.GPUsPerReplica, nil },
	})
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	configureGroveEngineGroupContainer(&clique.Spec.PodSpec.Containers[0], GetMainContainer(component))
	clique.Spec.MinAvailable = ptr.To(profile.InitialReplicas)
	clique.Labels[consts.KubeLabelDynamoEngineGroup] = EngineGroupNameForComponent(dgd.Name, component.ComponentName, 0)
	clique.Labels[consts.KubeLabelDynamoScaleRepresentative] = consts.KubeLabelDynamoScaleRepresentativeYes

	// PCSG replicas count worlds, independently of this clique's allocation count.
	world := buildGroveScalingGroupConfig(component.ComponentName, component, dgd.Spec.TopologyConstraint,
		[]ServiceRole{role}, []string{clique.Name}, nil, false)
	return clique, world, nil
}
