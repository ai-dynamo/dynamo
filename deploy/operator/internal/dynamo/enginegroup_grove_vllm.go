/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grove "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
)

func (r groveEngineGroupRenderer) renderVLLM(
	ctx context.Context,
	dgd *api.DynamoGraphDeployment,
	component *api.DynamoComponentDeploymentSharedSpec,
	queueName string,
	topologyDomains []api.TopologyDomain,
	restart *RestartState,
	restartAnnotations map[string]string,
) (*grove.PodCliqueTemplateSpec, grove.PodCliqueScalingGroupConfig, error) {
	// Validate the explicit native template before constructing Grove metadata.
	_, profile, err := ResolveVLLMComponentEngineGroup(component, dgd.Spec.Env)
	if err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	// This Grove realization releases vLLM allocations through an explicit retention gate.
	if err := RequireGroveEngineGroupRetentionGate(component.PodTemplate); err != nil {
		return nil, grove.PodCliqueScalingGroupConfig{}, err
	}
	return r.renderWorld(ctx, dgd, component, profile.Geometry, profile.InitialReplicas,
		queueName, topologyDomains, restart, restartAnnotations)
}
