/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
)

type dynamoComponentDeploymentSharedSpecDefaultOptions struct {
	defaultReplicas        bool
	defaultProviderTargets bool
	provider               string
	providerSelected       bool
	providerMinimum        bool
}

func defaultDynamoComponentDeploymentSharedSpec(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	opts dynamoComponentDeploymentSharedSpecDefaultOptions,
) {
	if opts.defaultProviderTargets && component.ProviderOverride != nil {
		provideroverride.DefaultTarget(
			component.ProviderOverride,
			opts.provider,
			provideroverride.ScopeComponent,
			component,
		)
	}

	if opts.defaultReplicas && !component.IsLPX() && component.Replicas == nil {
		component.Replicas = new(int32(1))
	}
	if opts.providerSelected && opts.provider == consts.WorkloadProviderGrove &&
		!opts.providerMinimum && component.MinAvailable == nil {
		component.MinAvailable = new(int32(1))
	}

	for i := range component.Roles {
		defaultComponentRoleSpec(&component.Roles[i], componentRoleSpecDefaultOptions{
			component:              component,
			defaultProviderTargets: opts.defaultProviderTargets,
			provider:               opts.provider,
		})
	}
}

type componentRoleSpecDefaultOptions struct {
	component              *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec
	defaultProviderTargets bool
	provider               string
}

func defaultComponentRoleSpec(
	role *nvidiacomv1beta1.ComponentRoleSpec,
	opts componentRoleSpecDefaultOptions,
) {
	if opts.component.Multinode != nil && role.Replicas == nil {
		switch role.Name {
		case nvidiacomv1beta1.ComponentRoleLeader:
			role.Replicas = new(int32(1))
		case nvidiacomv1beta1.ComponentRoleWorker:
			if opts.component.Multinode.NodeCount >= 2 {
				role.Replicas = new(opts.component.Multinode.NodeCount - 1)
			}
		}
	}

	if !opts.defaultProviderTargets || role.ProviderOverride == nil {
		return
	}
	scope, ok := provideroverride.ScopeForComponentRole(role.Name)
	if ok {
		provideroverride.DefaultTarget(role.ProviderOverride, opts.provider, scope, opts.component)
	}
}
