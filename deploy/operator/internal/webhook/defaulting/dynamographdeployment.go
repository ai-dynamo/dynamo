/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	admissionv1 "k8s.io/api/admission/v1"
)

type dynamoGraphDeploymentDefaultOptions struct {
	operation       admissionv1.Operation
	operatorVersion string
	groveEnabled    bool
}

func defaultDynamoGraphDeployment(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	opts dynamoGraphDeploymentDefaultOptions,
) {
	provider, providerSelected := workloadProviderForDynamoGraphDeployment(dgd, opts)

	if opts.operation == admissionv1.Create {
		if dgd.Annotations == nil {
			dgd.Annotations = make(map[string]string)
		}

		// These annotations are authoritative admission state, not user defaults.
		// Replace submitted values so a new DGD records the creating operator and
		// the provider selected from the request's routing intent.
		dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = opts.operatorVersion
		dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = provider
	}

	legacyMinimum := provideroverride.HasLegacyGroveMinAvailable(dgd)
	providerMinimum := !legacyMinimum && (provideroverride.HasGroveMinAvailableOverrides(dgd) ||
		compatibility.GroveNativeMinAvailable.Enabled(dgd.Annotations))
	defaultDynamoGraphDeploymentSpec(&dgd.Spec, dynamoGraphDeploymentSpecDefaultOptions{
		provider:         provider,
		providerSelected: providerSelected,
		providerMinimum:  providerMinimum,
	})
}

type dynamoGraphDeploymentSpecDefaultOptions struct {
	provider         string
	providerSelected bool
	providerMinimum  bool
}

func defaultDynamoGraphDeploymentSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec,
	opts dynamoGraphDeploymentSpecDefaultOptions,
) {
	if spec.ProviderOverride != nil {
		provideroverride.DefaultTarget(spec.ProviderOverride, opts.provider, provideroverride.ScopeRoot, nil)
	}

	for i := range spec.Components {
		defaultDynamoComponentDeploymentSharedSpec(
			&spec.Components[i],
			dynamoComponentDeploymentSharedSpecDefaultOptions{
				defaultReplicas:        true,
				defaultProviderTargets: true,
				provider:               opts.provider,
				providerSelected:       opts.providerSelected,
				providerMinimum:        opts.providerMinimum,
			},
		)
	}
}
