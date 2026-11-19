/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
)

func workloadProviderForDynamoGraphDeployment(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	opts dynamoGraphDeploymentDefaultOptions,
) (string, bool) {
	if opts.operation != admissionv1.Create {
		provider, exists := dgd.Annotations[consts.KubeAnnotationWorkloadProvider]
		return provider, exists
	}

	provider := consts.WorkloadProviderComponent
	if opts.groveEnabled &&
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableGrove]) != consts.KubeLabelValueFalse {
		provider = consts.WorkloadProviderGrove
	}
	if provider == consts.WorkloadProviderComponent &&
		strings.ToLower(dgd.Annotations[consts.KubeAnnotationEnableDisaggregatedSet]) == consts.KubeLabelValueTrue {
		provider = consts.WorkloadProviderDisaggregatedSet
	}
	return provider, true
}
