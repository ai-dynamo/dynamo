/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
)

type dynamoGraphDeploymentRequestDefaultOptions struct {
	operation admissionv1.Operation
	image     string
}

func defaultDynamoGraphDeploymentRequest(
	dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	opts dynamoGraphDeploymentRequestDefaultOptions,
) {
	defaultDynamoGraphDeploymentRequestSpec(&dgdr.Spec, opts)
}

func defaultDynamoGraphDeploymentRequestSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestSpec,
	opts dynamoGraphDeploymentRequestDefaultOptions,
) {
	// Leave v1beta2 storage envelopes free of native v1beta1 defaults.
	if spec.V1Beta2 != nil {
		return
	}

	defaultDynamoGraphDeploymentRequestNativeSpec(&spec.DynamoGraphDeploymentRequestNativeSpec, opts)
}

func defaultDynamoGraphDeploymentRequestNativeSpec(
	spec *nvidiacomv1beta1.DynamoGraphDeploymentRequestNativeSpec,
	opts dynamoGraphDeploymentRequestDefaultOptions,
) {
	// Preserve native defaults that moved out of the structural schema.
	if spec.Backend == "" {
		spec.Backend = nvidiacomv1beta1.BackendTypeAuto
	}

	// Derive the image only on CREATE so updates can intentionally clear it.
	if opts.operation == admissionv1.Create && spec.Image == "" && opts.image != "" {
		spec.Image = opts.image
	}

	// Fill the remaining native defaults without replacing explicit values.
	if spec.SearchStrategy == "" {
		spec.SearchStrategy = nvidiacomv1beta1.SearchStrategyRapid
	}
	if spec.AutoApply == nil {
		spec.AutoApply = new(true)
	}
}
