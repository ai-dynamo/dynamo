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
	if opts.operation == admissionv1.Create && spec.Image == "" && opts.image != "" {
		spec.Image = opts.image
	}
}
