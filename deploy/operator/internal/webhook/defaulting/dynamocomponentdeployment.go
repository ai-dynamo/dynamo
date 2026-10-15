/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
)

func defaultDynamoComponentDeployment(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	operation admissionv1.Operation,
) {
	if operation == admissionv1.Create && dcd.Spec.ComponentName == "" && dcd.Name != "" {
		dcd.Spec.ComponentName = dcd.Name
	}
	defaultDynamoComponentDeploymentSpec(&dcd.Spec)
}

func defaultDynamoComponentDeploymentSpec(spec *nvidiacomv1beta1.DynamoComponentDeploymentSpec) {
	defaultDynamoComponentDeploymentSharedSpec(
		&spec.DynamoComponentDeploymentSharedSpec,
		dynamoComponentDeploymentSharedSpecDefaultOptions{},
	)
}
