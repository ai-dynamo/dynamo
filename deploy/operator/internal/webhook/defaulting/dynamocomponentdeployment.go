/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package defaulting

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
)

func defaultDynamoComponentDeployment(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	operation admissionv1.Operation,
	operatorVersion string,
) {
	if operation == admissionv1.Create && dcd.Spec.ComponentName == "" && dcd.Name != "" {
		dcd.Spec.ComponentName = dcd.Name
	}

	// Only standalone CREATEs acquire a creation origin. Graph children already
	// carry their parent's origin in the template, even when created much later.
	if operation == admissionv1.Create && dcd.GetParentGraphDeploymentName() == "" {
		if dcd.Annotations == nil {
			dcd.Annotations = map[string]string{}
		}
		dcd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = operatorVersion
	}

	defaultDynamoComponentDeploymentSpec(&dcd.Spec)
}

func defaultDynamoComponentDeploymentSpec(spec *nvidiacomv1beta1.DynamoComponentDeploymentSpec) {
	defaultDynamoComponentDeploymentSharedSpec(
		&spec.DynamoComponentDeploymentSharedSpec,
		dynamoComponentDeploymentSharedSpecDefaultOptions{},
	)
}
