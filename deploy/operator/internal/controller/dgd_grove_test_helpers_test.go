/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"

func testGroveReconcileRequest(dgd *nvidiacomv1beta1.DynamoGraphDeployment) groveReconcileRequest {
	return groveReconcileRequest{
		DGD: dgd,
		Managed: func(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
			return !component.ManagedByExternalController()
		},
	}
}

func testComponentByName(
	components []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	name string,
) *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	for i := range components {
		if components[i].ComponentName == name {
			return &components[i]
		}
	}
	return nil
}
