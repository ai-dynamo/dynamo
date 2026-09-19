// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// projectOrdinaryGroveDeployment returns the component subset materialized by
// the DGD-owned Grove PodCliqueSet. The source is not mutated.
func projectOrdinaryGroveDeployment(source *v1beta1.DynamoGraphDeployment) *v1beta1.DynamoGraphDeployment {
	ordinary := source.DeepCopy()
	ordinary.Spec.Components = slices.DeleteFunc(
		ordinary.Spec.Components,
		func(component v1beta1.DynamoComponentDeploymentSharedSpec) bool {
			return component.ManagedByExternalController()
		},
	)
	return ordinary
}
