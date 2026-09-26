// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation

import (
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func ordinaryGroveComponents(components []v1beta1.DynamoComponentDeploymentSharedSpec) []v1beta1.DynamoComponentDeploymentSharedSpec {
	return slices.DeleteFunc(
		slices.Clone(components),
		func(component v1beta1.DynamoComponentDeploymentSharedSpec) bool {
			return component.ManagedByExternalController()
		},
	)
}
