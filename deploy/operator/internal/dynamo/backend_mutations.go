/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"

func leaderRole(_ *v1beta1.DynamoComponentDeploymentSharedSpec, role Role) bool {
	return role == RoleLeader
}

func workerRole(_ *v1beta1.DynamoComponentDeploymentSharedSpec, role Role) bool {
	return role == RoleWorker
}

func interPodGMS(component *v1beta1.DynamoComponentDeploymentSharedSpec, _ Role) bool {
	return component != nil && component.IsInterPodGMSEnabled()
}

func interPodFailover(component *v1beta1.DynamoComponentDeploymentSharedSpec, _ Role) bool {
	return component != nil && component.IsInterPodFailoverEnabled()
}
