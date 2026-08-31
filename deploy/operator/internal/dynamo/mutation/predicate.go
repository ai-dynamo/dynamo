/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
)

// LeaderRole selects mutations for the multinode leader.
func LeaderRole(_ *v1beta1.DynamoComponentDeploymentSharedSpec, role workload.Role) bool {
	return role == workload.RoleLeader
}

// WorkerRole selects mutations for multinode workers.
func WorkerRole(_ *v1beta1.DynamoComponentDeploymentSharedSpec, role workload.Role) bool {
	return role == workload.RoleWorker
}
