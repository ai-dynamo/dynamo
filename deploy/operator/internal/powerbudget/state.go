/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"

// ReplicaFenceClosed derives the replica fence from the persisted DGPB phase.
func ReplicaFenceClosed(phase nvidiacomv1beta1.DynamoGraphPowerBudgetPhase) bool {
	return phase != nvidiacomv1beta1.DynamoGraphPowerBudgetPhaseIdle
}
