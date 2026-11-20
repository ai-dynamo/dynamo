// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/client"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// Leave headroom below the usual API/etcd object limit. Existing large objects
// remain writable by omitting the optional export rather than changing workloads.
const maxTargetStatusObjectBytes = 1 << 20

func targetRoleStatusFits(object client.Object) bool {
	data, err := json.Marshal(object)
	return err == nil && len(data) <= maxTargetStatusObjectBytes
}

func (r *workloadProgramResult) boundTargetRoleStatus(dgd *v1beta1.DynamoGraphDeployment) {
	// Marshal a read-only view with the program's complete returned status.
	view := *dgd
	view.Status = r.Status
	if targetRoleStatusFits(&view) {
		return
	}

	// Retain all existing status when the new observational data has no headroom.
	for name, status := range r.Status.Components {
		status.Roles = nil
		r.Status.Components[name] = status
	}
}
