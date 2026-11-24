// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// applyRetuneStatus publishes the semantic target binding independently of
// optional workload projections or Pod readiness. dgd must not be nil.
func (r *workloadProgramResult) applyRetuneStatus(dgd *v1beta1.DynamoGraphDeployment, targetsAcknowledged bool) {
	// Never retain a removed capability when the new target has been acknowledged.
	for name, status := range r.Status.Components {
		status.Retune = nil
		r.Status.Components[name] = status
	}
	if r.Status.ObservedGeneration != dgd.Generation || !targetsAcknowledged {
		return
	}
	plannerName, configuration, err := dynamo.RetuneConfiguration(dgd)
	if err != nil || configuration == nil {
		return
	}

	// The provider acknowledges persisted targets, not the running process revision.
	plannerStatus, present := r.Status.Components[plannerName]
	if !present {
		return
	}
	plannerStatus.Retune = &v1beta1.RetuneStatus{ConfigHash: configuration.ConfigHash}
	r.Status.Components[plannerName] = plannerStatus
}

// retuneDCDTargetsAcknowledged observes the existing child-lowering boundary.
// dgd and each child must be non-nil. Unrelated worker readiness is not a gate.
func retuneDCDTargetsAcknowledged(dgd *v1beta1.DynamoGraphDeployment, children []*v1beta1.DynamoComponentDeployment) bool {
	plannerName, configuration, err := dynamo.RetuneConfiguration(dgd)
	if err != nil || configuration == nil {
		return false
	}
	acknowledged := make(map[string]bool, 2)
	for _, child := range children {
		name := child.Spec.ComponentName
		if name == plannerName || name == configuration.FrontendComponentName {
			// A normal child observation supplies the required component identity,
			// even when no replicas are ready and no workload projection is exported.
			acknowledged[name] = child.DeletionTimestamp.IsZero() && child.Status.ObservedGeneration == child.Generation && child.Status.Component != nil
		}
	}
	return acknowledged[plannerName] && acknowledged[configuration.FrontendComponentName]
}
