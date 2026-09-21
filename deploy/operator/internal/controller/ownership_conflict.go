/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"errors"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ownershipConflictTransition uint8

const (
	ownershipConflictNoTransition ownershipConflictTransition = iota
	ownershipConflictRaised
	ownershipConflictOngoing
	ownershipConflictResolved
)

// applyOwnershipConflict derives the condition update and transition for a
// reconciliation result. Callers own applying and persisting the condition,
// because their status and event lifecycles differ.
func applyOwnershipConflict(
	conditions []metav1.Condition,
	generation int64,
	reconcileErr error,
) (*metav1.Condition, ownershipConflictTransition) {
	previous := meta.FindStatusCondition(conditions, nvidiacomv1beta1.ConditionTypeOwnershipConflict)

	var ownershipConflict *commoncontroller.OwnershipConflictError
	if errors.As(reconcileErr, &ownershipConflict) {
		condition := &metav1.Condition{
			Type:               nvidiacomv1beta1.ConditionTypeOwnershipConflict,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: generation,
			Reason:             commoncontroller.EventReasonOwnershipConflict,
			Message:            ownershipConflict.Error(),
		}
		if previous == nil || previous.Status != metav1.ConditionTrue {
			return condition, ownershipConflictRaised
		}
		return condition, ownershipConflictOngoing
	}

	if reconcileErr == nil && previous != nil && previous.Status == metav1.ConditionTrue {
		return &metav1.Condition{
			Type:               nvidiacomv1beta1.ConditionTypeOwnershipConflict,
			Status:             metav1.ConditionFalse,
			ObservedGeneration: generation,
			Reason:             "OwnershipConflictResolved",
			Message:            "No resource ownership conflicts observed.",
		}, ownershipConflictResolved
	}

	return nil, ownershipConflictNoTransition
}
