/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"errors"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestApplyOwnershipConflict(t *testing.T) {
	ownershipConflict := &commoncontroller.OwnershipConflictError{Cause: errors.New("resource is controlled by another parent")}
	tests := []struct {
		name           string
		existingStatus *metav1.ConditionStatus
		reconcileErr   error
		wantCondition  bool
		wantStatus     metav1.ConditionStatus
		wantReason     string
		wantTransition ownershipConflictTransition
	}{
		{
			name:           "absent condition transitions to conflict",
			reconcileErr:   ownershipConflict,
			wantCondition:  true,
			wantStatus:     metav1.ConditionTrue,
			wantReason:     commoncontroller.EventReasonOwnershipConflict,
			wantTransition: ownershipConflictRaised,
		},
		{
			name:           "false condition transitions to conflict",
			existingStatus: ptr.To(metav1.ConditionFalse),
			reconcileErr:   ownershipConflict,
			wantCondition:  true,
			wantStatus:     metav1.ConditionTrue,
			wantReason:     commoncontroller.EventReasonOwnershipConflict,
			wantTransition: ownershipConflictRaised,
		},
		{
			name:           "active condition remains conflicted",
			existingStatus: ptr.To(metav1.ConditionTrue),
			reconcileErr:   ownershipConflict,
			wantCondition:  true,
			wantStatus:     metav1.ConditionTrue,
			wantReason:     commoncontroller.EventReasonOwnershipConflict,
			wantTransition: ownershipConflictOngoing,
		},
		{
			name:           "successful reconciliation clears active condition",
			existingStatus: ptr.To(metav1.ConditionTrue),
			wantCondition:  true,
			wantStatus:     metav1.ConditionFalse,
			wantReason:     "OwnershipConflictResolved",
			wantTransition: ownershipConflictResolved,
		},
		{
			name:           "another error preserves active condition",
			existingStatus: ptr.To(metav1.ConditionTrue),
			reconcileErr:   errors.New("unrelated failure"),
			wantStatus:     metav1.ConditionTrue,
			wantReason:     "PreviousConflict",
			wantTransition: ownershipConflictNoTransition,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var conditions []metav1.Condition
			if tt.existingStatus != nil {
				meta.SetStatusCondition(&conditions, metav1.Condition{
					Type:    nvidiacomv1beta1.ConditionTypeOwnershipConflict,
					Status:  *tt.existingStatus,
					Reason:  "PreviousConflict",
					Message: "Previously observed conflict",
				})
			}

			condition, transition := applyOwnershipConflict(conditions, 7, tt.reconcileErr)
			assert.Equal(t, tt.wantTransition, transition)
			if tt.wantCondition {
				require.NotNil(t, condition)
				meta.SetStatusCondition(&conditions, *condition)
			} else {
				require.Nil(t, condition)
			}

			stored := meta.FindStatusCondition(conditions, nvidiacomv1beta1.ConditionTypeOwnershipConflict)
			require.NotNil(t, stored)
			assert.Equal(t, tt.wantStatus, stored.Status)
			assert.Equal(t, tt.wantReason, stored.Reason)
			if tt.wantCondition {
				assert.Equal(t, int64(7), stored.ObservedGeneration)
			}
		})
	}
}
