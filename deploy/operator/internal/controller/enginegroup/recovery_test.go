//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"errors"
	"fmt"
	"testing"
	"time"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupErrorScheduling(t *testing.T) {
	transient := errors.New("connection reset")
	for _, test := range []struct {
		name      string
		err       error
		interval  time.Duration
		wantError bool
	}{
		{name: "unexpected transport failure", err: transient, wantError: true},
		{name: "known pending observation", err: fmt.Errorf("cache lag: %w", domain.ErrObservationPending), interval: engineGroupRequeueAfter},
		{name: "confirmed recovery fault", err: fmt.Errorf("native instance changed: %w", domain.ErrRecoveryRequired), interval: engineGroupObservationInterval},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("classify a wrapped failure without mixing controller-runtime error backoff and RequeueAfter")
			result, err := engineGroupErrorResult(test.err)
			assert.Equal(t, test.interval, result.RequeueAfter)
			if test.wantError {
				require.ErrorIs(t, err, transient)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestRecoveryDiagnosisPreservesFreshServingEvidence(t *testing.T) {
	t.Log("record a proven serving world before an invocation discovers a separate recovery fault")
	group := &api.DynamoGraphDeploymentEngineGroup{Generation: 1}
	setEngineGroupCondition(group, engineGroupConditionAvailable, metav1.ConditionTrue, "ServingAvailable", "Fresh serving proof")
	projectEngineGroupRecovery(group, fmt.Errorf("capacity apply: %w", domain.ErrRecoveryRequired))

	t.Log("explain the required action without overwriting independently fresh availability")
	assert.True(t, meta.IsStatusConditionTrue(group.Status.Conditions, engineGroupConditionAvailable))
	condition := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionRuntimeReady)
	require.NotNil(t, condition)
	assert.Equal(t, "RecoveryRequired", condition.Reason)
	assert.Contains(t, condition.Message, "explicit recovery or a fenced world restart")
}
