/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package rollout

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

func TestStateToAPIPhase(t *testing.T) {
	tests := []struct {
		name     string
		state    State
		expected v1alpha1.RolloutPhase
	}{
		{
			name:     "idle maps to none",
			state:    StateIdle,
			expected: v1alpha1.RolloutPhaseNone,
		},
		{
			name:     "complete maps to completed",
			state:    StateComplete,
			expected: v1alpha1.RolloutPhaseCompleted,
		},
		{
			name:     "failed maps to failed",
			state:    StateFailed,
			expected: v1alpha1.RolloutPhaseFailed,
		},
		{
			name:     "creating frontend maps to in progress",
			state:    StateCreatingNewFrontend,
			expected: v1alpha1.RolloutPhaseInProgress,
		},
		{
			name:     "scaling workers maps to in progress",
			state:    StateScalingWorkers,
			expected: v1alpha1.RolloutPhaseInProgress,
		},
		{
			name:     "cleaning up maps to in progress",
			state:    StateCleaningUp,
			expected: v1alpha1.RolloutPhaseInProgress,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.state.ToAPIPhase()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestStateIsTerminal(t *testing.T) {
	tests := []struct {
		name     string
		state    State
		expected bool
	}{
		{
			name:     "idle is not terminal",
			state:    StateIdle,
			expected: false,
		},
		{
			name:     "complete is terminal",
			state:    StateComplete,
			expected: true,
		},
		{
			name:     "failed is terminal",
			state:    StateFailed,
			expected: true,
		},
		{
			name:     "scaling workers is not terminal",
			state:    StateScalingWorkers,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.state.IsTerminal()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestStateIsActive(t *testing.T) {
	tests := []struct {
		name     string
		state    State
		expected bool
	}{
		{
			name:     "idle is not active",
			state:    StateIdle,
			expected: false,
		},
		{
			name:     "complete is not active",
			state:    StateComplete,
			expected: false,
		},
		{
			name:     "failed is not active",
			state:    StateFailed,
			expected: false,
		},
		{
			name:     "creating frontend is active",
			state:    StateCreatingNewFrontend,
			expected: true,
		},
		{
			name:     "scaling workers is active",
			state:    StateScalingWorkers,
			expected: true,
		},
		{
			name:     "cleaning up is active",
			state:    StateCleaningUp,
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.state.IsActive()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestNewStateMachine(t *testing.T) {
	sm := NewStateMachine()
	assert.Equal(t, StateIdle, sm.CurrentState)
	assert.Empty(t, sm.OldWorkerHash)
	assert.Empty(t, sm.NewWorkerHash)
	assert.Empty(t, sm.FailureReason)
}

func TestStateMachineStart(t *testing.T) {
	tests := []struct {
		name          string
		initialState  State
		expectedState State
		shouldStart   bool
	}{
		{
			name:          "start from idle",
			initialState:  StateIdle,
			expectedState: StateCreatingNewFrontend,
			shouldStart:   true,
		},
		{
			name:          "cannot start from active state",
			initialState:  StateScalingWorkers,
			expectedState: StateScalingWorkers,
			shouldStart:   false,
		},
		{
			name:          "cannot start from complete",
			initialState:  StateComplete,
			expectedState: StateComplete,
			shouldStart:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := &StateMachine{CurrentState: tt.initialState}
			sm.Start("old-hash", "new-hash", "old-ns", "new-ns")

			assert.Equal(t, tt.expectedState, sm.CurrentState)
			if tt.shouldStart {
				assert.Equal(t, "old-hash", sm.OldWorkerHash)
				assert.Equal(t, "new-hash", sm.NewWorkerHash)
				assert.Equal(t, "old-ns", sm.OldNamespace)
				assert.Equal(t, "new-ns", sm.NewNamespace)
			}
		})
	}
}

func TestStateMachineTransitionTo(t *testing.T) {
	tests := []struct {
		name          string
		initialState  State
		targetState   State
		expectedOK    bool
		expectedState State
	}{
		{
			name:          "valid transition: idle to creating frontend",
			initialState:  StateIdle,
			targetState:   StateCreatingNewFrontend,
			expectedOK:    true,
			expectedState: StateCreatingNewFrontend,
		},
		{
			name:          "valid transition: creating frontend to waiting for frontend",
			initialState:  StateCreatingNewFrontend,
			targetState:   StateWaitingForFrontend,
			expectedOK:    true,
			expectedState: StateWaitingForFrontend,
		},
		{
			name:          "valid transition: scaling workers to cleaning up",
			initialState:  StateScalingWorkers,
			targetState:   StateCleaningUp,
			expectedOK:    true,
			expectedState: StateCleaningUp,
		},
		{
			name:          "valid transition: cleaning up to complete",
			initialState:  StateCleaningUp,
			targetState:   StateComplete,
			expectedOK:    true,
			expectedState: StateComplete,
		},
		{
			name:          "can always transition to failed",
			initialState:  StateScalingWorkers,
			targetState:   StateFailed,
			expectedOK:    true,
			expectedState: StateFailed,
		},
		{
			name:          "invalid transition: cannot skip states",
			initialState:  StateIdle,
			targetState:   StateScalingWorkers,
			expectedOK:    false,
			expectedState: StateIdle,
		},
		{
			name:          "invalid transition: cannot go backwards",
			initialState:  StateCleaningUp,
			targetState:   StateScalingWorkers,
			expectedOK:    false,
			expectedState: StateCleaningUp,
		},
		{
			name:          "cannot transition from complete",
			initialState:  StateComplete,
			targetState:   StateIdle,
			expectedOK:    false,
			expectedState: StateComplete,
		},
		{
			name:          "cannot transition from failed (except to failed)",
			initialState:  StateFailed,
			targetState:   StateIdle,
			expectedOK:    false,
			expectedState: StateFailed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := &StateMachine{CurrentState: tt.initialState}
			ok := sm.TransitionTo(tt.targetState)

			assert.Equal(t, tt.expectedOK, ok)
			assert.Equal(t, tt.expectedState, sm.CurrentState)
		})
	}
}

func TestStateMachineFail(t *testing.T) {
	sm := NewStateMachine()
	sm.Start("old", "new", "old-ns", "new-ns")

	sm.Fail("test failure reason")

	assert.Equal(t, StateFailed, sm.CurrentState)
	assert.Equal(t, "test failure reason", sm.FailureReason)
}

func TestStateMachineComplete(t *testing.T) {
	sm := &StateMachine{CurrentState: StateCleaningUp}
	sm.Complete()

	assert.Equal(t, StateComplete, sm.CurrentState)
}

func TestNewStateMachineFromRolloutStatus(t *testing.T) {
	tests := []struct {
		name          string
		status        *v1alpha1.RolloutStatus
		internalState State
		expectedState State
	}{
		{
			name:          "nil status returns idle",
			status:        nil,
			internalState: "",
			expectedState: StateIdle,
		},
		{
			name:          "explicit internal state is used",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhaseInProgress},
			internalState: StateUpdatingProxyWeights,
			expectedState: StateUpdatingProxyWeights,
		},
		{
			name:          "phase none maps to idle",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhaseNone},
			internalState: "",
			expectedState: StateIdle,
		},
		{
			name:          "phase pending maps to creating frontend",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhasePending},
			internalState: "",
			expectedState: StateCreatingNewFrontend,
		},
		{
			name:          "phase in progress maps to scaling workers",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhaseInProgress},
			internalState: "",
			expectedState: StateScalingWorkers,
		},
		{
			name:          "phase completed maps to complete",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhaseCompleted},
			internalState: "",
			expectedState: StateComplete,
		},
		{
			name:          "phase failed maps to failed",
			status:        &v1alpha1.RolloutStatus{Phase: v1alpha1.RolloutPhaseFailed},
			internalState: "",
			expectedState: StateFailed,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sm := NewStateMachineFromRolloutStatus(tt.status, tt.internalState)
			assert.Equal(t, tt.expectedState, sm.CurrentState)
		})
	}
}
