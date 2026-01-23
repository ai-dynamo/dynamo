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
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
)

// State represents the internal state of a rolling update.
// These are more granular than the API-level RolloutPhase.
type State string

const (
	StateIdle                   State = "Idle"
	StateCreatingNewFrontend    State = "CreatingNewFrontend"
	StateWaitingForFrontend     State = "WaitingForFrontend"
	StateCreatingNewWorkers     State = "CreatingNewWorkers"
	StateScalingWorkers         State = "ScalingWorkers"
	StateWaitingForMinimumReady State = "WaitingForMinimumReady"
	StateUpdatingProxyWeights   State = "UpdatingProxyWeights"
	StateCleaningUp             State = "CleaningUp"
	StateComplete               State = "Complete"
	StateFailed                 State = "Failed"
)

// ToAPIPhase maps internal state to the API-level RolloutPhase.
func (s State) ToAPIPhase() v1alpha1.RolloutPhase {
	switch s {
	case StateIdle:
		return v1alpha1.RolloutPhaseNone
	case StateComplete:
		return v1alpha1.RolloutPhaseCompleted
	case StateFailed:
		return v1alpha1.RolloutPhaseFailed
	default:
		return v1alpha1.RolloutPhaseInProgress
	}
}

// IsTerminal returns true if the state is a terminal state (Complete or Failed).
func (s State) IsTerminal() bool {
	return s == StateComplete || s == StateFailed
}

// IsActive returns true if a rollout is actively in progress.
func (s State) IsActive() bool {
	return s != StateIdle && s != StateComplete && s != StateFailed
}

// StateMachine manages the rollout lifecycle state transitions.
type StateMachine struct {
	CurrentState    State
	OldWorkerHash   string
	NewWorkerHash   string
	OldNamespace    string
	NewNamespace    string
	FailureReason   string
}

// NewStateMachine creates a new state machine in the Idle state.
func NewStateMachine() *StateMachine {
	return &StateMachine{
		CurrentState: StateIdle,
	}
}

// NewStateMachineFromRolloutStatus reconstructs a state machine from API status.
// This is used to resume a rollout after controller restart.
func NewStateMachineFromRolloutStatus(status *v1alpha1.RolloutStatus, internalState State) *StateMachine {
	if status == nil {
		return NewStateMachine()
	}

	sm := &StateMachine{
		CurrentState: internalState,
	}

	// Map API phase to internal state if no internal state provided
	if internalState == "" {
		switch status.Phase {
		case v1alpha1.RolloutPhaseNone:
			sm.CurrentState = StateIdle
		case v1alpha1.RolloutPhasePending:
			sm.CurrentState = StateCreatingNewFrontend
		case v1alpha1.RolloutPhaseInProgress:
			sm.CurrentState = StateScalingWorkers
		case v1alpha1.RolloutPhaseCompleted:
			sm.CurrentState = StateComplete
		case v1alpha1.RolloutPhaseFailed:
			sm.CurrentState = StateFailed
		}
	}

	return sm
}

// Start transitions from Idle to the first active state.
func (sm *StateMachine) Start(oldHash, newHash, oldNamespace, newNamespace string) {
	if sm.CurrentState != StateIdle {
		return
	}
	sm.OldWorkerHash = oldHash
	sm.NewWorkerHash = newHash
	sm.OldNamespace = oldNamespace
	sm.NewNamespace = newNamespace
	sm.CurrentState = StateCreatingNewFrontend
}

// TransitionTo moves to a new state. Returns false if the transition is invalid.
func (sm *StateMachine) TransitionTo(newState State) bool {
	if !sm.isValidTransition(newState) {
		return false
	}
	sm.CurrentState = newState
	return true
}

// Fail transitions to the Failed state with a reason.
func (sm *StateMachine) Fail(reason string) {
	sm.FailureReason = reason
	sm.CurrentState = StateFailed
}

// Complete transitions to the Complete state.
func (sm *StateMachine) Complete() {
	sm.CurrentState = StateComplete
}

// isValidTransition checks if a state transition is allowed.
func (sm *StateMachine) isValidTransition(newState State) bool {
	// Can always transition to Failed
	if newState == StateFailed {
		return true
	}

	// Cannot transition from terminal states
	if sm.CurrentState.IsTerminal() {
		return false
	}

	// Define valid transitions
	validTransitions := map[State][]State{
		StateIdle:                   {StateCreatingNewFrontend},
		StateCreatingNewFrontend:    {StateWaitingForFrontend, StateFailed},
		StateWaitingForFrontend:     {StateCreatingNewWorkers, StateFailed},
		StateCreatingNewWorkers:     {StateScalingWorkers, StateFailed},
		StateScalingWorkers:         {StateWaitingForMinimumReady, StateUpdatingProxyWeights, StateCleaningUp, StateFailed},
		StateWaitingForMinimumReady: {StateUpdatingProxyWeights, StateFailed},
		StateUpdatingProxyWeights:   {StateScalingWorkers, StateCleaningUp, StateFailed},
		StateCleaningUp:             {StateComplete, StateFailed},
	}

	allowed, exists := validTransitions[sm.CurrentState]
	if !exists {
		return false
	}

	for _, s := range allowed {
		if s == newState {
			return true
		}
	}
	return false
}
