/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Package dgdcreconcile computes the create/delete diff between the candidates a run
// snapshot wants to exist and the DynamoGraphDeploymentCandidates (DGDCs) observed in
// the cluster.
//
// Identity is the stable candidate id carried by the snapshot, which depends only on
// the evaluated point -- never on rank or list position. A DGDC is immutable once its
// initial status is populated, so an identity present on both sides needs no action:
// there is deliberately no "update" action. Reordering candidates is expressed only
// through the ordered candidate references on the run status, which this package does
// not touch.
//
// The package is pure (no Kubernetes types, no I/O) so the diff can be tested on its
// own; callers fetch current state and apply the returned Actions.
package dgdcreconcile

import "fmt"

// DiffInputError means the desired or current set was malformed (e.g. a duplicate or
// empty identity). Retrying will not fix it, so callers treat it as terminal.
type DiffInputError struct {
	msg string
}

func (e *DiffInputError) Error() string { return e.msg }

func newDiffInputError(format string, args ...any) *DiffInputError {
	return &DiffInputError{msg: fmt.Sprintf(format, args...)}
}

// DesiredCandidate is one candidate the snapshot wants to exist.
type DesiredCandidate struct {
	// ID is the stable candidate identity from the snapshot.
	ID string
	// Spec is the DGD manifest (YAML) to materialize.
	Spec string
	// Parameters and Metrics are the immutable evaluation facts recorded in the
	// DGDC status at creation.
	Parameters map[string]any
	Metrics    map[string]any
}

// CurrentDGDC is one DGDC observed in the cluster. ID is read back from the label
// recorded at creation, never recomputed.
type CurrentDGDC struct {
	Name string
	ID   string
}

// Actions is what the caller must apply. There are no updates: DGDCs are immutable.
type Actions struct {
	Creates []DesiredCandidate
	Deletes []string // DGDC names
}

// ComputeActions diffs desired against current. Output order follows the input order
// so results are deterministic.
func ComputeActions(desired []DesiredCandidate, current []CurrentDGDC) (Actions, error) {
	desiredIDs := make(map[string]struct{}, len(desired))
	for _, candidate := range desired {
		if candidate.ID == "" {
			return Actions{}, newDiffInputError("desired candidate has an empty id")
		}
		if _, dup := desiredIDs[candidate.ID]; dup {
			return Actions{}, newDiffInputError("duplicate id in desired set: %s", candidate.ID)
		}
		desiredIDs[candidate.ID] = struct{}{}
	}
	currentIDs := make(map[string]struct{}, len(current))
	for _, existing := range current {
		if existing.ID == "" {
			return Actions{}, newDiffInputError("DGDC %s has no candidate id", existing.Name)
		}
		if _, dup := currentIDs[existing.ID]; dup {
			return Actions{}, newDiffInputError("duplicate id in current set: %s", existing.ID)
		}
		currentIDs[existing.ID] = struct{}{}
	}

	var actions Actions
	for _, candidate := range desired {
		if _, exists := currentIDs[candidate.ID]; !exists {
			actions.Creates = append(actions.Creates, candidate)
		}
	}
	for _, existing := range current {
		if _, wanted := desiredIDs[existing.ID]; !wanted {
			actions.Deletes = append(actions.Deletes, existing.Name)
		}
	}
	return actions, nil
}
