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

// Package dgdcreconcile computes the reconcile-diff for DynamoGraphDeploymentCandidate
// (DGDC) lifecycle: given the set of candidates the DGDR(v2) controller currently wants
// to exist (desired, read from the relayed sweeper_status.yaml ConfigMap blob) and the
// set it currently observes in the cluster (current, a real List call against the
// DGDC CRD), it returns which to create, delete, and status-update.
//
// This is a direct Go port of the Python prototype in
// dynamo.profiler.sweeper.dgdc_reconcile_diff (tracking issue #13545, item 5), kept
// deliberately pure and CRD-independent: it takes and returns plain Go types, not the
// v1beta2 DynamoGraphDeploymentCandidate type itself (which is still feature-gated and
// inactive -- see PRs #13603/#13744), so this package compiles, tests, and can be
// reviewed on its own before that CRD lands and the controller is wired to call it.
//
// Callers perform all I/O: fetching current state (a real List against the cluster)
// and applying the returned Actions (Create/Delete/Patch calls). Identity deliberately
// excludes Rank, which is mutable; only a status update is possible for an identity
// present in both desired and current.
package dgdcreconcile

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// DiffInputError means the desired or current set was malformed in a way that makes
// the diff impossible to compute safely (e.g. a duplicate identity). This is a
// TERMINAL condition -- the caller should persist a failure condition rather than
// silently skip or retry, since retrying malformed input won't fix it.
type DiffInputError struct {
	msg string
}

func (e *DiffInputError) Error() string { return e.msg }

func newDiffInputError(format string, args ...any) *DiffInputError {
	return &DiffInputError{msg: fmt.Sprintf(format, args...)}
}

// ComputeIdentity returns a stable, content-addressed identity for one candidate's
// spec+experimental context: the same spec and experimental content (regardless of Go
// map key order, which json.Marshal always serializes sorted) always produces the same
// identity, and any difference in either produces a different one.
func ComputeIdentity(spec, experimental map[string]any) (string, error) {
	if spec == nil {
		spec = map[string]any{}
	}
	if experimental == nil {
		experimental = map[string]any{}
	}
	// encoding/json marshals map[string]any keys in sorted order at every nesting
	// level, matching Python's json.dumps(..., sort_keys=True) byte-for-byte modulo
	// separator whitespace, which json.Marshal also omits by default -- so this
	// does not need to reproduce Python's separators=(",", ":") explicitly.
	canonical, err := json.Marshal(map[string]any{
		"experimental": experimental,
		"spec":         spec,
	})
	if err != nil {
		return "", fmt.Errorf("computing candidate identity: %w", err)
	}
	sum := sha256.Sum256(canonical)
	return hex.EncodeToString(sum[:])[:16], nil
}

// DesiredCandidate is one candidate the publisher (the DGDR(v2) controller, reading
// the relayed sweeper_status.yaml blob) has selected for materialization.
// Spec/Experimental are exactly the "manifest" content and any accompanying
// experimental context carried in the relayed CandidateStatusEntry.
type DesiredCandidate struct {
	Spec         map[string]any
	Experimental map[string]any
	// Rank is nil for a Pareto goal, where candidates are not ordered; for a scalar
	// goal it is the candidate's one-based rank.
	Rank *int32
}

// Identity recomputes this candidate's content-addressed identity on every call,
// mirroring the Python prototype's read-only @property -- callers are not expected to
// call it more than once per candidate per diff, so caching is not worth the
// complexity of invalidating it if Spec/Experimental are ever mutated in place.
func (d DesiredCandidate) Identity() (string, error) {
	return ComputeIdentity(d.Spec, d.Experimental)
}

// CurrentDGDC is one DGDC observed in the cluster right now. Identity is read back
// from wherever the create action originally recorded it (e.g. a label or annotation
// on the real object), not recomputed from a live Spec -- the controller never
// re-derives identity from cluster state, only from the desired side.
type CurrentDGDC struct {
	Name     string
	Identity string
	Rank     *int32
}

// StatusUpdate is one existing DGDC whose Rank needs to change to NewRank; nothing
// else about it changed (same identity), so this is a status-only patch, never a
// delete-and-recreate.
type StatusUpdate struct {
	Name    string
	NewRank *int32
}

// Actions is the full set of operations the caller must apply to reconcile current
// state to desired state. A nil/empty Desired set (no feasible candidate this round)
// is expected-state, not an error -- it produces a Delete for every CurrentDGDC.
// Whether "delete everything" is the right response to an empty round is a policy
// decision belonging to the caller, not this package.
type Actions struct {
	Creates       []DesiredCandidate
	Deletes       []string
	StatusUpdates []StatusUpdate
}

func ranksEqual(a, b *int32) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

// ComputeActions computes the reconcile-diff between desired and current.
//
// Duplicate identities within either desired or current return a *DiffInputError --
// a TERMINAL condition, since the diff cannot safely decide which of two
// identical-identity entries owns a given name (for current) or which Rank wins (for
// desired). This module detects and refuses duplicate identities in its inputs; it
// cannot prevent them from being created. Whatever applies Creates against a real
// cluster must guarantee identity uniqueness at creation time (e.g. via the identity
// recorded in a label, checked before create) -- this function's error is a safety
// net, not a substitute for that guarantee.
func ComputeActions(desired []DesiredCandidate, current []CurrentDGDC) (Actions, error) {
	desiredByIdentity := make(map[string]DesiredCandidate, len(desired))
	for _, candidate := range desired {
		identity, err := candidate.Identity()
		if err != nil {
			return Actions{}, err
		}
		if _, exists := desiredByIdentity[identity]; exists {
			return Actions{}, newDiffInputError("duplicate identity in desired set: %s", identity)
		}
		desiredByIdentity[identity] = candidate
	}

	currentByIdentity := make(map[string]CurrentDGDC, len(current))
	for _, existing := range current {
		if _, exists := currentByIdentity[existing.Identity]; exists {
			return Actions{}, newDiffInputError("duplicate identity in current set: %s", existing.Identity)
		}
		currentByIdentity[existing.Identity] = existing
	}

	// Iterate the original slices, not the maps -- Go randomizes map iteration
	// order on every run, which would make Creates/Deletes/StatusUpdates order
	// non-deterministic; the maps above are kept only for membership checks and
	// identity lookups, and iterating the input order here matches what the
	// Python prototype gets for free from its insertion-ordered dicts.
	var actions Actions
	for _, candidate := range desired {
		identity, err := candidate.Identity()
		if err != nil {
			return Actions{}, err
		}
		if _, exists := currentByIdentity[identity]; !exists {
			actions.Creates = append(actions.Creates, candidate)
		}
	}
	for _, existing := range current {
		if _, exists := desiredByIdentity[existing.Identity]; !exists {
			actions.Deletes = append(actions.Deletes, existing.Name)
		}
	}
	for _, existing := range current {
		candidate, exists := desiredByIdentity[existing.Identity]
		if !exists {
			continue
		}
		if !ranksEqual(existing.Rank, candidate.Rank) {
			actions.StatusUpdates = append(actions.StatusUpdates, StatusUpdate{
				Name:    existing.Name,
				NewRank: candidate.Rank,
			})
		}
	}

	return actions, nil
}
