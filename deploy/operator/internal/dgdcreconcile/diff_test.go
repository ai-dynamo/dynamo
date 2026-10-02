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

package dgdcreconcile

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func int32Ptr(v int32) *int32 { return &v }

var (
	specA = map[string]any{
		"components":       []any{map[string]any{"name": "worker", "replicas": 2}},
		"backendFramework": "trtllm",
	}
	specB = map[string]any{
		"components":       []any{map[string]any{"name": "worker", "replicas": 4}},
		"backendFramework": "trtllm",
	}
)

func mustIdentity(t *testing.T, spec, experimental map[string]any) string {
	t.Helper()
	identity, err := ComputeIdentity(spec, experimental)
	if err != nil {
		t.Fatalf("ComputeIdentity: %v", err)
	}
	return identity
}

func TestIdentityIsStableRegardlessOfKeyOrder(t *testing.T) {
	specA := map[string]any{
		"components":       []any{map[string]any{"name": "worker"}},
		"backendFramework": "trtllm",
	}
	specB := map[string]any{
		"backendFramework": "trtllm",
		"components":       []any{map[string]any{"name": "worker"}},
	}
	idA := mustIdentity(t, specA, nil)
	idB := mustIdentity(t, specB, nil)
	if idA != idB {
		t.Errorf("expected identical identity for the same content in different map insertion order, got %q vs %q", idA, idB)
	}
}

func TestIdentityChangesWhenExperimentalContextDiffers(t *testing.T) {
	idA := mustIdentity(t, specA, map[string]any{"kv_load_ratio": 0.25})
	idB := mustIdentity(t, specA, map[string]any{"kv_load_ratio": 1.0})
	if idA == idB {
		t.Errorf("expected different identities for different experimental context, both got %q", idA)
	}
}

func TestNewCandidateProducesACreateAndNothingElse(t *testing.T) {
	desired := []DesiredCandidate{{Spec: specA, Rank: int32Ptr(1)}}
	actions, err := ComputeActions(desired, nil)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Creates) != 1 {
		t.Fatalf("expected exactly 1 create, got %d", len(actions.Creates))
	}
	if fmt.Sprint(actions.Creates[0].Spec) != fmt.Sprint(specA) {
		t.Errorf("expected the create to carry specA through unchanged, got %v", actions.Creates[0].Spec)
	}
	if len(actions.Deletes) != 0 || len(actions.StatusUpdates) != 0 {
		t.Errorf("expected no deletes/status updates, got %+v", actions)
	}
}

func TestCandidateNoLongerSelectedProducesADelete(t *testing.T) {
	identity := mustIdentity(t, specA, nil)
	current := []CurrentDGDC{{Name: "cand-000", Identity: identity, Rank: int32Ptr(1)}}

	actions, err := ComputeActions(nil, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Deletes) != 1 || actions.Deletes[0] != "cand-000" {
		t.Errorf("expected Deletes == [cand-000], got %v", actions.Deletes)
	}
	if len(actions.Creates) != 0 || len(actions.StatusUpdates) != 0 {
		t.Errorf("expected no creates/status updates, got %+v", actions)
	}
}

func TestRankOnlyChangeProducesAStatusUpdateNotDeleteAndRecreate(t *testing.T) {
	identity := mustIdentity(t, specA, nil)
	current := []CurrentDGDC{{Name: "cand-000", Identity: identity, Rank: int32Ptr(2)}}
	desired := []DesiredCandidate{{Spec: specA, Rank: int32Ptr(1)}}

	actions, err := ComputeActions(desired, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Creates) != 0 || len(actions.Deletes) != 0 {
		t.Errorf("expected no creates/deletes, got %+v", actions)
	}
	if len(actions.StatusUpdates) != 1 || actions.StatusUpdates[0].Name != "cand-000" || *actions.StatusUpdates[0].NewRank != 1 {
		t.Errorf("expected a single status update to rank 1 for cand-000, got %+v", actions.StatusUpdates)
	}
}

func TestUnchangedCandidateProducesNoActionsAtAll(t *testing.T) {
	identity := mustIdentity(t, specA, nil)
	current := []CurrentDGDC{{Name: "cand-000", Identity: identity, Rank: int32Ptr(1)}}
	desired := []DesiredCandidate{{Spec: specA, Rank: int32Ptr(1)}}

	actions, err := ComputeActions(desired, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Creates) != 0 || len(actions.Deletes) != 0 || len(actions.StatusUpdates) != 0 {
		t.Errorf("expected zero actions for an unchanged candidate, got %+v", actions)
	}
}

func TestChangedContentAtTheSameRankIsDeleteAndCreateNotUpdate(t *testing.T) {
	oldIdentity := mustIdentity(t, specA, nil)
	current := []CurrentDGDC{{Name: "cand-000", Identity: oldIdentity, Rank: int32Ptr(1)}}
	desired := []DesiredCandidate{{Spec: specB, Rank: int32Ptr(1)}}

	actions, err := ComputeActions(desired, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Deletes) != 1 || actions.Deletes[0] != "cand-000" {
		t.Errorf("expected Deletes == [cand-000], got %v", actions.Deletes)
	}
	if len(actions.Creates) != 1 || fmt.Sprint(actions.Creates[0].Spec) != fmt.Sprint(specB) {
		t.Errorf("expected a single create carrying specB, got %+v", actions.Creates)
	}
	if len(actions.StatusUpdates) != 0 {
		t.Errorf("expected no status updates when content actually changed, got %+v", actions.StatusUpdates)
	}
}

func TestDuplicateIdentityInDesiredSetRaisesTerminalError(t *testing.T) {
	desired := []DesiredCandidate{
		{Spec: specA, Rank: int32Ptr(1)},
		{Spec: specA, Rank: int32Ptr(2)},
	}
	_, err := ComputeActions(desired, nil)
	var diffErr *DiffInputError
	if !errors.As(err, &diffErr) {
		t.Fatalf("expected a *DiffInputError, got %v", err)
	}
	if want := "duplicate identity in desired"; !strings.Contains(err.Error(), want) {
		t.Errorf("expected error to mention %q, got %q", want, err.Error())
	}
}

func TestDuplicateIdentityInCurrentWithEmptyDesiredRaisesNotSilentlyDrops(t *testing.T) {
	identity := mustIdentity(t, specA, nil)
	current := []CurrentDGDC{
		{Name: "cand-000", Identity: identity, Rank: int32Ptr(1)},
		{Name: "cand-001", Identity: identity, Rank: int32Ptr(1)},
	}
	_, err := ComputeActions(nil, current)
	var diffErr *DiffInputError
	if !errors.As(err, &diffErr) {
		t.Fatalf("expected a *DiffInputError, got %v", err)
	}
	if want := "duplicate identity in current"; !strings.Contains(err.Error(), want) {
		t.Errorf("expected error to mention %q, got %q", want, err.Error())
	}
}

// TestRankReorderingAcrossMultipleEntriesProducesOnlyStatusUpdates is generic
// multi-entry rank-change coverage for the diff algorithm itself, which is agnostic to
// what rank means -- it only detects whether the value differs. NOT framed as a
// "Pareto" scenario: a DGDC's real Rank field is the one-based scalar ordering and is
// absent for Pareto searches -- real Pareto candidates never carry a non-nil,
// reshuffling rank like this at all. See
// TestParetoCandidatesNeverProduceRankStatusUpdates below for what realistic Pareto
// data actually looks like.
func TestRankReorderingAcrossMultipleEntriesProducesOnlyStatusUpdates(t *testing.T) {
	replicaCounts := []int{2, 4, 8}
	identities := make([]string, len(replicaCounts))
	for i, n := range replicaCounts {
		identities[i] = mustIdentity(t, map[string]any{"components": []any{map[string]any{"replicas": n}}}, nil)
	}

	current := make([]CurrentDGDC, len(identities))
	for i, identity := range identities {
		current[i] = CurrentDGDC{Name: fmt.Sprintf("cand-%03d", i), Identity: identity, Rank: int32Ptr(int32(i + 1))}
	}
	desired := []DesiredCandidate{
		{Spec: map[string]any{"components": []any{map[string]any{"replicas": 8}}}, Rank: int32Ptr(1)},
		{Spec: map[string]any{"components": []any{map[string]any{"replicas": 2}}}, Rank: int32Ptr(2)},
		{Spec: map[string]any{"components": []any{map[string]any{"replicas": 4}}}, Rank: int32Ptr(3)},
	}

	actions, err := ComputeActions(desired, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Creates) != 0 || len(actions.Deletes) != 0 {
		t.Errorf("expected no creates/deletes, got %+v", actions)
	}

	got := make(map[string]int32, len(actions.StatusUpdates))
	for _, u := range actions.StatusUpdates {
		got[u.Name] = *u.NewRank
	}
	want := map[string]int32{"cand-000": 2, "cand-001": 3, "cand-002": 1}
	if len(got) != len(want) {
		t.Fatalf("expected %d status updates, got %d: %+v", len(want), len(got), actions.StatusUpdates)
	}
	for name, rank := range want {
		if got[name] != rank {
			t.Errorf("expected %s -> rank %d, got %d", name, rank, got[name])
		}
	}
}

func TestParetoCandidatesNeverProduceRankStatusUpdates(t *testing.T) {
	// Pareto candidates never carry a rank, so an unchanged front produces no
	// status updates.
	replicaCounts := []int{2, 4, 8}
	identities := make([]string, len(replicaCounts))
	for i, n := range replicaCounts {
		identities[i] = mustIdentity(t, map[string]any{"components": []any{map[string]any{"replicas": n}}}, nil)
	}

	current := make([]CurrentDGDC, len(identities))
	for i, identity := range identities {
		current[i] = CurrentDGDC{Name: fmt.Sprintf("cand-%03d", i), Identity: identity, Rank: nil}
	}
	desired := make([]DesiredCandidate, len(replicaCounts))
	for i, n := range replicaCounts {
		desired[i] = DesiredCandidate{Spec: map[string]any{"components": []any{map[string]any{"replicas": n}}}, Rank: nil}
	}

	actions, err := ComputeActions(desired, current)
	if err != nil {
		t.Fatalf("ComputeActions: %v", err)
	}
	if len(actions.Creates) != 0 || len(actions.Deletes) != 0 || len(actions.StatusUpdates) != 0 {
		t.Errorf("expected zero actions for an unchanged Pareto front, got %+v", actions)
	}
}

// TestOutputOrderMatchesInputOrder covers the Go-specific risk this port introduces
// that the Python original did not have: Go randomizes map iteration order, so
// Creates/Deletes/StatusUpdates must be built by iterating the input slices, not the
// identity maps, or their order (and therefore any test asserting on index position)
// would be flaky.
func TestOutputOrderMatchesInputOrder(t *testing.T) {
	desired := make([]DesiredCandidate, 20)
	for i := range desired {
		rank := int32(i + 1)
		desired[i] = DesiredCandidate{
			Spec: map[string]any{"components": []any{map[string]any{"replicas": i}}},
			Rank: &rank,
		}
	}
	for attempt := 0; attempt < 20; attempt++ {
		actions, err := ComputeActions(desired, nil)
		if err != nil {
			t.Fatalf("ComputeActions: %v", err)
		}
		if len(actions.Creates) != len(desired) {
			t.Fatalf("expected %d creates, got %d", len(desired), len(actions.Creates))
		}
		for i, create := range actions.Creates {
			if fmt.Sprint(create.Spec) != fmt.Sprint(desired[i].Spec) {
				t.Fatalf("attempt %d: expected Creates[%d] to match desired[%d] (input order), got a different candidate -- map iteration order leaked into the output", attempt, i, i)
			}
		}
	}
}
