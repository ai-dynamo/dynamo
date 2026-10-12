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

package dgdrrunpublisher

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestCandidateNameIsBoundedAndUniquePerRunIncarnation(t *testing.T) {
	id := "evaluated-point-0123456789ab"
	if cn("a") != CandidateName("run", testRunUID, "a") || cn("a") == CandidateName("run", "uid-2", "a") {
		t.Fatalf("a recreated run must get different names: %q", cn("a"))
	}
	long := strings.Repeat("r", 253)
	other := strings.Repeat("r", 252) + "s"
	name := CandidateName(long, testRunUID, id)
	if len(name) > 253 || !strings.HasSuffix(name, "-"+id) || name == CandidateName(other, testRunUID, id) || name != CandidateName(long, testRunUID, id) {
		t.Fatalf("name = %q (%d)", name, len(name))
	}
}

func desire(t *testing.T, data []byte) DesiredState {
	t.Helper()
	return Desire("run", testRunUID, mustParse(t, data))
}

func TestSwapKeepsCandidateNamesAndOnlyReordersReferences(t *testing.T) {
	first := desire(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "b"}))
	second := desire(t, snapshotJSON(t, PhaseRunning, cand{id: "b"}, cand{id: "a"}))
	if !reflect.DeepEqual(first.Names, second.Names) {
		t.Fatalf("names depend on rank: %v vs %v", first.Names, second.Names)
	}
	if want := []string{cn("b"), cn("a")}; !reflect.DeepEqual(second.Status.CandidateNames, want) {
		t.Fatalf("ordered references = %v, want %v", second.Status.CandidateNames, want)
	}
	if !second.Status.NeedsUpdate(first.Status) {
		t.Fatal("a reorder must update the run's references")
	}
}

func TestFailedMaterializationIsNotDesiredButCounted(t *testing.T) {
	state := desire(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}, cand{id: "x", failed: true}))
	if len(state.Candidates) != 1 || state.Candidates[0].ID != "a" {
		t.Fatalf("candidates = %+v", state.Candidates)
	}
	if !reflect.DeepEqual(state.Status.CandidateNames, []string{cn("a")}) {
		t.Fatalf("refs must not include the failed candidate: %v", state.Status.CandidateNames)
	}
	if !strings.Contains(state.Status.Message, "1 candidate(s) failed materialization") {
		t.Fatalf("message = %q", state.Status.Message)
	}
}

func TestRunErrorReplacesTheMessage(t *testing.T) {
	snap := mustParse(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}))
	snap.Run.Message, snap.Run.Error = "searching", "out of memory"
	if got := Desire("run", testRunUID, snap).Status.Message; got != "out of memory" {
		t.Fatalf("message = %q", got)
	}
}

func TestLastProgressComesFromTheSnapshotTimestamp(t *testing.T) {
	data := withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:12Z")
	want := time.Date(2026, 10, 5, 21, 35, 12, 0, time.UTC)
	if got := desire(t, data).Status.LastProgress; !got.Equal(want) {
		t.Fatalf("LastProgress = %v, want %v", got, want)
	}
}

func TestMissingOrMalformedTimestampLeavesProgressTimeAlone(t *testing.T) {
	for _, timestamp := range []string{"", "yesterday"} {
		state := desire(t, withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), timestamp))
		if !state.Status.LastProgress.IsZero() {
			t.Fatalf("timestamp %q: LastProgress = %v, want zero", timestamp, state.Status.LastProgress)
		}
		observed := state.Status
		observed.LastProgress = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
		if state.Status.NeedsUpdate(observed) {
			t.Fatalf("timestamp %q: an unknown progress time must not force an update", timestamp)
		}
	}
}

func TestNeedsUpdateOnlyWhenSomethingDiffers(t *testing.T) {
	base := desire(t, withTimestamp(t, snapshotJSON(t, PhaseRunning, cand{id: "a"}), "2026-10-05T21:35:12Z")).Status
	if base.NeedsUpdate(base) {
		t.Fatal("identical status needs no write")
	}
	changes := map[string]func(*RunStatus){
		"message":       func(s *RunStatus) { s.Message = "other" },
		"round":         func(s *RunStatus) { s.Round++ },
		"evaluated":     func(s *RunStatus) { s.Evaluated++ },
		"no progress":   func(s *RunStatus) { s.HasProgress = false },
		"progress time": func(s *RunStatus) { s.LastProgress = s.LastProgress.Add(time.Second) },
		"references":    func(s *RunStatus) { s.CandidateNames = nil },
	}
	for name, mutate := range changes {
		observed := base
		observed.CandidateNames = append([]string(nil), base.CandidateNames...)
		mutate(&observed)
		if !base.NeedsUpdate(observed) {
			t.Errorf("%s: a difference must trigger an update", name)
		}
	}
}
