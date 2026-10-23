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
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// withRun sets the producer's message and error on a snapshot's run block.
func withRun(t *testing.T, data []byte, message, errText string) []byte {
	t.Helper()
	doc := map[string]any{}
	if err := json.Unmarshal(data, &doc); err != nil {
		t.Fatal(err)
	}
	run, ok := doc["run"].(map[string]any)
	if !ok {
		t.Fatalf("snapshot has no run block: %s", data)
	}
	run["message"], run["error"] = message, errText
	out, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestTerminalResult(t *testing.T) {
	materialized := []dgdcreconcile.DesiredCandidate{{ID: "a"}}
	succeeded := DesiredState{Candidates: materialized, Terminal: true, Outcome: TerminalOutcome{Phase: PhaseSucceeded}}
	if err := terminalResult(succeeded); err != nil {
		t.Fatalf("succeeded: %v", err)
	}

	failed := DesiredState{Candidates: materialized, Terminal: true, Outcome: TerminalOutcome{Phase: PhaseFailed, Message: "m", Error: "e"}}
	err := terminalResult(failed)
	var failure *RunFailedError
	if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || failure.Outcome != failed.Outcome {
		t.Fatalf("failed: %v", err)
	}
	if got, want := err.Error(), ErrRunFailed.Error()+": m: e"; got != want {
		t.Fatalf("error text = %q, want %q", got, want)
	}

	empty := DesiredState{Terminal: true, Outcome: TerminalOutcome{Phase: PhaseSucceeded}}
	if err := terminalResult(empty); !errors.Is(err, ErrRunFailed) {
		t.Fatalf("success without a candidate must fail the run: %v", err)
	}
}

func TestDesireCarriesBoundedDiagnostics(t *testing.T) {
	long := strings.Repeat("e", 3*maxDiagnosticBytes)
	state := desire(t, withRun(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}), "no recommendation\ncould be materialized", long))
	if state.Outcome.Phase != PhaseFailed || state.Outcome.Message != "no recommendation could be materialized" {
		t.Fatalf("outcome = %+v", state.Outcome)
	}
	if len(state.Outcome.Error) > maxDiagnosticBytes || !strings.HasPrefix(state.Outcome.Error, "eee") {
		t.Fatalf("error must be kept but bounded, got %d bytes", len(state.Outcome.Error))
	}
}

func TestBoundDiagnostic(t *testing.T) {
	if got := boundDiagnostic("  a\nb\x00c \t"); got != "a b c" {
		t.Fatalf("got %q", got)
	}
	multibyte := strings.Repeat("é", maxDiagnosticBytes) // two bytes each, so the cut must not split one
	got := boundDiagnostic(multibyte)
	if len(got) > maxDiagnosticBytes || !utf8.ValidString(got) || !strings.HasSuffix(got, "…") {
		t.Fatalf("got %d bytes, valid=%t", len(got), utf8.ValidString(got))
	}
	if got := boundDiagnostic("bad\xffbyte"); !utf8.ValidString(got) {
		t.Fatalf("invalid UTF-8 must be replaced: %q", got)
	}
}

func TestTerminalSucceededWithoutAMaterializedCandidateIsARunFailure(t *testing.T) {
	cases := map[string][]cand{
		"only failed materializations": {{id: "x", failed: true}},
		"no candidates at all":         nil,
	}
	for name, candidates := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFixture(t)
			f.publish(snapshotJSON(t, PhaseSucceeded, candidates...))
			f.mustReconcile()
			for _, op := range f.takeOps() {
				if strings.HasPrefix(op, "create") {
					t.Fatalf("nothing was materialized, nothing may be created: %s", op)
				}
			}
			err := f.r.Completion.Err()
			var failure *RunFailedError
			if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || ExitCode(err) != ExitRunFailed {
				t.Fatalf("err = %v, want the run-failed outcome instead of exit 0", err)
			}
			if len(candidates) > 0 && !strings.HasPrefix(failure.Outcome.Error, "x: ") {
				t.Fatalf("the candidate failure must be part of the diagnostics: %+v", failure.Outcome)
			}
		})
	}
}

func TestTerminalFailureDiagnosticsSurviveTheCompletionHandoff(t *testing.T) {
	f := newFixture(t)
	message := "no recommendation could be materialized"
	long := "backend vllm rejected the generated worker configuration " + strings.Repeat("x", 3*maxDiagnosticBytes)
	f.publish(withRun(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}), message, long))
	f.mustReconcile()

	err := f.r.Completion.Err()
	var failure *RunFailedError
	if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || ExitCode(err) != ExitRunFailed {
		t.Fatalf("err = %v", err)
	}
	if failure.Outcome.Message != message || !strings.HasPrefix(failure.Outcome.Error, "backend vllm rejected") || len(failure.Outcome.Error) > maxDiagnosticBytes {
		t.Fatalf("outcome = %+v", failure.Outcome)
	}

	// The handoff the run controller reads: the publisher container's termination message.
	path := filepath.Join(t.TempDir(), "termination-log")
	if err := WriteTerminationMessage(path, err); err != nil {
		t.Fatal(err)
	}
	data, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if len(data) > 4096 {
		t.Fatalf("termination message is %d bytes; the kubelet keeps 4096", len(data))
	}
	var got TerminalOutcome
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got != failure.Outcome {
		t.Fatalf("termination message = %+v, want %+v", got, failure.Outcome)
	}
}

func TestTerminationMessageForOtherFailuresAndSuccess(t *testing.T) {
	dir := t.TempDir()
	untouched := filepath.Join(dir, "success")
	if err := WriteTerminationMessage(untouched, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(untouched); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("success writes no termination message: %v", err)
	}
	other := filepath.Join(dir, "other")
	if err := WriteTerminationMessage(other, errors.New("api down")); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(other)
	if err != nil {
		t.Fatal(err)
	}
	var got TerminalOutcome
	if err := json.Unmarshal(data, &got); err != nil || got.Error != "api down" || got.Phase != "" {
		t.Fatalf("got %+v, err %v", got, err)
	}
}

func TestOnlyAControllerOwnedObjectBelongsToTheRun(t *testing.T) {
	candidate := func(owners []metav1.OwnerReference) *v1beta2.DynamoGraphDeploymentCandidate {
		return &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{OwnerReferences: owners}}
	}
	notController := controllerRef(testRunUID)
	notController[0].Controller = nil
	cases := map[string]struct {
		owners []metav1.OwnerReference
		want   bool
	}{
		"controlled by the run":       {controllerRef(testRunUID), true},
		"controlled by another run":   {controllerRef("another-run-uid"), false},
		"no owner":                    {nil, false},
		"owned but not as controller": {notController, false},
	}
	for name, tc := range cases {
		if got := controlledByRun(candidate(tc.owners), types.UID(testRunUID)); got != tc.want {
			t.Errorf("%s: got %t, want %t", name, got, tc.want)
		}
	}
	if controlledByRun(candidate(controllerRef(testRunUID)), "") {
		t.Error("an empty run UID must never match")
	}
}
