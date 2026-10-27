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
)

func TestTerminalResultIsAStructuredRunFailure(t *testing.T) {
	var failure *RunFailedError
	err := terminalResult(DesiredState{Phase: PhaseFailed, Message: "m", Error: "e"})
	if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || failure.Outcome != (TerminalOutcome{Phase: PhaseFailed, Message: "m", Error: "e"}) {
		t.Fatalf("failed: %v", err)
	}
	if got, want := err.Error(), ErrRunFailed.Error()+": m: e"; got != want {
		t.Fatalf("error text = %q, want %q", got, want)
	}
	err = terminalResult(DesiredState{Phase: PhaseSucceeded, Failed: 1, FirstFailure: "x: render failed"})
	if !errors.As(err, &failure) || failure.Outcome.Error != "x: render failed" || !strings.Contains(failure.Outcome.Message, "1 candidate materialization failure") {
		t.Fatalf("empty success: %v", err)
	}
}

func TestDesireCarriesBoundedDiagnostics(t *testing.T) {
	long := strings.Repeat("e", 3*maxDiagnosticBytes)
	state := desire(t, withRunDiagnostics(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}), "no recommendation\ncould be materialized", long))
	if state.Phase != PhaseFailed || state.Message != "no recommendation could be materialized" {
		t.Fatalf("state = %+v", state)
	}
	if len(state.Error) > maxDiagnosticBytes || !strings.HasPrefix(state.Error, "eee") {
		t.Fatalf("error must be kept but bounded, got %d bytes", len(state.Error))
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

func TestTerminalSucceededWithoutAMaterializedCandidateCarriesTheCandidateFailure(t *testing.T) {
	f := newFixture(t)
	f.publish(snapshotJSON(t, PhaseSucceeded, cand{id: "x", failed: true}))
	f.mustReconcile()
	var failure *RunFailedError
	err := f.r.Completion.Err()
	if !errors.Is(err, ErrRunFailed) || !errors.As(err, &failure) || ExitCode(err) != ExitRunFailed {
		t.Fatalf("err = %v", err)
	}
	if !strings.HasPrefix(failure.Outcome.Error, "x: ") {
		t.Fatalf("the candidate failure must be part of the diagnostics: %+v", failure.Outcome)
	}
}

func TestTerminalFailureDiagnosticsSurviveTheCompletionHandoff(t *testing.T) {
	f := newFixture(t)
	message := "no recommendation could be materialized"
	long := "backend vllm rejected the generated worker configuration " + strings.Repeat("x", 3*maxDiagnosticBytes)
	f.publish(withRunDiagnostics(t, snapshotJSON(t, PhaseFailed, cand{id: "a"}), message, long))
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
