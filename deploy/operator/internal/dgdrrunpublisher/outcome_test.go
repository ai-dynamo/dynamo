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
	"errors"
	"fmt"
	"testing"
)

func TestExitCodes(t *testing.T) {
	cases := []struct {
		err  error
		want int
	}{
		{nil, ExitAcknowledged},
		{ErrRunFailed, ExitRunFailed},
		{ErrMissingTerminalSnapshot, ExitProtocolViolation},
		{violation("bad"), ExitProtocolViolation},
		{fmt.Errorf("reconciling last snapshot: %w", violation("bad")), ExitProtocolViolation},
		{errors.New("api down"), ExitReconcileFailed},
	}
	for _, tc := range cases {
		if got := ExitCode(tc.err); got != tc.want {
			t.Errorf("ExitCode(%v) = %d, want %d", tc.err, got, tc.want)
		}
	}
}

func TestTerminalResult(t *testing.T) {
	if err := terminalResult(PhaseSucceeded); err != nil {
		t.Fatalf("succeeded: %v", err)
	}
	if err := terminalResult(PhaseFailed); !errors.Is(err, ErrRunFailed) {
		t.Fatalf("failed: %v", err)
	}
}

func TestExitResultWithoutATerminalSnapshot(t *testing.T) {
	if err := exitResult(SweeperState{Exited: true, ExitCode: 137}); err != nil {
		t.Fatalf("a crash is reported by the Sweeper's own exit code, not the publisher: %v", err)
	}
	if err := exitResult(SweeperState{Exited: true}); !errors.Is(err, ErrMissingTerminalSnapshot) {
		t.Fatalf("a clean exit without a terminal snapshot is a protocol violation: %v", err)
	}
}

func TestCompletionKeepsTheFirstOutcome(t *testing.T) {
	c := NewCompletion()
	if c.Finished() {
		t.Fatal("finished before Finish")
	}
	c.Finish(ErrRunFailed)
	c.Finish(nil)
	select {
	case <-c.Done():
	default:
		t.Fatal("Done not closed")
	}
	if !c.Finished() || !errors.Is(c.Err(), ErrRunFailed) {
		t.Fatalf("err = %v", c.Err())
	}
}
