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
	"sync"
)

// Process exit codes of the publisher binary.
const (
	ExitAcknowledged      = 0
	ExitReconcileFailed   = 1
	ExitProtocolViolation = 3
	ExitRunFailed         = 4
)

// ErrMissingTerminalSnapshot means the Sweeper exited successfully without publishing a
// terminal snapshot, which violates the producer protocol.
var ErrMissingTerminalSnapshot = fmt.Errorf("%w: sweeper exited without publishing a terminal snapshot", ErrProtocolViolation)

// ErrRunFailed means the Sweeper reported failure or produced no usable recommendation.
// The snapshot is reconciled first; the error keeps that outcome from letting the Job
// complete successfully. The concrete error is a *RunFailedError carrying the bounded
// diagnostics.
var ErrRunFailed = errors.New("sweeper reported a failed run")

// ExitCode maps the result of Run to the process exit code.
func ExitCode(err error) int {
	switch {
	case err == nil:
		return ExitAcknowledged
	case errors.Is(err, ErrProtocolViolation):
		return ExitProtocolViolation
	case errors.Is(err, ErrRunFailed):
		return ExitRunFailed
	default:
		return ExitReconcileFailed
	}
}

// SweeperState is the observed state of the Sweeper container in this pod.
type SweeperState struct {
	Exited   bool
	ExitCode int32
}

// terminalResult is the outcome once a terminal snapshot has been reconciled. A Failed
// snapshot fails the run with the producer's own (bounded) diagnostics. A Succeeded one
// must also contain a Valid=True candidate: success with nothing to recommend is a
// failed run, not a result. The error is a *RunFailedError, so errors.Is(err,
// ErrRunFailed) and ExitCode are unchanged.
func terminalResult(desired DesiredState) error {
	if desired.Phase == PhaseSucceeded && desired.Valid > 0 {
		return nil
	}
	outcome := TerminalOutcome{Phase: PhaseFailed, Message: desired.Message, Error: desired.Error}
	if desired.Phase == PhaseSucceeded {
		outcome.Message = "no candidate is valid"
		if desired.Failed > 0 {
			noun := "failures"
			if desired.Failed == 1 {
				noun = "failure"
			}
			outcome.Message += fmt.Sprintf(": %d candidate %s", desired.Failed, noun)
		}
		outcome.Error = desired.FirstFailure
	}
	return &RunFailedError{Outcome: outcome}
}

// exitResult is the outcome when the Sweeper exited and the last snapshot it wrote (if
// any) has been reconciled, but that snapshot is not terminal.
func exitResult(state SweeperState) error {
	if state.ExitCode != 0 {
		// Crash or caught failure without a terminal snapshot: the last-known projection
		// is reconciled; the Sweeper's own exit code fails the Job.
		return nil
	}
	return ErrMissingTerminalSnapshot
}

// Completion records the publisher's outcome exactly once and wakes whoever waits on it.
type Completion struct {
	once sync.Once
	done chan struct{}
	err  error
}

// NewCompletion returns an unfinished Completion.
func NewCompletion() *Completion { return &Completion{done: make(chan struct{})} }

// Finish records err as the outcome. Only the first call has an effect.
func (c *Completion) Finish(err error) {
	c.once.Do(func() {
		c.err = err
		close(c.done)
	})
}

// Done is closed once the outcome is known.
func (c *Completion) Done() <-chan struct{} { return c.done }

// Err is the outcome. It is only meaningful after Done is closed.
func (c *Completion) Err() error { return c.err }

// Finished reports whether the outcome is known.
func (c *Completion) Finished() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}
