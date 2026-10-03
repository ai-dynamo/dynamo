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

package namespace_scope

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// leaseWorkResult carries a worker panic back to the goroutine that owns cleanup.
type leaseWorkResult struct {
	err        error
	panicValue any
	panicked   bool
}

// Guard holds the namespace scope marker lease while it runs work, releasing the
// lease before returning or propagating a worker panic. ctx and work must be non-nil.
// cleanupTimeout must be positive. The caller must terminate after a fatal lease error.
func (lm *LeaseManager) Guard(ctx context.Context, cleanupTimeout time.Duration, work func(context.Context) error) error {
	if cleanupTimeout <= 0 {
		return errors.New("lease cleanup timeout must be positive")
	}

	// Register cleanup before acquisition: a failed response does not prove the write failed.
	// Stop checks holder identity and resource version before deleting anything.
	var shutdownDeadline time.Time
	defer func() {
		deadline := time.Now().Add(cleanupTimeout)
		if !shutdownDeadline.IsZero() && shutdownDeadline.Before(deadline) {
			deadline = shutdownDeadline
		}
		cleanupCtx, cancelCleanup := context.WithDeadline(context.Background(), deadline)
		defer cancelCleanup()
		if err := lm.Stop(cleanupCtx); err != nil {
			lm.logger.Error(err, "Failed to stop namespace scope marker lease manager cleanly")
		}
	}()
	if err := lm.Start(ctx); err != nil {
		return fmt.Errorf("failed to start namespace scope marker lease manager: %w", err)
	}

	// Do not launch work under an acquisition whose response arrived too late.
	renewalDeadline, _ := lm.leaseDeadlines()
	if !time.Now().Before(renewalDeadline) {
		_, shutdownDeadline = lm.leaseDeadlines()
		return errors.New("initial namespace lease response arrived after the renewal deadline")
	}

	// Work cancellation is independent of the renewal context owned by the manager.
	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()

	// Recover only to move the panic to the cleanup-owning goroutine, then re-panic there.
	// The buffer also lets abandoned work finish without blocking its result delivery.
	workDone := make(chan leaseWorkResult, 1)
	go func() {
		result := leaseWorkResult{panicked: true}
		defer func() {
			result.panicValue = recover()
			workDone <- result
		}()
		result.err = work(workCtx)
		result.panicked = false
	}()

	// Watch expiry independently of RPC completion, reserving a shutdown window and
	// a final safety margin instead of assuming a whole renewInterval remains.
	expired := time.NewTimer(time.Until(renewalDeadline))
	defer expired.Stop()
	var result leaseWorkResult
	var fatalErr error
watch:
	for {
		select {
		case result = <-workDone:
			_, shutdownDeadline = lm.leaseDeadlines()
			break watch
		case fatalErr = <-lm.Errors():
			break watch
		case <-lm.leaseUpdated:
			renewalDeadline, _ = lm.leaseDeadlines()
			expired.Reset(time.Until(renewalDeadline))
		case <-expired.C:
			// A successful renewal may have raced the timer; consult the latest expiry.
			renewalDeadline, _ = lm.leaseDeadlines()
			if time.Now().Before(renewalDeadline) {
				expired.Reset(time.Until(renewalDeadline))
				continue
			}
			fatalErr = errors.New("lease renewal did not complete before the shutdown window")
			break watch
		}
	}

	// Derive both waits from the actual remaining lifetime, without a positive floor.
	// Cleanup shares the same absolute deadline, so a late work return cannot extend it.
	if fatalErr != nil {
		cancelWork()
		_, shutdownDeadline = lm.leaseDeadlines()
		remaining := max(0, time.Until(shutdownDeadline))
		cleanupBudget := min(cleanupTimeout, remaining/2)
		expired.Reset(remaining - cleanupBudget)
		select {
		case result = <-workDone:
		case <-expired.C:
			lm.logger.Error(nil, "Giving up on orderly shutdown before namespace lease expiry")
		}
	}

	// Re-panicking here runs Guard's deferred lease cleanup first.
	if result.panicked {
		panic(result.panicValue)
	}
	if fatalErr != nil && (result.err == nil || errors.Is(result.err, context.Canceled)) {
		return fmt.Errorf("namespace scope marker lease is unrecoverable: %w", fatalErr)
	}
	return result.err
}
