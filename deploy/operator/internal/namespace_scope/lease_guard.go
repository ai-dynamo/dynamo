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

// unwindBudgetFloor keeps the post-fatal wait useful when renew interval and cleanup
// timeout leave nothing to spend on it.
const unwindBudgetFloor = time.Second

// unwindBudget is how long Guard waits for work once lease renewal is unrecoverable: the
// renewal loop leaves one renewInterval before expiry, and cleanupTimeout comes out of it.
func unwindBudget(renewInterval, cleanupTimeout time.Duration) time.Duration {
	budget := renewInterval - cleanupTimeout
	if budget < unwindBudgetFloor {
		return unwindBudgetFloor
	}

	return budget
}

// Deprecated: Guard holds the namespace scope marker lease while it runs work, releasing the
// lease before returning. ctx and work must be non-nil.
func (lm *LeaseManager) Guard(ctx context.Context, cleanupTimeout time.Duration, work func(context.Context) error) error {
	if err := lm.Start(ctx); err != nil {
		return fmt.Errorf("failed to start namespace scope marker lease manager: %w", err)
	}

	// The lease exists from here on, so every path out of this function has to release it.
	defer func() {
		cleanupCtx, cancelCleanup := context.WithTimeout(context.Background(), cleanupTimeout)
		defer cancelCleanup()

		if err := lm.Stop(cleanupCtx); err != nil {
			lm.logger.Error(err, "Failed to stop namespace scope marker lease manager cleanly")
		}
	}()

	workCtx, cancelWork := context.WithCancel(ctx)
	defer cancelWork()

	// work runs on its own goroutine so an unrecoverable lease error can stop waiting for it.
	// The buffer keeps abandoned work from blocking forever on the send.
	workDone := make(chan error, 1)
	go func() {
		workDone <- work(workCtx)
	}()

	var workErr, fatalErr error

	select {
	case workErr = <-workDone:
	case fatalErr = <-lm.Errors():
		// Cancelling work instead of ending the process lets the deferred release run first.
		// The wait is bounded because the lease is expiring while it happens.
		cancelWork()

		budget := unwindBudget(lm.renewInterval, cleanupTimeout)
		expired := time.NewTimer(budget)
		defer expired.Stop()

		select {
		case workErr = <-workDone:
		case <-expired.C:
			lm.logger.Error(nil, "Giving up on orderly shutdown after unrecoverable lease failure; releasing the lease before it expires to prevent split-brain",
				"unwindBudget", budget)
		}
	}

	if fatalErr != nil && (workErr == nil || errors.Is(workErr, context.Canceled)) {
		return fmt.Errorf("namespace scope marker lease is unrecoverable: %w", fatalErr)
	}

	return workErr
}
