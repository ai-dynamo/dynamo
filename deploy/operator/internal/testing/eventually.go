/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package testing provides shared test helpers for the Dynamo operator.
package testing

import (
	"context"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// pollT is the minimal interface needed by the synchronous polling loop.
// testing.TB cannot be implemented outside the standard library.
type pollT interface {
	Helper()
	Logf(format string, args ...any)
	Errorf(format string, args ...any)
	FailNow()
}

// poll runs conditions on the test goroutine so conditions may safely use
// require functions that call FailNow.
func poll(
	ctx context.Context,
	t pollT,
	condition func() bool,
	callerSkip int,
	waitFor time.Duration,
	tick time.Duration,
	msgAndArgs ...any,
) {
	t.Helper()

	callerLocation := "unknown"
	if _, file, line, ok := runtime.Caller(callerSkip); ok {
		callerLocation = fmt.Sprintf("%s:%d", file, line)
	}

	deadline := time.Now().Add(waitFor)
	ticker := time.NewTicker(tick)
	defer ticker.Stop()

	condRunning := make(chan struct{}, 1)
	condDone := make(chan struct{}, 1)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		warned := false
		for {
			select {
			case <-stop:
				return
			case <-condRunning:
				select {
				case <-condDone:
				case <-time.After(tick):
					if !warned {
						warned = true
						t.Logf("WARNING: Eventually condition at %s has been running for longer than the tick interval (%v)", callerLocation, tick)
					}
					select {
					case <-condDone:
					case <-stop:
						return
					}
				case <-stop:
					return
				}
			}
		}
	}()

	check := func() bool {
		t.Helper()
		condRunning <- struct{}{}
		defer func() { condDone <- struct{}{} }()
		return condition()
	}

	if check() {
		return
	}

	for {
		select {
		case <-ctx.Done():
			require.Fail(t, fmt.Sprintf("Context cancelled before condition was satisfied: %v", ctx.Err()), msgAndArgs...)
			return
		case <-ticker.C:
			if check() {
				return
			}
			if time.Now().After(deadline) {
				require.Fail(t, "Condition never satisfied", msgAndArgs...)
				return
			}
		}
	}
}

func eventuallyWithReason(
	ctx context.Context,
	t pollT,
	condition func() (success bool, reason string),
	callerSkip int,
	waitFor time.Duration,
	tick time.Duration,
	msgAndArgs ...any,
) {
	t.Helper()

	var last string
	start := time.Now()
	poll(ctx, t, func() bool {
		t.Helper()

		ok, message := condition()
		if time.Since(start) > waitFor/5 {
			if !ok && message != "" && message != last {
				last = message
				t.Logf("Waiting but got: %s", message)
			} else if ok && message != "" && last != "" {
				t.Logf("Wait finished: %s", message)
			}
		}
		return ok
	}, callerSkip, waitFor, tick, msgAndArgs...)
}

// Eventually requires condition to succeed within waitFor. It logs distinct
// failure reasons after 20 percent of waitFor has elapsed.
func Eventually(
	t testing.TB,
	condition func() (success bool, reason string),
	waitFor time.Duration,
	tick time.Duration,
	msgAndArgs ...any,
) {
	t.Helper()
	eventuallyWithReason(t.Context(), t, condition, 3, waitFor, tick, msgAndArgs...)
}
