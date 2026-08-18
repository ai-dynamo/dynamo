/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package testing

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestPollImmediateSuccess(t *testing.T) {
	t.Log("Return immediately when the condition succeeds on its first call")
	calls := 0
	poll(t.Context(), t, func() bool {
		calls++
		return true
	}, 1, time.Second, 10*time.Millisecond)
	require.Equal(t, 1, calls)
}

func TestPollSucceedsAfterRetries(t *testing.T) {
	t.Log("Retry until the condition succeeds")
	var calls atomic.Int32
	poll(t.Context(), t, func() bool {
		return calls.Add(1) >= 3
	}, 1, time.Second, 10*time.Millisecond)
	require.GreaterOrEqual(t, calls.Load(), int32(3))
}

func TestPollTimeout(t *testing.T) {
	t.Log("Mark the test failed when the condition never succeeds")
	mock := &mockT{}
	poll(context.Background(), mock, func() bool {
		return false
	}, 1, 50*time.Millisecond, 10*time.Millisecond)
	require.True(t, mock.failed)
}

func TestPollContextCancellation(t *testing.T) {
	t.Log("Stop polling and fail when the context is cancelled")
	mock := &mockT{}
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()

	poll(ctx, mock, func() bool {
		return false
	}, 1, 5*time.Second, 10*time.Millisecond)
	require.True(t, mock.failed)
}

func TestPollRequireInsideConditionIsSafe(t *testing.T) {
	t.Log("Run require assertions inside the condition on the test goroutine")
	calls := 0
	poll(t.Context(), t, func() bool {
		calls++
		require.NotNil(t, t)
		return calls >= 2
	}, 1, time.Second, 10*time.Millisecond)
}

func TestPollWatchdogWarnsOnSlowCondition(t *testing.T) {
	t.Log("Warn when one condition call exceeds the tick interval")
	mock := &mockT{}
	tick := 20 * time.Millisecond
	calls := 0
	poll(context.Background(), mock, func() bool {
		calls++
		if calls == 1 {
			time.Sleep(3 * tick)
		}
		return calls >= 3
	}, 1, time.Second, tick)

	require.False(t, mock.failed)
	require.Contains(t, mock.logOutput(), "WARNING")
}

func TestPollWatchdogWarnsOnlyOnce(t *testing.T) {
	t.Log("Warn only once when multiple condition calls are slow")
	mock := &mockT{}
	tick := 20 * time.Millisecond
	calls := 0
	poll(context.Background(), mock, func() bool {
		calls++
		time.Sleep(3 * tick)
		return calls >= 4
	}, 1, 2*time.Second, tick)

	require.False(t, mock.failed)
	require.Equal(t, 1, strings.Count(mock.logOutput(), "WARNING"))
}

func TestPollWatchdogWarningIncludesCallerLocation(t *testing.T) {
	t.Log("Include the caller location in watchdog warnings")
	mock := &mockT{}
	tick := 20 * time.Millisecond
	calls := 0
	poll(context.Background(), mock, func() bool {
		calls++
		if calls <= 2 {
			time.Sleep(3 * tick)
		}
		return calls >= 3
	}, 1, time.Second, tick)

	require.Contains(t, mock.logOutput(), "eventually_test.go:")
}

func TestEventuallyWithReasonLogsReason(t *testing.T) {
	t.Log("Log a distinct waiting reason after 20 percent of the timeout")
	calls := 0
	mock := &mockT{}
	eventuallyWithReason(context.Background(), mock, func() (bool, string) {
		calls++
		if calls < 15 {
			return false, "not ready yet"
		}
		return true, "done"
	}, 2, 500*time.Millisecond, 10*time.Millisecond)

	require.False(t, mock.failed)
	require.Contains(t, mock.logOutput(), "Waiting but got: not ready yet")
}

func TestEventuallyImmediateSuccess(t *testing.T) {
	t.Log("Return immediately when Eventually succeeds on its first call")
	calls := 0
	Eventually(t, func() (bool, string) {
		calls++
		return true, ""
	}, time.Second, 10*time.Millisecond)
	require.Equal(t, 1, calls)
}

func TestEventuallyWatchdogWarningIncludesExternalCaller(t *testing.T) {
	t.Log("Run an Eventually condition that exceeds its watchdog interval")
	output := runHelperTest(t, "TestHelperEventuallyWatchdogWarning")

	t.Log("Point the warning at the test call site instead of the Eventually wrapper")
	var warning string
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, "WARNING: Eventually condition at") {
			warning = line
			break
		}
	}
	require.Contains(t, warning, "eventually_test.go:")
	require.NotContains(t, warning, "/eventually.go:")
}

func TestHelperEventuallyWatchdogWarning(t *testing.T) {
	if os.Getenv("EVENTUALLY_TEST_HELPER") != "1" {
		t.Skip("helper test run by TestEventuallyWatchdogWarningIncludesExternalCaller")
	}
	calls := 0
	Eventually(t, func() (bool, string) {
		calls++
		if calls == 1 {
			time.Sleep(60 * time.Millisecond)
		}
		return calls >= 2, ""
	}, time.Second, 20*time.Millisecond)
}

func TestPollTimeoutReportsCallerLine(t *testing.T) {
	t.Log("Point timeout failures at the Eventually caller")
	output := runHelperTest(t, "TestHelperPollTimeout")
	errorTrace := extractErrorTrace(output)
	require.NotEmpty(t, errorTrace)
	require.Contains(t, errorTrace[len(errorTrace)-1], "eventually_test.go")
}

func TestHelperPollTimeout(t *testing.T) {
	if os.Getenv("EVENTUALLY_TEST_HELPER") != "1" {
		t.Skip("helper test run by TestPollTimeoutReportsCallerLine")
	}
	Eventually(t, func() (bool, string) {
		return false, "never ready"
	}, 50*time.Millisecond, 10*time.Millisecond)
}

func TestPollRequireFailureReportsConditionLine(t *testing.T) {
	t.Log("Point require failures at the condition")
	output := runHelperTest(t, "TestHelperRequireFailure")
	errorTrace := extractErrorTrace(output)
	require.NotEmpty(t, errorTrace)
	require.Contains(t, errorTrace[0], "eventually_test.go")
}

func TestHelperRequireFailure(t *testing.T) {
	if os.Getenv("EVENTUALLY_TEST_HELPER") != "1" {
		t.Skip("helper test run by TestPollRequireFailureReportsConditionLine")
	}
	Eventually(t, func() (bool, string) {
		require.Fail(t, "deliberate failure from condition")
		return true, ""
	}, time.Second, 10*time.Millisecond)
}

func runHelperTest(t *testing.T, testName string) string {
	t.Helper()
	executable, err := os.Executable()
	require.NoError(t, err)

	command := exec.Command(executable, "-test.run=^"+testName+"$", "-test.v")
	command.Env = append(os.Environ(), "EVENTUALLY_TEST_HELPER=1")
	output, _ := command.CombinedOutput()
	return string(output)
}

func extractErrorTrace(output string) []string {
	var trace []string
	inTrace := false
	for _, line := range strings.Split(output, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "Error Trace:") {
			parts := strings.SplitN(trimmed, "\t", 2)
			if len(parts) == 2 {
				trace = append(trace, strings.TrimSpace(parts[1]))
			}
			inTrace = true
			continue
		}
		if !inTrace {
			continue
		}
		if strings.Contains(trimmed, ".go:") && !strings.Contains(trimmed, "Error:") {
			trace = append(trace, trimmed)
		} else {
			break
		}
	}
	return trace
}

type mockT struct {
	failed bool
	logs   []string
}

func (m *mockT) Errorf(_ string, _ ...any) {
	m.failed = true
}

func (m *mockT) FailNow() {
	m.failed = true
}

func (m *mockT) Helper() {}

func (m *mockT) Logf(format string, args ...any) {
	m.logs = append(m.logs, fmt.Sprintf(format, args...))
}

func (m *mockT) logOutput() string {
	return strings.Join(m.logs, "\n")
}
