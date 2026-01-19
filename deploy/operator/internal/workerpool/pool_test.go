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

package workerpool

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestExecute(t *testing.T) {
	tests := []struct {
		name               string
		maxWorkers         int
		timeout            time.Duration
		taskCount          int
		taskDuration       time.Duration
		failingTaskIndices []int
		expectError        bool
		errorContains      string
	}{
		{
			name:        "empty task list",
			maxWorkers:  5,
			timeout:     time.Second,
			taskCount:   0,
			expectError: false,
		},
		{
			name:         "single task success",
			maxWorkers:   1,
			timeout:      time.Second,
			taskCount:    1,
			taskDuration: 10 * time.Millisecond,
			expectError:  false,
		},
		{
			name:         "multiple tasks success",
			maxWorkers:   5,
			timeout:      time.Second,
			taskCount:    10,
			taskDuration: 10 * time.Millisecond,
			expectError:  false,
		},
		{
			name:               "single task failure",
			maxWorkers:         5,
			timeout:            time.Second,
			taskCount:          5,
			taskDuration:       10 * time.Millisecond,
			failingTaskIndices: []int{2},
			expectError:        true,
			errorContains:      "1 task(s) failed",
		},
		{
			name:               "multiple task failures",
			maxWorkers:         5,
			timeout:            time.Second,
			taskCount:          10,
			taskDuration:       10 * time.Millisecond,
			failingTaskIndices: []int{1, 3, 5},
			expectError:        true,
			errorContains:      "3 task(s) failed",
		},
		{
			name:         "more tasks than workers",
			maxWorkers:   3,
			timeout:      time.Second,
			taskCount:    10,
			taskDuration: 10 * time.Millisecond,
			expectError:  false,
		},
		{
			name:         "more workers than tasks",
			maxWorkers:   10,
			timeout:      time.Second,
			taskCount:    3,
			taskDuration: 10 * time.Millisecond,
			expectError:  false,
		},
		{
			name:         "single worker multiple tasks",
			maxWorkers:   1,
			timeout:      time.Second,
			taskCount:    5,
			taskDuration: 10 * time.Millisecond,
			expectError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			// Create tasks
			tasks := make([]Task[int], tt.taskCount)
			failingSet := make(map[int]bool)
			for _, idx := range tt.failingTaskIndices {
				failingSet[idx] = true
			}

			for i := range tasks {
				taskIndex := i
				tasks[i] = Task[int]{
					Index: taskIndex,
					Work: func(ctx context.Context) (int, error) {
						// Simulate work
						if tt.taskDuration > 0 {
							time.Sleep(tt.taskDuration)
						}

						// Return error if this task should fail
						if failingSet[taskIndex] {
							return 0, fmt.Errorf("task %d failed", taskIndex)
						}

						return taskIndex * 2, nil
					},
				}
			}

			// Execute tasks
			results, err := Execute(ctx, tt.maxWorkers, tt.timeout, tasks)

			// Verify error expectation
			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				} else if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}

			// Verify result count
			if len(results) != tt.taskCount {
				t.Errorf("expected %d results, got %d", tt.taskCount, len(results))
			}

			// Verify successful task results
			for i, result := range results {
				if result.Index != i {
					t.Errorf("result %d has wrong index: expected %d, got %d", i, i, result.Index)
				}

				if !failingSet[i] {
					// Successful tasks should have correct value
					expectedValue := i * 2
					if result.Value != expectedValue {
						t.Errorf("result %d has wrong value: expected %d, got %d", i, expectedValue, result.Value)
					}
					if result.Err != nil {
						t.Errorf("result %d has unexpected error: %v", i, result.Err)
					}
				} else {
					// Failed tasks should have error
					if result.Err == nil {
						t.Errorf("result %d should have error but got none", i)
					}
				}
			}
		})
	}
}

func TestExecute_InvalidMaxWorkers(t *testing.T) {
	tests := []struct {
		name          string
		maxWorkers    int
		errorContains string
	}{
		{
			name:          "zero workers",
			maxWorkers:    0,
			errorContains: "maxWorkers must be at least 1",
		},
		{
			name:          "negative workers",
			maxWorkers:    -1,
			errorContains: "maxWorkers must be at least 1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			tasks := []Task[int]{
				{
					Index: 0,
					Work: func(ctx context.Context) (int, error) {
						return 0, nil
					},
				},
			}

			_, err := Execute(ctx, tt.maxWorkers, time.Second, tasks)

			if err == nil {
				t.Error("expected error but got none")
			} else if !strings.Contains(err.Error(), tt.errorContains) {
				t.Errorf("expected error to contain %q, got %v", tt.errorContains, err)
			}
		})
	}
}

func TestExecute_Timeout(t *testing.T) {
	ctx := context.Background()

	// Create tasks that take longer than the timeout
	tasks := []Task[int]{
		{
			Index: 0,
			Work: func(ctx context.Context) (int, error) {
				select {
				case <-time.After(2 * time.Second):
					return 0, nil
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			},
		},
		{
			Index: 1,
			Work: func(ctx context.Context) (int, error) {
				select {
				case <-time.After(2 * time.Second):
					return 1, nil
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			},
		},
	}

	// Execute with short timeout
	results, err := Execute(ctx, 2, 100*time.Millisecond, tasks)

	// Should get error because tasks timed out
	if err == nil {
		t.Error("expected timeout error but got none")
	}

	// Should still get results (with errors)
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// All results should have context deadline exceeded error
	for i, result := range results {
		if result.Err == nil {
			t.Errorf("result %d should have timeout error but got none", i)
		}
	}
}

func TestExecute_Concurrency(t *testing.T) {
	ctx := context.Background()
	maxWorkers := 5
	taskCount := 20

	// Track concurrent execution
	var currentConcurrent int32
	var maxConcurrent int32

	tasks := make([]Task[int], taskCount)
	for i := range tasks {
		taskIndex := i
		tasks[i] = Task[int]{
			Index: taskIndex,
			Work: func(ctx context.Context) (int, error) {
				// Increment counter
				current := atomic.AddInt32(&currentConcurrent, 1)

				// Update max if needed
				for {
					max := atomic.LoadInt32(&maxConcurrent)
					if current <= max || atomic.CompareAndSwapInt32(&maxConcurrent, max, current) {
						break
					}
				}

				// Simulate work
				time.Sleep(50 * time.Millisecond)

				// Decrement counter
				atomic.AddInt32(&currentConcurrent, -1)

				return taskIndex, nil
			},
		}
	}

	_, err := Execute(ctx, maxWorkers, 5*time.Second, tasks)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Verify concurrency stayed within bounds
	if maxConcurrent > int32(maxWorkers) {
		t.Errorf("expected max concurrent workers <= %d, got %d", maxWorkers, maxConcurrent)
	}

	// Verify we actually used concurrency (should be at least 2 concurrent)
	if maxConcurrent < 2 {
		t.Errorf("expected concurrent execution, but maxConcurrent was only %d", maxConcurrent)
	}
}

func TestExecute_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// Create tasks that check for cancellation
	tasks := make([]Task[int], 5)
	for i := range tasks {
		taskIndex := i
		tasks[i] = Task[int]{
			Index: taskIndex,
			Work: func(ctx context.Context) (int, error) {
				select {
				case <-time.After(2 * time.Second):
					return taskIndex, nil
				case <-ctx.Done():
					return 0, ctx.Err()
				}
			},
		}
	}

	// Cancel context after short delay
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	results, err := Execute(ctx, 3, 5*time.Second, tasks)

	// Should get error
	if err == nil {
		t.Error("expected cancellation error but got none")
	}

	// Should still get results
	if len(results) != 5 {
		t.Errorf("expected 5 results, got %d", len(results))
	}

	// All results should have cancellation error
	for i, result := range results {
		if result.Err == nil {
			t.Errorf("result %d should have cancellation error but got none", i)
		} else if !errors.Is(result.Err, context.Canceled) {
			t.Errorf("result %d expected context.Canceled, got %v", i, result.Err)
		}
	}
}

func TestExecute_ResultOrdering(t *testing.T) {
	ctx := context.Background()
	taskCount := 10

	// Create tasks that complete in reverse order
	tasks := make([]Task[int], taskCount)
	for i := range tasks {
		taskIndex := i
		tasks[i] = Task[int]{
			Index: taskIndex,
			Work: func(ctx context.Context) (int, error) {
				// Later tasks sleep less (complete faster)
				sleepDuration := time.Duration(taskCount-taskIndex) * 10 * time.Millisecond
				time.Sleep(sleepDuration)
				return taskIndex * 10, nil
			},
		}
	}

	results, err := Execute(ctx, 5, 5*time.Second, tasks)

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Verify results are in original order despite reverse completion
	for i, result := range results {
		if result.Index != i {
			t.Errorf("result %d has wrong index: expected %d, got %d", i, i, result.Index)
		}
		expectedValue := i * 10
		if result.Value != expectedValue {
			t.Errorf("result %d has wrong value: expected %d, got %d", i, expectedValue, result.Value)
		}
	}
}
