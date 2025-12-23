/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"fmt"
	"sync"
	"time"
)

// Task represents a unit of work to be executed
type Task[T any] struct {
	Index int
	Work  func(ctx context.Context) (T, error)
}

// Result represents the outcome of executing a task
type Result[T any] struct {
	Index int
	Value T
	Err   error
}

// Execute runs all tasks in parallel with bounded concurrency using a worker pool
// Returns results in the same order as input tasks, even if execution order differs
// Continues executing all tasks even if some fail
// Spawns exactly maxWorkers goroutines regardless of task count
func Execute[T any](ctx context.Context, maxWorkers int, timeout time.Duration, tasks []Task[T]) ([]Result[T], error) {
	// Validate maxWorkers to prevent panics or hangs
	if maxWorkers < 1 {
		return nil, fmt.Errorf("maxWorkers must be at least 1, got %d", maxWorkers)
	}

	if len(tasks) == 0 {
		return nil, nil
	}

	// Create context with timeout
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Create channels
	taskChan := make(chan Task[T])
	results := make(chan Result[T], len(tasks))

	// Start exactly maxWorkers worker goroutines
	var wg sync.WaitGroup
	for range maxWorkers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Each worker pulls tasks from the channel until it's closed
			for task := range taskChan {
				// Execute the task
				value, err := task.Work(execCtx)

				// Send result through channel
				results <- Result[T]{
					Index: task.Index,
					Value: value,
					Err:   err,
				}
			}
		}()
	}

	// Feed tasks to workers in a separate goroutine to avoid blocking
	go func() {
		for _, task := range tasks {
			taskChan <- task
		}
		close(taskChan) // Signal workers that no more tasks are coming
	}()

	// Close results channel when all workers complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results from channel
	collectedResults := make([]Result[T], len(tasks))
	var errorCount int

	for result := range results {
		collectedResults[result.Index] = result
		if result.Err != nil {
			errorCount++
		}
	}

	// Return error if any tasks failed
	if errorCount > 0 {
		return collectedResults, fmt.Errorf("%d task(s) failed", errorCount)
	}

	return collectedResults, nil
}
