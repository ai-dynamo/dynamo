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

// Execute runs all tasks in parallel with bounded concurrency
// Returns results in the same order as input tasks, even if execution order differs
// Continues executing all tasks even if some fail
func Execute[T any](ctx context.Context, maxWorkers int, timeout time.Duration, tasks []Task[T]) ([]Result[T], error) {
	if len(tasks) == 0 {
		return nil, nil
	}

	// Create context with timeout
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Create buffered channels
	results := make(chan Result[T], len(tasks))
	semaphore := make(chan struct{}, maxWorkers)

	// Launch all task goroutines
	var wg sync.WaitGroup
	for _, task := range tasks {
		wg.Add(1)
		go func(t Task[T]) {
			defer wg.Done()

			// Acquire semaphore slot (bounded concurrency)
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Execute the task
			value, err := t.Work(execCtx)

			// Send result through channel
			results <- Result[T]{
				Index: t.Index,
				Value: value,
				Err:   err,
			}
		}(task)
	}

	// Close results channel when all goroutines complete
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
