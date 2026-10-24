/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"errors"
	"fmt"
	"sync"
)

// Participant pins an existing DP rank to one API process observed before submission.
// Client must not be nil. The expected instance must never be refreshed merely to make a retry succeed.
type Participant struct {
	Rank               int32
	ExpectedInstanceID string
	Client             *Client
}

// Resize describes the native projection of one already-persisted membership target.
// Participants includes every old rank, including ranks that will leave on suffix shrink.
// Joiners are launched separately; this control does not wait for their serving readiness before submission.
type Resize struct {
	OperationID          string
	BaseDataParallelSize int32
	NewDataParallelSize  int32
	DrainTimeout         int32
	Participants         []Participant
}

// RankResult preserves each invocation independently; a mixed fan-out is not a group rejection.
// Status is authoritative only when Err is nil, and is native evidence rather than a serving proof.
type RankResult struct {
	Rank   int32
	Status Status
	Err    error
}

// Apply sends an identical operation and target concurrently to all old-rank API processes.
// The resize must already be durable before this call. No background task or in-memory transaction survives it.
// An invocation failure does not cancel peer requests: they may already be executing the collective.
// Any returned error requires observation and, where safe, replay of this identical resize, never rollback by HTTP code.
func (r Resize) Apply(ctx context.Context) ([]RankResult, error) {
	// Validate the entire fan-out before allowing the first external effect.
	if err := r.validate(); err != nil {
		return nil, err
	}
	results := make([]RankResult, len(r.Participants))
	var pending sync.WaitGroup
	for i, participant := range r.Participants {
		pending.Go(func() {
			status, err := participant.Client.Scale(ctx, ScaleRequest{
				OperationID: r.OperationID, ExpectedInstanceID: participant.ExpectedInstanceID,
				NewDataParallelSize: r.NewDataParallelSize, DrainTimeout: r.DrainTimeout,
			})
			results[i] = RankResult{Rank: participant.Rank, Status: status, Err: err}
		})
	}
	pending.Wait()

	// Preserve successful peer evidence even when another request failed or lost its response.
	var failures []error
	for _, result := range results {
		if result.Err != nil {
			failures = append(failures, fmt.Errorf("vLLM DP rank %d: %w", result.Rank, result.Err))
		}
	}
	return results, errors.Join(failures...)
}

func (r Resize) validate() error {
	// Planned resizing is over a complete, healthy old world, not a sparse survivor mask.
	if r.BaseDataParallelSize <= 0 || int64(len(r.Participants)) != int64(r.BaseDataParallelSize) {
		return fmt.Errorf("vLLM external resize requires every rank of a positive base DP world")
	}
	if r.NewDataParallelSize == r.BaseDataParallelSize {
		return fmt.Errorf("an unchanged vLLM target is not a membership transition")
	}
	ranks := make(map[int32]bool, len(r.Participants))
	instances := make(map[string]bool, len(r.Participants))
	for _, participant := range r.Participants {
		if participant.Rank < 0 || participant.Rank >= r.BaseDataParallelSize || ranks[participant.Rank] {
			return fmt.Errorf("vLLM participants must contain every old DP rank exactly once")
		}
		if instances[participant.ExpectedInstanceID] {
			return fmt.Errorf("vLLM participants must identify independent API processes")
		}
		request := ScaleRequest{
			OperationID: r.OperationID, ExpectedInstanceID: participant.ExpectedInstanceID,
			NewDataParallelSize: r.NewDataParallelSize, DrainTimeout: r.DrainTimeout,
		}
		if err := request.validate(); err != nil {
			return err
		}
		ranks[participant.Rank] = true
		instances[participant.ExpectedInstanceID] = true
	}
	return nil
}
