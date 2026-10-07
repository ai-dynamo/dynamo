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

// Package dgdrrunpublisher is the sidecar that turns the Sweeper's pod-local desired
// state snapshot into Kubernetes state: it creates/deletes DynamoGraphDeploymentCandidates
// and patches the ordered candidate references and progress on
// DynamoGraphDeploymentRun.status.
//
// It runs as a regular container of the same Job pod as the Sweeper, sharing an
// emptyDir with it. There is no ConfigMap and no cross-pod channel: the file is the
// interface. Because the publisher is a regular container, the Job only completes once
// the publisher has exited, so its successful exit is the acknowledgement that the final
// snapshot was reconciled. The run controller derives the terminal Completed condition
// from the Job result; the publisher never writes a phase.
//
// Termination state machine (see Publisher.Run):
//
//	terminal Succeeded snapshot reconciled             -> exit 0
//	terminal Failed snapshot reconciled                -> exit ExitRunFailed
//	Sweeper exited non-zero, no terminal snapshot      -> reconcile last snapshot, exit 0
//	                                                      (the Sweeper's exit fails the Job)
//	Sweeper exited zero, no terminal snapshot          -> reconcile last snapshot,
//	                                                      exit ExitMissingTerminalSnapshot
//	final/last snapshot cannot be reconciled           -> exit ExitReconcileFailed
package dgdrrunpublisher

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
)

// Process exit codes of the publisher binary.
const (
	ExitAcknowledged            = 0
	ExitReconcileFailed         = 1
	ExitMissingTerminalSnapshot = 3
	ExitRunFailed               = 4
)

// ErrMissingTerminalSnapshot means the Sweeper exited successfully without publishing a
// terminal snapshot, which violates the producer protocol.
var ErrMissingTerminalSnapshot = errors.New("sweeper exited without publishing a terminal snapshot")

// ErrRunFailed means the Sweeper published a terminal Failed snapshot. The snapshot is
// reconciled first; the error keeps a producer that reports failure but exits 0 from
// letting the Job complete successfully.
var ErrRunFailed = errors.New("sweeper reported a failed run")

// ExitCode maps the result of Run to the process exit code.
func ExitCode(err error) int {
	switch {
	case err == nil:
		return ExitAcknowledged
	case errors.Is(err, ErrMissingTerminalSnapshot):
		return ExitMissingTerminalSnapshot
	case errors.Is(err, ErrRunFailed):
		return ExitRunFailed
	default:
		return ExitReconcileFailed
	}
}

// SnapshotFileName is the file the Sweeper container writes.
const SnapshotFileName = "dgdr_run_snapshot.yaml"

// RunStatus is the part of DynamoGraphDeploymentRun.status the publisher owns: the
// ordered candidate references plus progress. CandidateNames is in rank order (best
// first for scalar searches). LastProgress is the zero time when the snapshot carries
// no usable timestamp, in which case the existing value is left untouched.
type RunStatus struct {
	Message        string
	Round          int32
	Evaluated      int32
	LastProgress   time.Time
	CandidateNames []string
}

// SweeperState is the observed state of the Sweeper container in this pod.
type SweeperState struct {
	Exited   bool
	ExitCode int32
}

// Cluster is everything the publisher needs from Kubernetes.
type Cluster interface {
	ListCandidates(ctx context.Context) ([]dgdcreconcile.CurrentDGDC, error)
	// CreateCandidate must be idempotent: an already-existing candidate is success.
	CreateCandidate(ctx context.Context, name string, candidate dgdcreconcile.DesiredCandidate) error
	DeleteCandidate(ctx context.Context, name string) error
	PatchRunStatus(ctx context.Context, status RunStatus) error
	// SweeperState reads the Sweeper container's status from this pod.
	SweeperState(ctx context.Context) (SweeperState, error)
}

// Publisher reconciles snapshots into the cluster.
type Publisher struct {
	Cluster      Cluster
	SnapshotDir  string
	RunName      string
	PollInterval time.Duration

	lastRaw      []byte
	lastStatus   *RunStatus
	haveSnapshot bool // a snapshot has been reconciled
	terminal     bool // the reconciled snapshot was terminal
	phase        string
}

// CandidateName is the DGDC name for a candidate id. It depends only on the run and the
// stable id, never on rank.
func CandidateName(runName, id string) string {
	return fmt.Sprintf("%s-%s", runName, id)
}

// Reconcile applies one snapshot: create missing candidates, patch the run status
// (ordered references and progress), then delete obsolete candidates. Existing
// candidates are never updated. It is idempotent and safe to retry.
func (p *Publisher) Reconcile(ctx context.Context, snap *Snapshot) error {
	var desired []dgdcreconcile.DesiredCandidate
	var names []string
	failed := 0
	for _, candidate := range snap.Candidates {
		if candidate.Outcome != OutcomeMaterialized {
			failed++
			continue
		}
		desired = append(desired, dgdcreconcile.DesiredCandidate{
			ID:         candidate.ID,
			Spec:       candidate.Manifest,
			Parameters: candidate.Parameters,
			Metrics:    candidate.Metrics,
		})
		names = append(names, CandidateName(p.RunName, candidate.ID))
	}

	current, err := p.Cluster.ListCandidates(ctx)
	if err != nil {
		return fmt.Errorf("listing candidates: %w", err)
	}
	actions, err := dgdcreconcile.ComputeActions(desired, current)
	if err != nil {
		return err
	}
	for _, candidate := range actions.Creates {
		if err := p.Cluster.CreateCandidate(ctx, CandidateName(p.RunName, candidate.ID), candidate); err != nil {
			return fmt.Errorf("creating candidate %s: %w", candidate.ID, err)
		}
	}

	status := RunStatus{
		Message:        runMessage(snap, failed),
		Round:          snap.Progress.Round,
		Evaluated:      snap.Progress.Evaluated,
		LastProgress:   progressTime(snap),
		CandidateNames: names,
	}
	if err := p.patchStatus(ctx, status); err != nil {
		return err
	}

	for _, name := range actions.Deletes {
		if err := p.Cluster.DeleteCandidate(ctx, name); err != nil {
			return fmt.Errorf("deleting candidate %s: %w", name, err)
		}
	}
	return nil
}

func runMessage(snap *Snapshot, failed int) string {
	message := snap.Run.Message
	if snap.Run.Error != "" {
		message = snap.Run.Error
	}
	if failed > 0 {
		suffix := fmt.Sprintf("%d candidate(s) failed materialization", failed)
		if message == "" {
			return suffix
		}
		return message + "; " + suffix
	}
	return message
}

// progressTime is the snapshot's own timestamp (written by the Sweeper container in the
// same pod), or the zero time when it is missing or malformed. Using the snapshot's time
// instead of the clock keeps reconciliation deterministic.
func progressTime(snap *Snapshot) time.Time {
	parsed, err := time.Parse(time.RFC3339, snap.Timestamp)
	if err != nil {
		return time.Time{}
	}
	return parsed.UTC()
}

// patchStatus skips the API call when nothing changed since the last successful patch,
// so repeated identical snapshots cause no writes.
func (p *Publisher) patchStatus(ctx context.Context, status RunStatus) error {
	if p.lastStatus != nil && reflect.DeepEqual(*p.lastStatus, status) {
		return nil
	}
	if err := p.Cluster.PatchRunStatus(ctx, status); err != nil {
		return fmt.Errorf("patching run status: %w", err)
	}
	p.lastStatus = &status
	return nil
}

// syncOnce reads the snapshot file and reconciles it if it changed. A missing file or
// an unchanged snapshot is not an error.
func (p *Publisher) syncOnce(ctx context.Context) error {
	data, err := os.ReadFile(filepath.Join(p.SnapshotDir, SnapshotFileName))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if p.lastRaw != nil && string(p.lastRaw) == string(data) {
		return nil
	}
	snap, err := ParseSnapshot(data)
	if err != nil {
		return err
	}
	if err := p.Reconcile(ctx, snap); err != nil {
		return err
	}
	// Record exactly the bytes that were reconciled, even if the file was replaced
	// since: the next sync will see the newer content as a change.
	p.lastRaw = data
	p.haveSnapshot = true
	p.terminal = snap.Run.Terminal
	p.phase = snap.Run.Phase
	return nil
}

// Run consumes snapshots until a terminal snapshot has been reconciled or the Sweeper
// container has exited. See the package comment for the exit semantics.
func (p *Publisher) Run(ctx context.Context) error {
	ticker := time.NewTicker(p.retryDelay())
	defer ticker.Stop()

	for {
		// Read the container state BEFORE the snapshot: the Sweeper's last write
		// precedes its exit, so an exit observed here means that write is visible below.
		state, stateErr := p.Cluster.SweeperState(ctx)
		if stateErr != nil {
			log.Printf("reading sweeper container state: %v", stateErr)
		}
		syncErr := p.syncOnce(ctx)
		var inputErr *dgdcreconcile.DiffInputError
		if errors.As(syncErr, &inputErr) {
			return syncErr // malformed projection; retrying cannot fix it
		}
		if syncErr != nil {
			log.Printf("sync: %v", syncErr)
		}

		if syncErr == nil && p.terminal {
			return p.terminalResult()
		}
		if stateErr == nil && state.Exited {
			return p.finish(ctx, state, syncErr)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// terminalResult is the outcome once a terminal snapshot has been reconciled.
func (p *Publisher) terminalResult() error {
	if p.phase == PhaseFailed {
		return ErrRunFailed
	}
	return nil
}

// finish handles the Sweeper having exited without a reconciled terminal snapshot. Any
// snapshot it ever wrote is now visible, so the last one is reconciled (with retries,
// since nothing will rewrite it) before deciding the outcome.
func (p *Publisher) finish(ctx context.Context, state SweeperState, syncErr error) error {
	for attempt := 1; syncErr != nil; attempt++ {
		if attempt >= finalReconcileAttempts {
			return fmt.Errorf("reconciling last snapshot: %w", syncErr)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(p.retryDelay()):
		}
		syncErr = p.syncOnce(ctx)
	}
	switch {
	case p.terminal:
		return p.terminalResult()
	case state.ExitCode != 0:
		// Crash or caught failure without a terminal snapshot: the last-known projection
		// is reconciled above; the Sweeper's own exit code fails the Job.
		return nil
	default:
		return ErrMissingTerminalSnapshot
	}
}

const finalReconcileAttempts = 3

func (p *Publisher) retryDelay() time.Duration {
	if p.PollInterval > 0 {
		return p.PollInterval
	}
	return time.Second
}
