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
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
)

// SnapshotFileName is the file the Sweeper container writes.
const SnapshotFileName = "dgdr_run_snapshot.yaml"

// maxObjectName is the Kubernetes limit on an object name (DNS subdomain).
const maxObjectName = 253

// RunStatus is the part of DynamoGraphDeploymentRun.status the publisher owns: the
// ordered candidate references plus progress. CandidateNames is in rank order (best
// first for scalar searches). LastProgress is the zero time when the snapshot carries
// no usable timestamp, in which case the existing value is left untouched.
type RunStatus struct {
	Message        string
	HasProgress    bool
	Round          int32
	Evaluated      int32
	LastProgress   time.Time
	CandidateNames []string
}

// DesiredState is what one snapshot says should exist in the cluster.
type DesiredState struct {
	// Candidates are the materialized candidates, in snapshot (rank) order.
	Candidates []dgdcreconcile.DesiredCandidate
	// Names maps a candidate id to its DGDC name.
	Names    map[string]string
	Status   RunStatus
	Terminal bool
	Phase    string
}

// CandidateName is the DGDC name for a candidate id. It depends only on the run
// incarnation (name and UID) and the stable id, never on rank. The UID component keeps a
// recreated run of the same name from colliding with candidates of the previous one that
// are still waiting for garbage collection. A run name too long to leave room for the
// rest is truncated and a hash of the full run name is added, so names stay unique and
// within the limit.
func CandidateName(runName, runUID, id string) string {
	uidSum := sha256.Sum256([]byte(runUID))
	suffix := "-" + hex.EncodeToString(uidSum[:4]) + "-" + id
	if len(runName)+len(suffix) <= maxObjectName {
		return runName + suffix
	}
	nameSum := sha256.Sum256([]byte(runName))
	hash := hex.EncodeToString(nameSum[:6])
	room := maxObjectName - len(suffix) - len(hash) - 1
	return strings.TrimRight(runName[:room], "-.") + "-" + hash + suffix
}

// Desire projects a snapshot onto the desired cluster state of one run incarnation.
// Candidates whose materialization failed are not created; they are only counted in the
// run message.
func Desire(runName, runUID string, snap *Snapshot) DesiredState {
	state := DesiredState{
		Names:    make(map[string]string, len(snap.Candidates)),
		Terminal: snap.Run.Terminal,
		Phase:    snap.Run.Phase,
	}
	failed := 0
	var names []string
	for _, candidate := range snap.Candidates {
		if candidate.Outcome != OutcomeMaterialized {
			failed++
			continue
		}
		name := CandidateName(runName, runUID, candidate.ID)
		state.Names[candidate.ID] = name
		names = append(names, name)
		state.Candidates = append(state.Candidates, dgdcreconcile.DesiredCandidate{
			ID:         candidate.ID,
			Spec:       candidate.Manifest,
			Parameters: candidate.Parameters,
			Metrics:    candidate.Metrics,
		})
	}
	state.Status = RunStatus{
		Message:        runMessage(snap, failed),
		HasProgress:    true,
		Round:          snap.Progress.Round,
		Evaluated:      snap.Progress.Evaluated,
		LastProgress:   progressTime(snap),
		CandidateNames: names,
	}
	return state
}

func runMessage(snap *Snapshot, failed int) string {
	message := snap.Run.Message
	if snap.Run.Error != "" {
		message = snap.Run.Error
	}
	if failed > 0 {
		suffix := fmtFailed(failed)
		if message == "" {
			return suffix
		}
		return message + "; " + suffix
	}
	return message
}

func fmtFailed(failed int) string {
	return strconv.Itoa(failed) + " candidate(s) failed materialization"
}

// progressTime is the snapshot's own timestamp (written by the Sweeper container in the
// same pod) truncated to the second the API stores, or the zero time when it is missing
// or malformed. Using the snapshot's time instead of the clock keeps reconciliation
// deterministic.
func progressTime(snap *Snapshot) time.Time {
	parsed, err := time.Parse(time.RFC3339, snap.Timestamp)
	if err != nil {
		return time.Time{}
	}
	return parsed.UTC().Truncate(time.Second)
}

// NeedsUpdate reports whether the observed run status differs from the desired one, so
// the status is only written when something changed. A zero desired LastProgress means
// "leave the observed value alone".
func (desired RunStatus) NeedsUpdate(observed RunStatus) bool {
	if desired.Message != observed.Message || desired.HasProgress != observed.HasProgress {
		return true
	}
	if desired.HasProgress && (desired.Round != observed.Round || desired.Evaluated != observed.Evaluated) {
		return true
	}
	if !desired.LastProgress.IsZero() && !desired.LastProgress.Equal(observed.LastProgress) {
		return true
	}
	return !slices.Equal(desired.CandidateNames, observed.CandidateNames)
}
