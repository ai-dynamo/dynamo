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
	"fmt"

	"sigs.k8s.io/yaml"
)

// SnapshotSchemaVersion is the only wire version this publisher understands.
const SnapshotSchemaVersion = 1

// Candidate outcomes carried by the snapshot.
const (
	OutcomeMaterialized          = "materialized"
	OutcomeMaterializationFailed = "materialization_failed"
)

// Run phases carried by the snapshot.
const (
	PhaseRunning   = "Running"
	PhaseSucceeded = "Succeeded"
	PhaseFailed    = "Failed"
)

// Snapshot is the complete current projection of one run, written atomically by the
// Sweeper container (see dynamo.aisimulate.output.dgdr_run.snapshot). It is desired
// state, not a command list: each snapshot supersedes the previous one entirely.
type Snapshot struct {
	SchemaVersion int                 `json:"schemaVersion"`
	Timestamp     string              `json:"timestamp,omitempty"`
	Run           SnapshotRun         `json:"run"`
	Progress      SnapshotProgress    `json:"progress"`
	Candidates    []SnapshotCandidate `json:"candidates"`
}

type SnapshotRun struct {
	Phase    string `json:"phase"`
	Terminal bool   `json:"terminal"`
	Message  string `json:"message,omitempty"`
	Error    string `json:"error,omitempty"`
}

type SnapshotProgress struct {
	Round     int32 `json:"round"`
	Evaluated int32 `json:"evaluated"`
}

// SnapshotCandidate is one evaluated point. Candidates are ordered best-first for
// scalar searches; the id never depends on that order.
type SnapshotCandidate struct {
	ID         string         `json:"id"`
	Outcome    string         `json:"outcome"`
	Parameters map[string]any `json:"parameters,omitempty"`
	Metrics    map[string]any `json:"metrics,omitempty"`
	Manifest   string         `json:"manifest,omitempty"`
	Error      string         `json:"error,omitempty"`
}

func ParseSnapshot(data []byte) (*Snapshot, error) {
	var snap Snapshot
	if err := yaml.Unmarshal(data, &snap); err != nil {
		return nil, fmt.Errorf("decoding snapshot: %w", err)
	}
	if snap.SchemaVersion != SnapshotSchemaVersion {
		return nil, fmt.Errorf("unsupported snapshot schemaVersion %d (want %d)", snap.SchemaVersion, SnapshotSchemaVersion)
	}
	switch snap.Run.Phase {
	case PhaseRunning, PhaseSucceeded, PhaseFailed:
	default:
		return nil, fmt.Errorf("unknown run phase %q", snap.Run.Phase)
	}
	if snap.Run.Terminal != (snap.Run.Phase != PhaseRunning) {
		return nil, fmt.Errorf("run phase %q inconsistent with terminal=%t", snap.Run.Phase, snap.Run.Terminal)
	}
	seen := make(map[string]struct{}, len(snap.Candidates))
	for i, candidate := range snap.Candidates {
		if candidate.ID == "" {
			return nil, fmt.Errorf("candidate %d has an empty id", i)
		}
		if _, dup := seen[candidate.ID]; dup {
			return nil, fmt.Errorf("duplicate candidate id %q", candidate.ID)
		}
		seen[candidate.ID] = struct{}{}
		switch candidate.Outcome {
		case OutcomeMaterialized:
			if candidate.Manifest == "" {
				return nil, fmt.Errorf("materialized candidate %q has no manifest", candidate.ID)
			}
		case OutcomeMaterializationFailed:
			if candidate.Error == "" {
				return nil, fmt.Errorf("failed candidate %q has no error", candidate.ID)
			}
		default:
			return nil, fmt.Errorf("candidate %q has unknown outcome %q", candidate.ID, candidate.Outcome)
		}
	}
	return &snap, nil
}
