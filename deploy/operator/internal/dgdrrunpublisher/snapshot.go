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
	"errors"
	"fmt"
	"regexp"

	"sigs.k8s.io/yaml"
)

// ErrProtocolViolation means the Sweeper broke the snapshot contract: a snapshot that
// cannot be decoded or validated, one that moves progress backwards, or a missing
// terminal snapshot. It maps to ExitProtocolViolation.
var ErrProtocolViolation = errors.New("snapshot protocol violation")

func violation(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrProtocolViolation, fmt.Sprintf(format, args...))
}

// maxCandidateIDLength keeps ids well inside the 63-character Kubernetes label-value
// limit, since the id is recorded verbatim as a label on the candidate.
const maxCandidateIDLength = 48

// candidateIDPattern is a DNS-1123 label, valid both as a label value and inside an
// object name.
var candidateIDPattern = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`)

func validateManifest(id, manifest string) error {
	var doc map[string]any
	if err := yaml.Unmarshal([]byte(manifest), &doc); err != nil {
		return violation("manifest of candidate %q is not a YAML mapping: %v", id, err)
	}
	if doc["kind"] != "DynamoGraphDeployment" {
		return violation("manifest of candidate %q is not a DynamoGraphDeployment", id)
	}
	if _, ok := doc["spec"].(map[string]any); !ok {
		return violation("manifest of candidate %q has no spec", id)
	}
	return nil
}

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
		return nil, violation("decoding snapshot: %v", err)
	}
	if snap.SchemaVersion != SnapshotSchemaVersion {
		return nil, violation("unsupported snapshot schemaVersion %d (want %d)", snap.SchemaVersion, SnapshotSchemaVersion)
	}
	switch snap.Run.Phase {
	case PhaseRunning, PhaseSucceeded, PhaseFailed:
	default:
		return nil, violation("unknown run phase %q", snap.Run.Phase)
	}
	if snap.Run.Terminal != (snap.Run.Phase != PhaseRunning) {
		return nil, violation("run phase %q inconsistent with terminal=%t", snap.Run.Phase, snap.Run.Terminal)
	}
	seen := make(map[string]struct{}, len(snap.Candidates))
	for i, candidate := range snap.Candidates {
		if candidate.ID == "" {
			return nil, violation("candidate %d has an empty id", i)
		}
		if len(candidate.ID) > maxCandidateIDLength || !candidateIDPattern.MatchString(candidate.ID) {
			return nil, violation("candidate id %q is not a lowercase DNS label of at most %d characters", candidate.ID, maxCandidateIDLength)
		}
		if _, dup := seen[candidate.ID]; dup {
			return nil, violation("duplicate candidate id %q", candidate.ID)
		}
		seen[candidate.ID] = struct{}{}
		switch candidate.Outcome {
		case OutcomeMaterialized:
			if candidate.Manifest == "" {
				return nil, violation("materialized candidate %q has no manifest", candidate.ID)
			}
			if err := validateManifest(candidate.ID, candidate.Manifest); err != nil {
				return nil, err
			}
		case OutcomeMaterializationFailed:
			if candidate.Error == "" {
				return nil, violation("failed candidate %q has no error", candidate.ID)
			}
		default:
			return nil, violation("candidate %q has unknown outcome %q", candidate.ID, candidate.Outcome)
		}
	}
	return &snap, nil
}
