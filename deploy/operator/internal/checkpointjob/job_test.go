// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpointjob

import (
	"testing"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestDesiredCheckpointJobName(t *testing.T) {
	name := DesiredCheckpointJobName("abc123def4567890", map[string]string{
		snapshotprotocol.CheckpointArtifactVersionAnnotation: "2",
	})
	if name != "checkpoint-job-abc123def4567890-2" {
		t.Fatalf("unexpected checkpoint job name: %s", name)
	}

	defaultName := DesiredCheckpointJobName("abc123def4567890", nil)
	if defaultName != "checkpoint-job-abc123def4567890-"+snapshotprotocol.DefaultCheckpointArtifactVersion {
		t.Fatalf("unexpected default checkpoint job name: %s", defaultName)
	}
}
