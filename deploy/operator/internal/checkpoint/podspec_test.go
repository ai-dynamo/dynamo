/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"maps"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
)

func TestApplyRestoreCandidateMetadataAutomaticCaptureIsStable(t *testing.T) {
	t.Log("Given pending and Ready states for the same automatic SnapshotJob candidate")
	job := &SnapshotJobReference{Name: "checkpoint-worker", UID: types.UID("job-uid")}
	pending := &CheckpointInfo{
		Enabled:              true,
		AutomaticCapture:     true,
		StartupPolicy:        nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
		AutomaticSnapshotJob: job,
	}
	ready := &CheckpointInfo{
		Enabled:              true,
		Exists:               true,
		Ready:                true,
		AutomaticCapture:     true,
		CheckpointName:       "worker-snapshot",
		StartupPolicy:        nvidiacomv1alpha1.CheckpointStartupPolicyImmediate,
		AutomaticSnapshotJob: job,
		NativeSnapshot: &ResolvedPodSnapshot{
			UID:                  types.UID("snapshot-uid"),
			BoundContentName:     "content-a",
			SourceContainer:      consts.MainContainerName,
			CompatibilityVersion: consts.SnapshotCompatibilityVersion,
			GMSMode:              consts.SnapshotGMSModeDisabled,
		},
	}

	pendingAnnotations := map[string]string{"user.example/preserved": "true"}
	readyAnnotations := maps.Clone(pendingAnnotations)

	t.Log("When Dynamo applies its private restore-candidate metadata to both states")
	require.NoError(t, ApplyRestoreCandidateMetadata(pendingAnnotations, pending))
	require.NoError(t, ApplyRestoreCandidateMetadata(readyAnnotations, ready))

	t.Log("Then capture readiness does not change the rendered annotations")
	assert.Equal(t, pendingAnnotations, readyAnnotations)
	assert.Equal(t, consts.RestoreCandidateSourceSnapshotJob,
		pendingAnnotations[consts.RestoreCandidateSourceKindAnnotation])
	assert.Equal(t, job.Name, pendingAnnotations[consts.CheckpointNameAnnotation])
	assert.Equal(t, string(job.UID), pendingAnnotations[consts.SnapshotJobCandidateUIDAnnotation])
	assert.NotContains(t, pendingAnnotations, consts.SnapshotCandidateUIDAnnotation)
}

func TestAutomaticSnapshotJobReferenceFromAnnotations(t *testing.T) {
	t.Log("Given a complete automatic SnapshotJob candidate annotation set")
	annotations := map[string]string{
		consts.RestoreCandidateSourceKindAnnotation: consts.RestoreCandidateSourceSnapshotJob,
		consts.CheckpointNameAnnotation:             "checkpoint-worker",
		consts.SnapshotJobCandidateUIDAnnotation:    "job-uid",
	}

	t.Log("When Dynamo parses the immutable SnapshotJob reference")
	reference, found, err := AutomaticSnapshotJobReferenceFromAnnotations(annotations)

	t.Log("Then the name and UID are preserved")
	require.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, &SnapshotJobReference{Name: "checkpoint-worker", UID: types.UID("job-uid")}, reference)
}
