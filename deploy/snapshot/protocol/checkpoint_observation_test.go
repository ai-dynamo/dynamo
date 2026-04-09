// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"testing"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestObserveCheckpointJob(t *testing.T) {
	makeJob := func(annotation string, conditions ...batchv1.JobCondition) *batchv1.Job {
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{},
			},
			Status: batchv1.JobStatus{
				Conditions: conditions,
			},
		}
		if annotation != "" {
			job.Annotations[CheckpointStatusAnnotation] = annotation
		}
		return job
	}

	tests := []struct {
		name                   string
		job                    *batchv1.Job
		checkpointWorkerActive bool
		wantPhase              CheckpointObservationPhase
		wantReason             string
		wantMessage            string
	}{
		{
			name:      "running job stays running",
			job:       makeJob(""),
			wantPhase: CheckpointObservationPhaseRunning,
		},
		{
			name: "completed job with completion annotation is ready",
			job: makeJob(
				CheckpointStatusCompleted,
				batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
			),
			wantPhase:   CheckpointObservationPhaseReady,
			wantReason:  "JobSucceeded",
			wantMessage: "Checkpoint job completed successfully",
		},
		{
			name: "completed job waits for terminal confirmation while worker is active",
			job: makeJob(
				"",
				batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
			),
			checkpointWorkerActive: true,
			wantPhase:              CheckpointObservationPhaseWaitingForConfirmation,
		},
		{
			name: "completed job fails without confirmation once worker is inactive",
			job: makeJob(
				"",
				batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
			),
			wantPhase:   CheckpointObservationPhaseFailed,
			wantReason:  "CheckpointVerificationFailed",
			wantMessage: "Checkpoint job completed without snapshot-agent completion confirmation",
		},
		{
			name: "failed checkpoint annotation wins over completed job",
			job: makeJob(
				CheckpointStatusFailed,
				batchv1.JobCondition{Type: batchv1.JobComplete, Status: corev1.ConditionTrue},
			),
			checkpointWorkerActive: true,
			wantPhase:              CheckpointObservationPhaseFailed,
			wantReason:             "CheckpointVerificationFailed",
			wantMessage:            "Checkpoint job completed but snapshot-agent reported checkpoint failure",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			observation := ObserveCheckpointJob(tc.job, tc.checkpointWorkerActive)
			if observation.Phase != tc.wantPhase {
				t.Fatalf("phase = %q, want %q", observation.Phase, tc.wantPhase)
			}
			if observation.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", observation.Reason, tc.wantReason)
			}
			if observation.Message != tc.wantMessage {
				t.Fatalf("message = %q, want %q", observation.Message, tc.wantMessage)
			}
		})
	}
}
