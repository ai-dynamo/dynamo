// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpointjob

import (
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
)

type ObservationPhase string

const (
	ObservationPhaseRunning                ObservationPhase = "running"
	ObservationPhaseWaitingForConfirmation ObservationPhase = "waiting_for_confirmation"
	ObservationPhaseReady                  ObservationPhase = "ready"
	ObservationPhaseFailed                 ObservationPhase = "failed"
)

type Observation struct {
	Phase   ObservationPhase
	Reason  string
	Message string
}

func Observe(job *batchv1.Job, checkpointWorkerActive bool) Observation {
	jobComplete := false
	jobFailed := false
	for _, condition := range job.Status.Conditions {
		if condition.Status != corev1.ConditionTrue {
			continue
		}
		if condition.Type == batchv1.JobComplete {
			jobComplete = true
			continue
		}
		if condition.Type == batchv1.JobFailed {
			jobFailed = true
		}
	}

	status := job.Annotations[snapshotprotocol.CheckpointStatusAnnotation]
	if status == snapshotprotocol.CheckpointStatusFailed {
		observation := Observation{
			Phase:   ObservationPhaseFailed,
			Reason:  "JobFailed",
			Message: "Checkpoint job failed",
		}
		if jobComplete {
			observation.Reason = "CheckpointVerificationFailed"
			observation.Message = "Checkpoint job completed but snapshot-agent reported checkpoint failure"
		}
		return observation
	}

	if jobComplete {
		if status == snapshotprotocol.CheckpointStatusCompleted {
			return Observation{
				Phase:   ObservationPhaseReady,
				Reason:  "JobSucceeded",
				Message: "Checkpoint job completed successfully",
			}
		}
		if checkpointWorkerActive {
			return Observation{Phase: ObservationPhaseWaitingForConfirmation}
		}
		return Observation{
			Phase:   ObservationPhaseFailed,
			Reason:  "CheckpointVerificationFailed",
			Message: "Checkpoint job completed without snapshot-agent completion confirmation",
		}
	}

	if jobFailed {
		return Observation{
			Phase:   ObservationPhaseFailed,
			Reason:  "JobFailed",
			Message: "Checkpoint job failed",
		}
	}

	return Observation{Phase: ObservationPhaseRunning}
}
