// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"time"

	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
)

type CheckpointJobPhase string

const (
	CheckpointJobPhaseRunning         CheckpointJobPhase = "running"
	CheckpointJobPhaseWaitingForLease CheckpointJobPhase = "waiting_for_lease"
	CheckpointJobPhaseReady           CheckpointJobPhase = "ready"
	CheckpointJobPhaseFailed          CheckpointJobPhase = "failed"
)

type CheckpointJobObservation struct {
	Phase   CheckpointJobPhase
	Reason  string
	Message string
}

func LeaseExpired(lease *coordinationv1.Lease, now time.Time) bool {
	if lease == nil || lease.Spec.LeaseDurationSeconds == nil {
		return true
	}
	last := lease.Spec.RenewTime
	if last == nil {
		last = lease.Spec.AcquireTime
	}
	if last == nil {
		return true
	}
	return now.After(last.Time.Add(time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second))
}

func ObserveCheckpointJob(job *batchv1.Job, lease *coordinationv1.Lease, now time.Time) CheckpointJobObservation {
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

	status := job.Annotations[CheckpointStatusAnnotation]
	if status == CheckpointStatusFailed {
		observation := CheckpointJobObservation{
			Phase:   CheckpointJobPhaseFailed,
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
		if status == CheckpointStatusCompleted {
			return CheckpointJobObservation{
				Phase:   CheckpointJobPhaseReady,
				Reason:  "JobSucceeded",
				Message: "Checkpoint job completed successfully",
			}
		}
		if lease != nil && !LeaseExpired(lease, now) {
			return CheckpointJobObservation{Phase: CheckpointJobPhaseWaitingForLease}
		}
		return CheckpointJobObservation{
			Phase:   CheckpointJobPhaseFailed,
			Reason:  "CheckpointVerificationFailed",
			Message: "Checkpoint job completed without snapshot-agent completion confirmation",
		}
	}

	if jobFailed {
		return CheckpointJobObservation{
			Phase:   CheckpointJobPhaseFailed,
			Reason:  "JobFailed",
			Message: "Checkpoint job failed",
		}
	}

	return CheckpointJobObservation{Phase: CheckpointJobPhaseRunning}
}
