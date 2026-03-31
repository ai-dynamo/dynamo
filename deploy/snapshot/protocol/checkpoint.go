// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

type CheckpointJobOptions struct {
	Namespace             string
	CheckpointID          string
	ArtifactVersion       string
	SeccompProfile        string
	Name                  string
	ActiveDeadlineSeconds *int64
	TTLSecondsAfterFinish *int32
	WrapLaunchJob         bool
}

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

func NewCheckpointJob(podTemplate *corev1.PodTemplateSpec, opts CheckpointJobOptions) (*batchv1.Job, error) {
	podTemplate = podTemplate.DeepCopy()
	if podTemplate.Labels == nil {
		podTemplate.Labels = map[string]string{}
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = map[string]string{}
	}
	applyCheckpointSourceMetadata(podTemplate.Labels, podTemplate.Annotations, opts.CheckpointID, opts.ArtifactVersion)
	podTemplate.Spec.RestartPolicy = corev1.RestartPolicyNever
	if opts.SeccompProfile != "" {
		EnsureLocalhostSeccompProfile(&podTemplate.Spec, opts.SeccompProfile)
	}
	if opts.WrapLaunchJob {
		if len(podTemplate.Spec.Containers) == 0 {
			return nil, fmt.Errorf("checkpoint job requires one worker container")
		}
		if len(podTemplate.Spec.Containers[0].Command) == 0 {
			return nil, fmt.Errorf("checkpoint job requires container.command when cuda-checkpoint launch-job wrapping is enabled")
		}
		podTemplate.Spec.Containers[0].Command, podTemplate.Spec.Containers[0].Args = wrapWithCudaCheckpointLaunchJob(
			podTemplate.Spec.Containers[0].Command,
			podTemplate.Spec.Containers[0].Args,
		)
	}

	return &batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: "batch/v1", Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      opts.Name,
			Namespace: opts.Namespace,
			Labels: map[string]string{
				CheckpointIDLabel: opts.CheckpointID,
			},
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds:   opts.ActiveDeadlineSeconds,
			BackoffLimit:            ptr.To[int32](0),
			TTLSecondsAfterFinished: opts.TTLSecondsAfterFinish,
			Template:                *podTemplate,
		},
	}, nil
}

func EnsureLocalhostSeccompProfile(podSpec *corev1.PodSpec, profile string) {
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: &profile,
	}
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

func wrapWithCudaCheckpointLaunchJob(command []string, args []string) ([]string, []string) {
	wrappedArgs := make([]string, 0, len(command)+len(args)+1)
	wrappedArgs = append(wrappedArgs, "--launch-job")
	wrappedArgs = append(wrappedArgs, command...)
	wrappedArgs = append(wrappedArgs, args...)
	return []string{"cuda-checkpoint"}, wrappedArgs
}
