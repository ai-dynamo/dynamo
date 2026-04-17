// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"strings"

	batchv1 "k8s.io/api/batch/v1"
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

type CheckpointObservationPhase string

const (
	CheckpointObservationPhaseRunning                CheckpointObservationPhase = "running"
	CheckpointObservationPhaseWaitingForConfirmation CheckpointObservationPhase = "waiting_for_confirmation"
	CheckpointObservationPhaseReady                  CheckpointObservationPhase = "ready"
	CheckpointObservationPhaseFailed                 CheckpointObservationPhase = "failed"
)

type CheckpointObservation struct {
	Phase   CheckpointObservationPhase
	Reason  string
	Message string
}

func GetCheckpointJobName(checkpointID string, artifactVersion string) string {
	return "checkpoint-job-" + checkpointID + "-" + ArtifactVersion(artifactVersion)
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
		container := ResolveMainContainer(&podTemplate.Spec)
		if container == nil {
			return nil, fmt.Errorf("checkpoint job requires at least one container")
		}
		if len(container.Command) == 0 {
			return nil, fmt.Errorf("checkpoint job requires container.command when cuda-checkpoint launch-job wrapping is enabled")
		}
		container.Command, container.Args = wrapWithCudaCheckpointLaunchJob(
			container.Command,
			container.Args,
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

func ObserveCheckpointJob(job *batchv1.Job, checkpointWorkerActive bool) CheckpointObservation {
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
		observation := CheckpointObservation{
			Phase:   CheckpointObservationPhaseFailed,
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
			return CheckpointObservation{
				Phase:   CheckpointObservationPhaseReady,
				Reason:  "JobSucceeded",
				Message: "Checkpoint job completed successfully",
			}
		}
		if checkpointWorkerActive {
			return CheckpointObservation{Phase: CheckpointObservationPhaseWaitingForConfirmation}
		}
		return CheckpointObservation{
			Phase:   CheckpointObservationPhaseFailed,
			Reason:  "CheckpointVerificationFailed",
			Message: "Checkpoint job completed without snapshot-agent completion confirmation",
		}
	}

	if jobFailed {
		return CheckpointObservation{
			Phase:   CheckpointObservationPhaseFailed,
			Reason:  "JobFailed",
			Message: "Checkpoint job failed",
		}
	}

	return CheckpointObservation{Phase: CheckpointObservationPhaseRunning}
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

func wrapWithCudaCheckpointLaunchJob(command []string, args []string) ([]string, []string) {
	// Unwrap "/bin/sh -c <single-string>" so cuda-checkpoint launches the
	// actual process directly. Otherwise sh sits between cuda-checkpoint and
	// the real process and swallows SIGUSR1.
	if len(command) >= 2 && command[len(command)-1] == "-c" && len(args) == 1 {
		shell := command[:len(command)-1] // e.g. ["/bin/sh"] — discarded
		_ = shell
		parts := strings.Fields(args[0])
		command = parts[:1] // e.g. ["python3"]
		args = parts[1:]    // e.g. ["-m", "dynamo.vllm", "--model", ...]
	}

	wrappedArgs := make([]string, 0, len(command)+len(args)+1)
	wrappedArgs = append(wrappedArgs, "--launch-job")
	wrappedArgs = append(wrappedArgs, command...)
	wrappedArgs = append(wrappedArgs, args...)
	return []string{"cuda-checkpoint"}, wrappedArgs
}
