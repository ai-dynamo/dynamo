// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
)

// MainContainerName is the conventional name of the workload container that
// checkpoint/restore operates on. The operator originated this convention
// for LWS multinode enforcement and the authoritative copy lives in
// deploy/operator/internal/consts (commonconsts.MainContainerName). This
// package keeps a local copy because deploy/snapshot is a separate Go
// module and cannot import deploy/operator/internal. Keep the two copies
// in sync.
const MainContainerName = "main"

// ResolveMainContainer returns a pointer to the workload container in podSpec.
// It prefers the container named MainContainerName and falls back to
// Containers[0] so that pods rendered before the naming convention (or by
// non-operator tooling in tests) still resolve to a sensible target.
// Returns nil only when podSpec has no containers.
func ResolveMainContainer(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil || len(podSpec.Containers) == 0 {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == MainContainerName {
			return &podSpec.Containers[i]
		}
	}
	return &podSpec.Containers[0]
}

// ResolveMainContainerName returns the name of the workload container in pod.
// Used by the snapshot agent, which reads pods from an informer cache and
// needs to send signals to the matching ContainerStatus entry by name.
// Returns "" only when pod has no containers.
func ResolveMainContainerName(pod *corev1.Pod) string {
	container := ResolveMainContainer(&pod.Spec)
	if container == nil {
		return ""
	}
	return container.Name
}

// RequireExecFormCommand returns an error if container uses a shell-wrapped
// entrypoint (e.g. /bin/sh -c "python3 ..."). Checkpoint/restore needs the
// real workload process to be PID 1 inside the container so that SIGUSR1
// from the snapshot agent (and SIGTERM from cuda-checkpoint --launch-job)
// reaches it directly; an intermediate shell swallows those signals unless
// the shell happens to tail-call exec, which only fires for simple payloads.
// Callers that build checkpoint artifacts should invoke this before emitting
// a checkpoint Job.
func RequireExecFormCommand(container *corev1.Container) error {
	if container == nil || len(container.Command) == 0 {
		return nil
	}
	base := filepath.Base(container.Command[0])
	switch base {
	case "sh", "bash", "dash", "zsh", "ash", "busybox":
	default:
		return nil
	}
	for _, token := range container.Command[1:] {
		if token == "-c" {
			return fmt.Errorf(
				"checkpoint/restore requires exec-form command on container %q; "+
					"got shell wrapper %v. Set extraPodSpec.mainContainer.command to the "+
					"actual workload entrypoint (e.g. command: [python3], "+
					"args: [-m, dynamo.vllm, --model, foo]) so the workload process is "+
					"PID 1 and receives SIGUSR1 from the snapshot agent directly",
				container.Name, container.Command,
			)
		}
	}
	return nil
}
