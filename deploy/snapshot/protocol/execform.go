// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"path/filepath"

	corev1 "k8s.io/api/core/v1"
)

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
