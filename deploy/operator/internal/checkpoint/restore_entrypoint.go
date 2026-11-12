/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"path"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// HasSupportedRestoreEntrypoint reports whether the container directly runs
// `python -m <module>`. Enabling checkpointing makes the workload author
// responsible for a module that calls maybe_run_restore_standby_mode(), so the
// module name is not restricted. Shell and other non-python launchers are
// rejected because admission cannot prove they honor SNAPSHOT_RESTORE_STANDBY.
// The container must be non-nil.
func HasSupportedRestoreEntrypoint(container *corev1.Container) bool {
	if len(container.Command) == 0 {
		return false
	}
	python := path.Base(container.Command[0])
	if python != "python" && python != "python3" && !strings.HasPrefix(python, "python3.") {
		return false
	}

	arguments := make([]string, 0, len(container.Command)+len(container.Args))
	arguments = append(arguments, container.Command...)
	arguments = append(arguments, container.Args...)

	// Skip only operand-free interpreter options so -m remains unambiguous.
	moduleFlagIndex := 1
	for moduleFlagIndex < len(arguments) && isOperandFreePythonInterpreterFlag(arguments[moduleFlagIndex]) {
		moduleFlagIndex++
	}
	if moduleFlagIndex+1 >= len(arguments) || arguments[moduleFlagIndex] != "-m" {
		return false
	}

	// An empty or option-like token is not a module name and fails at startup.
	module := arguments[moduleFlagIndex+1]
	return module != "" && !strings.HasPrefix(module, "-")
}

// isOperandFreePythonInterpreterFlag recognizes options that cannot consume
// the following -m argument. Operand-taking and execution-selector options
// remain fail-closed.
func isOperandFreePythonInterpreterFlag(argument string) bool {
	switch argument {
	case "-b", "-bb", "-B", "-d", "-E", "-i", "-I", "-O", "-OO", "-P", "-q", "-s", "-S", "-u", "-v", "-vv", "-x":
		return true
	default:
		return false
	}
}
