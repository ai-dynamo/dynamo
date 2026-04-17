// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
)

func TestRequireExecFormCommand(t *testing.T) {
	for _, tc := range []struct {
		name    string
		cmd     []string
		wantErr bool
	}{
		{"nil command", nil, false},
		{"empty command", []string{}, false},
		{"exec form python", []string{"python3", "-m", "dynamo.vllm"}, false},
		{"exec form /usr/bin/python3", []string{"/usr/bin/python3"}, false},
		{"shell without -c is fine", []string{"/bin/sh", "my-entrypoint.sh"}, false},
		{"bash without -c is fine", []string{"bash", "-l"}, false},
		{"/bin/sh -c rejected", []string{"/bin/sh", "-c"}, true},
		{"sh -c rejected", []string{"sh", "-c"}, true},
		{"bash -c rejected", []string{"/bin/bash", "-c"}, true},
		{"dash -c rejected", []string{"dash", "-c"}, true},
		{"busybox sh -c rejected", []string{"/bin/busybox", "sh", "-c"}, true},
		{"custom binary passes", []string{"/opt/my-runner", "-c", "config.yaml"}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := RequireExecFormCommand(&corev1.Container{Name: "main", Command: tc.cmd})
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}

	if err := RequireExecFormCommand(nil); err != nil {
		t.Fatalf("nil container should return nil error, got %v", err)
	}
}
