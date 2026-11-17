/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package gpu

import (
	"context"
	"errors"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDiscovererWithoutReader(t *testing.T) {
	d := &Discoverer{}
	if _, err := d.Discover(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "APIReader is not configured") {
		t.Fatalf("got %v", err)
	}
}

func TestDiscovererWithNoBackend(t *testing.T) {
	d := &Discoverer{Reader: fake.NewClientBuilder().Build()}
	if _, err := d.Discover(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "node-label discovery is disabled") {
		t.Fatalf("got %v", err)
	}
}

func TestDiscovererNodeLabelsFindNothing(t *testing.T) {
	d := &Discoverer{Reader: fake.NewClientBuilder().Build(), NodeLabels: func() bool { return true }}
	if _, err := d.Discover(context.Background(), ""); err == nil || !strings.Contains(err.Error(), "auto-discovery failed") {
		t.Fatalf("an empty cluster must be an error, got %v", err)
	}
}

func TestKnownSKU(t *testing.T) {
	cases := []struct {
		name string
		info *GPUInfo
		want nvidiacomv1beta1.GPUSKUType
	}{
		{"discovered system wins", &GPUInfo{System: "h100_sxm", Model: "NVIDIA H200"}, "h100_sxm"},
		{"inferred from the model", &GPUInfo{Model: "NVIDIA H200"}, "h200_sxm"},
		{"unknown model is not passed through", &GPUInfo{Model: "NVIDIA RTX 6000 Ada"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := KnownSKU(tc.info); got != tc.want {
				t.Fatalf("got %q want %q", got, tc.want)
			}
		})
	}
}

func TestDiscoveryFailureReason(t *testing.T) {
	cases := map[string]string{
		"list pods: forbidden":             "failed to list DCGM exporter pods (RBAC/cluster connectivity issue)",
		"GPU Operator is not installed":    "GPU Operator not installed in expected namespace",
		"http get failed":                  "failed to reach DCGM metrics endpoint on pod (network/port issue)",
		"metrics endpoint returned status": "DCGM pod metrics endpoint returned non-200 status",
		"no GPUs detected":                 "no GPUs detected in dcgm metrics (GPU model or metrics missing)",
		"something else":                   "unknown",
	}
	for msg, want := range cases {
		if got := DiscoveryFailureReason(errors.New(msg)); got != want {
			t.Errorf("%q: got %q, want %q", msg, got, want)
		}
	}
	if got := DiscoveryFailureReason(nil); got != "unknown" {
		t.Errorf("nil: got %q", got)
	}
}
