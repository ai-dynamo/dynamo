/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package discovery

import (
	"strings"
	"testing"
)

func TestGetK8sDiscoveryLabelValue(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		input        string
		want         string
		wantSame     bool
		wantMaxLen   int
		wantHashPart bool
	}{
		"short value unchanged": {
			input:      "short-name",
			wantSame:   true,
			wantMaxLen: maxLabelValueLen,
		},
		"boundary value unchanged": {
			input:      strings.Repeat("a", maxLabelValueLen),
			wantSame:   true,
			wantMaxLen: maxLabelValueLen,
		},
		"over limit truncated with hash": {
			// sha256(64 * 'b') starts with a0fab137; expected value is
			// 54 'b' chars + '-' + first 8 hex chars of that digest.
			input:        strings.Repeat("b", maxLabelValueLen+1),
			want:         strings.Repeat("b", maxLabelValueLen-hashLen-1) + "-a0fab137",
			wantSame:     false,
			wantMaxLen:   maxLabelValueLen,
			wantHashPart: true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			got := getK8sDiscoveryLabelValue(tc.input)

			if tc.want != "" && got != tc.want {
				t.Fatalf("getK8sDiscoveryLabelValue(%q) = %q, want %q", tc.input, got, tc.want)
			}
			if tc.wantSame && got != tc.input {
				t.Fatalf("getK8sDiscoveryLabelValue(%q) = %q, want unchanged", tc.input, got)
			}
			if !tc.wantSame && got == tc.input {
				t.Fatalf("getK8sDiscoveryLabelValue(%q) should change for over-limit values", tc.input)
			}
			if len(got) > tc.wantMaxLen {
				t.Fatalf("len(%q) = %d, want <= %d", got, len(got), tc.wantMaxLen)
			}
			if tc.wantHashPart && !strings.Contains(got, "-") {
				t.Fatalf("getK8sDiscoveryLabelValue(%q) = %q, want hash separator", tc.input, got)
			}
			if again := getK8sDiscoveryLabelValue(tc.input); again != got {
				t.Fatalf("getK8sDiscoveryLabelValue(%q) not deterministic: first=%q second=%q", tc.input, got, again)
			}
		})
	}
}

func TestGetK8sDiscoveryResourcesUseCappedLabel(t *testing.T) {
	t.Parallel()

	dgdName := strings.Repeat("a", 80)
	namespace := "default"

	sa := GetK8sDiscoveryServiceAccount(dgdName, namespace)
	role := GetK8sDiscoveryRole(dgdName, namespace)
	rb := GetK8sDiscoveryRoleBinding(dgdName, namespace)

	for _, label := range []string{
		sa.Labels["app.kubernetes.io/name"],
		role.Labels["app.kubernetes.io/name"],
		rb.Labels["app.kubernetes.io/name"],
	} {
		if len(label) > maxLabelValueLen {
			t.Fatalf("label %q length=%d, want <= %d", label, len(label), maxLabelValueLen)
		}
	}
}
