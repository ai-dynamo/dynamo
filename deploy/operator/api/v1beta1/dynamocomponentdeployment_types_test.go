// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestComputeDynamoNamespace(t *testing.T) {
	// Reported namespace and DGD names: 36 + 1 + 26 = 63 bytes deploys,
	// 36 + 1 + 27 = 64 bytes was rejected by the Grove admission webhook.
	const reportNamespace = "dgdr16-ab-length-20261013-abcdefghij"
	const name63 = "gaaaaaaaaaaaaaaaaaaaaaaaaa"
	const name64 = "gaaaaaaaaaaaaaaaaaaaaaaaaaa"
	// Same first 54 bytes of the joined value as name64, different tail.
	const name64Sibling = "gaaaaaaaaaaaaaaaaaaaaaaaaab"
	long63Namespace := strings.Repeat("n", 63)

	tests := []struct {
		name         string
		global       bool
		k8sNamespace string
		dgdName      string
		// want is the exact expected value; empty means the value must be shortened.
		want string
	}{
		{
			name:         "short value is unchanged",
			k8sNamespace: "my-namespace",
			dgdName:      "my-deployment",
			want:         "my-namespace-my-deployment",
		},
		{
			name:         "dots in DGD name are replaced",
			k8sNamespace: "ns",
			dgdName:      "qwen3-0.6b",
			want:         "ns-qwen3-0-6b",
		},
		{
			name:         "63-byte value is unchanged",
			k8sNamespace: reportNamespace,
			dgdName:      name63,
			want:         reportNamespace + "-" + name63,
		},
		{
			name:         "64-byte value is shortened",
			k8sNamespace: reportNamespace,
			dgdName:      name64,
		},
		{
			name:         "63-byte namespace with long DGD name is shortened",
			k8sNamespace: long63Namespace,
			dgdName:      "a-long-deployment-name-that-is-still-valid",
		},
		{
			name:         "global namespace",
			global:       true,
			k8sNamespace: reportNamespace,
			dgdName:      name64,
			want:         commonconsts.GlobalDynamoNamespace,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeDynamoNamespace(tt.global, tt.k8sNamespace, tt.dgdName)
			if errs := validation.IsValidLabelValue(got); len(errs) > 0 {
				t.Fatalf("ComputeDynamoNamespace() = %q (%d bytes) is not a valid label value: %v", got, len(got), errs)
			}
			if again := ComputeDynamoNamespace(tt.global, tt.k8sNamespace, tt.dgdName); again != got {
				t.Fatalf("ComputeDynamoNamespace() not deterministic: %q then %q", got, again)
			}
			if tt.want != "" {
				if got != tt.want {
					t.Fatalf("ComputeDynamoNamespace() = %q, want %q", got, tt.want)
				}
				return
			}
			joined := tt.k8sNamespace + "-" + tt.dgdName
			if len(joined) <= 63 {
				t.Fatalf("test input %q must be longer than 63 bytes", joined)
			}
			if got == joined {
				t.Fatalf("ComputeDynamoNamespace() = %q, want a shortened value", got)
			}
		})
	}

	t.Run("shortened values stay distinct for a shared prefix", func(t *testing.T) {
		a := ComputeDynamoNamespace(false, reportNamespace, name64)
		b := ComputeDynamoNamespace(false, reportNamespace, name64Sibling)
		if a == b {
			t.Fatalf("ComputeDynamoNamespace() returned %q for both %q and %q", a, name64, name64Sibling)
		}
	})
}
