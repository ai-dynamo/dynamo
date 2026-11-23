/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package api

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestComputeDynamoNamespace(t *testing.T) {
	tests := []struct {
		name      string
		namespace string
		dgdName   string
		global    bool
		want      string
	}{
		{name: "short", namespace: "default", dgdName: "graph", global: false, want: "default-graph"},
		{name: "dotted name", namespace: "default", dgdName: "qwen3-0.6b", global: false, want: "default-qwen3-0-6b"},
		{name: "global with long names", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg", global: true, want: "dynamo"},
		{name: "derived length 63", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gaaaaaaaaaaaaaaaaaaaaaaaaa", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-gaaaaaaaaaaaaaaaaaaaaaaaaa"},
		{name: "derived length 64", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gaaaaaaaaaaaaaaaaaaaaaaaaaa", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-gaaaaaaaa-27479653da05e5b6"},
		{name: "derived length 65", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gaaaaaaaaaaaaaaaaaaaaaaaaaaa", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-gaaaaaaaa-7a36c9d0a158ef97"},
		{name: "maximum resource lengths", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "ggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-157805ad40ce537b"},
		{name: "prefix ends at separator", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gggggggggggggggggggggggggggggg", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-06d6fb254bb1ce6f"},
		{name: "shared prefix v1", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "ggggggggggggggggggggggggggggggggggg-v1", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-ggggggggg-1e3408b7c8511096"},
		{name: "shared prefix v2", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "ggggggggggggggggggggggggggggggggggg-v2", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-ggggggggg-a79bf6d68f7c92a9"},
		{name: "long dotted name", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gggggggggggggggggggggggggggggg.v1", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-ggggggggg-f10edfd42a3d1379"},
		{name: "long hyphenated name", namespace: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn", dgdName: "gggggggggggggggggggggggggggggg-v1", global: false, want: "nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn-ggggggggg-5e504c90d7339d87"},
		{name: "ambiguous pair one", namespace: "a-b", dgdName: "cggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg", global: false, want: "a-b-cggggggggggggggggggggggggggggggggggggggggg-32d278bf93ca9881"},
		{name: "ambiguous pair two", namespace: "a", dgdName: "b-cggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg", global: false, want: "a-b-cggggggggggggggggggggggggggggggggggggggggg-c611648e5ddf3f08"},
	}
	implementations := []struct {
		name    string
		compute func(bool, string, string) string
	}{
		{name: "v1alpha1", compute: v1alpha1.ComputeDynamoNamespace},
		{name: "v1beta1", compute: v1beta1.ComputeDynamoNamespace},
	}

	// Pin public API outputs, including the normalization and global-mode contracts.
	for _, implementation := range implementations {
		t.Run(implementation.name, func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					t.Log("Verify fixture names are independently valid Kubernetes names")
					if errors := validation.IsDNS1123Label(tt.namespace); len(errors) != 0 {
						t.Fatalf("invalid namespace fixture: %v", errors)
					}
					if errors := validation.IsDNS1123Subdomain(tt.dgdName); len(errors) != 0 {
						t.Fatalf("invalid DGD fixture: %v", errors)
					}

					t.Log("Verify the derived identity matches its golden value and fits a label")
					got := implementation.compute(tt.global, tt.namespace, tt.dgdName)
					if got != tt.want {
						t.Fatalf("namespace = %q, want %q", got, tt.want)
					}
					if errors := validation.IsValidLabelValue(got); len(errors) != 0 {
						t.Fatalf("invalid derived label %q: %v", got, errors)
					}
					if repeated := implementation.compute(tt.global, tt.namespace, tt.dgdName); repeated != got {
						t.Fatalf("namespace changed between calls: %q != %q", repeated, got)
					}
				})
			}
		})
	}
}
