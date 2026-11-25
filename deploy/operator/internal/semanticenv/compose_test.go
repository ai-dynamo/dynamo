/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv_test

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/semanticenv"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func BenchmarkComposeUnterminatedReferences(b *testing.B) {
	for _, test := range []struct {
		name string
		size int
	}{{name: "1K", size: 1_000}, {name: "100K", size: 100_000}} {
		b.Run(test.name, func(b *testing.B) {
			layer := semanticenv.Layer{Env: []corev1.EnvVar{{Name: "VALUE", Value: strings.Repeat("$(", test.size)}}}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := semanticenv.Compose(layer); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func TestCompose(t *testing.T) {
	tests := []struct {
		name   string
		layers []semanticenv.Layer
		want   []corev1.EnvVar
	}{
		{name: "empty"},
		{
			name: "stable general then specific without alphabetical sorting",
			layers: []semanticenv.Layer{
				{Source: "defaults", Env: []corev1.EnvVar{{Name: "Z", Value: "default"}}},
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "B", Value: "b"}, {Name: "A", Value: "a"}}},
			},
			want: []corev1.EnvVar{{Name: "Z", Value: "default"}, {Name: "B", Value: "b"}, {Name: "A", Value: "a"}},
		},
		{
			name: "specific override feeds a general reference",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "BASE", Value: "/models"}, {Name: "MODEL", Value: "$(BASE)/qwen"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "STORAGE", Value: "/fast"}, {Name: "BASE", Value: "$(STORAGE)/models"}}},
			},
			want: []corev1.EnvVar{{Name: "STORAGE", Value: "/fast"}, {Name: "BASE", Value: "$(STORAGE)/models"}, {Name: "MODEL", Value: "$(BASE)/qwen"}},
		},
		{
			name: "specific reference to a general definition",
			layers: []semanticenv.Layer{
				{Source: "defaults", Env: []corev1.EnvVar{{Name: "Z_BASE", Value: "/models"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "A_MODEL", Value: "$(Z_BASE)/qwen"}}},
			},
			want: []corev1.EnvVar{{Name: "Z_BASE", Value: "/models"}, {Name: "A_MODEL", Value: "$(Z_BASE)/qwen"}},
		},
		{
			name: "three layers must remain distinct until composition",
			layers: []semanticenv.Layer{
				{Source: "defaults", Env: []corev1.EnvVar{{Name: "SYSTEM", Value: "$(B)"}}},
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "A", Value: "$(SYSTEM)"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "B", Value: "b"}}},
			},
			want: []corev1.EnvVar{{Name: "B", Value: "b"}, {Name: "SYSTEM", Value: "$(B)"}, {Name: "A", Value: "$(SYSTEM)"}},
		},
		{
			name: "local forward references are not repaired",
			layers: []semanticenv.Layer{{Source: "container.env", Env: []corev1.EnvVar{
				{Name: "A", Value: "$(B)"}, {Name: "B", Value: "$(A)"},
			}}},
			want: []corev1.EnvVar{{Name: "A", Value: "$(B)"}, {Name: "B", Value: "$(A)"}},
		},
		{
			name: "escaping follows Kubernetes dollar-pair consumption",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{
					{Name: "ESCAPED", Value: "$$(BASE)"}, {Name: "EXPANDED", Value: "$$$(BASE)-$(BASE)"},
				}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "BASE", Value: "b"}}},
			},
			want: []corev1.EnvVar{{Name: "ESCAPED", Value: "$$(BASE)"}, {Name: "BASE", Value: "b"}, {Name: "EXPANDED", Value: "$$$(BASE)-$(BASE)"}},
		},
		{
			name: "missing shell malformed and self references stay runtime lookups",
			layers: []semanticenv.Layer{{Source: "container.env", Env: []corev1.EnvVar{
				{Name: "PATH", Value: "$(PATH):/extra"}, {Name: "VALUE", Value: "$(EXTERNAL)-$PATH-${PATH}-$(unclosed"},
			}}},
			want: []corev1.EnvVar{{Name: "PATH", Value: "$(PATH):/extra"}, {Name: "VALUE", Value: "$(EXTERNAL)-$PATH-${PATH}-$(unclosed"}},
		},
		{
			name: "a valid reference before an unterminated suffix still orders its definition",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "VALUE", Value: "$(BASE)" + strings.Repeat("$(", 100_000)}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "BASE", Value: "b"}}},
			},
			want: []corev1.EnvVar{{Name: "BASE", Value: "b"}, {Name: "VALUE", Value: "$(BASE)" + strings.Repeat("$(", 100_000)}},
		},
		{
			name: "discarded definitions cannot introduce cycles",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "A", Value: "$(B)"}, {Name: "B", Value: "$(A)"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "A", Value: "override"}}},
			},
			want: []corev1.EnvVar{{Name: "A", Value: "override"}, {Name: "B", Value: "$(A)"}},
		},
		{
			name: "same-list duplicates retain sequential semantics",
			layers: []semanticenv.Layer{{Source: "container.env", Env: []corev1.EnvVar{
				{Name: "A", Value: "first"}, {Name: "B", Value: "$(A)"}, {Name: "A", Value: "last"},
			}}},
			want: []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "B", Value: "$(A)"}, {Name: "A", Value: "last"}},
		},
		{
			name: "a cross-layer reference follows the last same-list definition",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "RESULT", Value: "$(A)"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "last"}}},
			},
			want: []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "last"}, {Name: "RESULT", Value: "$(A)"}},
		},
		{
			name: "overriding a self reference does not resurrect its lower layer",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "PATH", Value: "/discard"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "PATH", Value: "$(PATH):/extra"}}},
			},
			want: []corev1.EnvVar{{Name: "PATH", Value: "$(PATH):/extra"}},
		},
		{
			name: "valueFrom replaces a literal as one opaque definition",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "BASE", Value: "discard"}, {Name: "MODEL", Value: "$(BASE)/model"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "BASE", ValueFrom: &corev1.EnvVarSource{
					ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "paths", Key: "base"},
				}}}},
			},
			want: []corev1.EnvVar{
				{Name: "BASE", ValueFrom: &corev1.EnvVarSource{
					ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "paths", Key: "base"},
				}},
				{Name: "MODEL", Value: "$(BASE)/model"},
			},
		},
		{
			name: "a literal replaces the complete lower valueFrom",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "POD", ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
				}}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "POD", Value: "override"}}},
			},
			want: []corev1.EnvVar{{Name: "POD", Value: "override"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Compose the complete source layers without expanding their values")
			got, err := semanticenv.Compose(tt.layers...)
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

func TestComposeRejectsCycles(t *testing.T) {
	tests := []struct {
		name   string
		layers []semanticenv.Layer
		want   string
	}{
		{
			name: "cross-layer dependency cycle",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "A", Value: "$(B)"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}}},
			},
			want: "environment dependency cycle: A (spec.env[0]) -> B (container.env[0]) -> A (spec.env[0])",
		},
		{
			name: "dependency conflicts with authored local order",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "A", Value: "$(B)"}, {Name: "C", Value: "c"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "B", Value: "$(C)"}}},
			},
			want: "environment dependency cycle: A (spec.env[0]) -> C (spec.env[1]) -> B (container.env[0]) -> A (spec.env[0])",
		},
		{
			name: "cycle diagnostics retain original indexes after override selection",
			layers: []semanticenv.Layer{
				{Source: "spec.env", Env: []corev1.EnvVar{{Name: "DISCARDED", Value: "old"}, {Name: "A", Value: "$(B)"}}},
				{Source: "container.env", Env: []corev1.EnvVar{{Name: "DISCARDED", Value: "new"}, {Name: "B", Value: "$(A)"}}},
			},
			want: "environment dependency cycle: A (spec.env[1]) -> B (container.env[1]) -> A (spec.env[1])",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Refuse an unsatisfiable composition and report the authored cycle")
			got, err := semanticenv.Compose(tt.layers...)
			require.EqualError(t, err, tt.want)
			require.Nil(t, got)
		})
	}
}

func TestComposeOwnsOutput(t *testing.T) {
	t.Log("Compose a sourced value without aliasing its caller-owned definition")
	input := semanticenv.Layer{Source: "container.env", Env: []corev1.EnvVar{{Name: "POD", ValueFrom: &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"},
	}}}}
	got, err := semanticenv.Compose(input)
	require.NoError(t, err)

	t.Log("Changing the result must not mutate the original Kubernetes objects")
	got[0].ValueFrom.FieldRef.FieldPath = "metadata.namespace"
	require.Equal(t, "metadata.name", input.Env[0].ValueFrom.FieldRef.FieldPath)
}

func TestWithoutPreservesSourcePositions(t *testing.T) {
	t.Log("Apply repeated container transformations without renumbering authored entries")
	general := semanticenv.Layer{Source: "spec.env", Env: []corev1.EnvVar{
		{Name: "REMOVE_FIRST"}, {Name: "REMOVE_SECOND"}, {Name: "A", Value: "$(B)"},
	}}
	specific := semanticenv.Layer{Source: "container.env", Env: []corev1.EnvVar{
		{Name: "REMOVE_FIRST"}, {Name: "B", Value: "$(A)"},
	}}
	filtered := general.Without("REMOVE_FIRST").Without("REMOVE_SECOND")
	_, err := semanticenv.Compose(filtered, specific.Without("REMOVE_FIRST"))
	require.EqualError(t, err, "environment dependency cycle: A (spec.env[2]) -> B (container.env[1]) -> A (spec.env[2])")
	require.Len(t, general.Env, 3)
	require.Len(t, specific.Env, 2)
}
