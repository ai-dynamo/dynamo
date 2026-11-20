// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package semanticenv

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestLookupResolvesOnlySelectedInputs(t *testing.T) {
	for _, test := range []struct {
		name      string
		general   []corev1.EnvVar
		specific  []corev1.EnvVar
		value     string
		known     bool
		absent    bool
		fullError bool
	}{
		{name: "absent", absent: true},
		{name: "known empty", specific: []corev1.EnvVar{{Name: "TARGET"}}, known: true},
		{name: "cross-layer alias ignores unrelated cycle", known: true, value: "known", fullError: true,
			general:  []corev1.EnvVar{{Name: "TARGET", Value: "$(VALUE)"}, {Name: "LEFT", Value: "$(RIGHT)"}},
			specific: []corev1.EnvVar{{Name: "VALUE", Value: "known"}, {Name: "RIGHT", Value: "$(LEFT)"}}},
		{name: "transitive cross-layer inputs", known: true, value: "root/child",
			general:  []corev1.EnvVar{{Name: "ROOT", Value: "root"}, {Name: "TARGET", Value: "$(PATH)"}},
			specific: []corev1.EnvVar{{Name: "PATH", Value: "$(ROOT)/child"}}},
		{name: "same-list reference uses its earlier duplicate", known: true, value: "first",
			specific: []corev1.EnvVar{{Name: "VALUE", Value: "first"}, {Name: "TARGET", Value: "$(VALUE)"}, {Name: "VALUE", Value: "last"}}},
		{name: "self-reference uses its earlier occurrence", known: true, value: "root/child",
			specific: []corev1.EnvVar{{Name: "TARGET", Value: "root"}, {Name: "TARGET", Value: "$(TARGET)/child"}}},
		{name: "final literal shadows earlier self-reference", known: true, value: "literal",
			specific: []corev1.EnvVar{{Name: "TARGET", Value: "root"}, {Name: "TARGET", Value: "$(TARGET)/child"}, {Name: "TARGET", Value: "literal"}}},
		{name: "discarded lower source cannot supply self-reference",
			general: []corev1.EnvVar{{Name: "TARGET", Value: "discarded"}}, specific: []corev1.EnvVar{{Name: "TARGET", Value: "$(TARGET)"}}},
		{name: "local forward reference stays unknown",
			specific: []corev1.EnvVar{{Name: "TARGET", Value: "$(VALUE)"}, {Name: "VALUE", Value: "later"}}},
		{name: "unknown source shadows lower literal",
			general:  []corev1.EnvVar{{Name: "VALUE", Value: "discarded"}, {Name: "TARGET", Value: "$(VALUE)"}},
			specific: []corev1.EnvVar{{Name: "VALUE", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "value"}}}}},
		{name: "escaped references remain literal", known: true, value: "$(MISSING)",
			specific: []corev1.EnvVar{{Name: "TARGET", Value: "$$(MISSING)"}}},
		{name: "inserted values are not expanded again", known: true, value: "$(MISSING)",
			specific: []corev1.EnvVar{{Name: "VALUE", Value: "$$(MISSING)"}, {Name: "TARGET", Value: "$(VALUE)"}}},
		{name: "unused earlier occurrence does not supply final value", known: true, value: "literal", fullError: true,
			general:  []corev1.EnvVar{{Name: "TARGET", Value: "$(VALUE)"}, {Name: "TARGET", Value: "literal"}},
			specific: []corev1.EnvVar{{Name: "VALUE", Value: "$(TARGET)"}}},
		{name: "dependent cycle stays unknown", fullError: true,
			general: []corev1.EnvVar{{Name: "TARGET", Value: "$(VALUE)"}}, specific: []corev1.EnvVar{{Name: "VALUE", Value: "$(TARGET)"}}},
		{name: "dependent local-order conflict stays unknown", fullError: true,
			general:  []corev1.EnvVar{{Name: "TARGET", Value: "$(VALUE)"}, {Name: "ROOT", Value: "literal"}},
			specific: []corev1.EnvVar{{Name: "VALUE", Value: "$(ROOT)"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the requested final binding without mutating or validating unrelated sources")
			general := (&corev1.Container{Env: test.general}).DeepCopy().Env
			specific := (&corev1.Container{Env: test.specific}).DeepCopy().Env
			layers := []Layer{{Source: "general", Env: test.general}, {Source: "specific", Env: test.specific}}
			value, exists := Lookup("TARGET", layers...)
			require.Equal(t, !test.absent, exists)
			require.Equal(t, test.known, value.Known)
			if test.known {
				require.Equal(t, test.value, value.Value)
			}
			require.Equal(t, general, test.general)
			require.Equal(t, specific, test.specific)

			t.Log("A final render still rejects the complete source graph's genuine conflicts")
			composed, err := Compose(layers...)
			if test.fullError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				for name, expected := range Replay(composed) {
					selected, declared := Lookup(name, layers...)
					require.True(t, declared)
					if selected.Known {
						require.Equal(t, expected, selected, "known lookup of %s agrees with full expansion", name)
					}
				}
			}
		})
	}
}

func FuzzLookupMatchesComposedReplay(f *testing.F) {
	for _, seed := range [][]byte{{0, 1, 2}, {0, 16, 33, 50}, {0, 32, 64, 96, 128, 160}, {255, 254, 253, 252, 251}} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		// Bound the input while varying sources, duplicates, self/forward references, and escaping.
		if len(data) > 64 {
			data = data[:64]
		}
		names := []string{"A", "B", "C", "D"}
		layers := []Layer{{Source: "general"}, {Source: "specific"}}
		for _, item := range data {
			variable := corev1.EnvVar{Name: names[int(item)%len(names)]}
			switch (item / 4) % 8 {
			case 0:
				variable.Value = "literal"
			case 1:
				variable.Value = ""
			case 2, 3, 4, 5:
				variable.Value = "$(" + names[int((item/4)%8)-2] + ")"
			case 6:
				variable.Value = "$$(" + variable.Name + ")"
			case 7:
				variable.ValueFrom = &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "value"}}
			}
			layer := int(item/32) % len(layers)
			layers[layer].Env = append(layers[layer].Env, variable)
		}
		composed, err := Compose(layers...)
		if err != nil {
			return
		}
		for name, expected := range Replay(composed) {
			selected, declared := Lookup(name, layers...)
			require.True(t, declared)
			if selected.Known {
				require.Equal(t, expected, selected, "known lookup of %s agrees with full expansion", name)
			}
		}
	})
}
