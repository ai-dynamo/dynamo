/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestReplayBoundsRepeatedExpansion(t *testing.T) {
	t.Log("Repeated aliases must not turn a small manifest into exponential operator allocations")
	env := []corev1.EnvVar{{Name: "A0", Value: "x"}}
	for i := 1; i <= 30; i++ {
		previous := "$(A" + strconv.Itoa(i-1) + ")"
		env = append(env, corev1.EnvVar{Name: "A" + strconv.Itoa(i), Value: previous + previous})
	}
	values := Replay(env)
	require.Equal(t, Value{Value: "xx", Known: true}, values["A1"])
	require.False(t, values["A30"].Known)
	total := 0
	for _, value := range values {
		total += len(value.Value)
	}
	require.LessOrEqual(t, total, maxReplayBytes)
}

func TestReplay(t *testing.T) {
	for _, test := range []struct {
		name string
		env  []corev1.EnvVar
		want Value
	}{
		{name: "known empty", env: []corev1.EnvVar{{Name: "B"}}, want: Value{Known: true}},
		{name: "earlier literal", env: []corev1.EnvVar{{Name: "A", Value: "30500"}, {Name: "B", Value: "$(A)"}}, want: Value{Value: "30500", Known: true}},
		{name: "expansion time not final map", env: []corev1.EnvVar{{Name: "A", Value: "30500"}, {Name: "B", Value: "$(A)"}, {Name: "A", Value: "30600"}}, want: Value{Value: "30500", Known: true}},
		{name: "forward reference stays unknown", env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}, {Name: "A", Value: "30500"}}, want: Value{Value: "$(A)"}},
		{name: "self reference reads prior value", env: []corev1.EnvVar{{Name: "B", Value: "30"}, {Name: "B", Value: "$(B)500"}}, want: Value{Value: "30500", Known: true}},
		{name: "unbound self reference", env: []corev1.EnvVar{{Name: "B", Value: "$(B)"}}, want: Value{Value: "$(B)"}},
		{name: "provider value stays unknown", env: []corev1.EnvVar{{Name: "B", Value: "$(GROVE_PCSG_POD_INDEX)"}}, want: Value{Value: "$(GROVE_PCSG_POD_INDEX)"}},
		{name: "empty reference is a runtime lookup", env: []corev1.EnvVar{{Name: "B", Value: "$()"}}, want: Value{Value: "$()"}},
		{name: "escaped reference is a known literal", env: []corev1.EnvVar{{Name: "A", Value: "30500"}, {Name: "B", Value: "$$(A)"}}, want: Value{Value: "$(A)", Known: true}},
		{name: "odd dollar count", env: []corev1.EnvVar{{Name: "A", Value: "30500"}, {Name: "B", Value: "$$$(A)"}}, want: Value{Value: "$30500", Known: true}},
		{name: "dollar pairs", env: []corev1.EnvVar{{Name: "B", Value: "$$$$"}}, want: Value{Value: "$$", Known: true}},
		{name: "inserted values are not reexpanded", env: []corev1.EnvVar{{Name: "A", Value: "$$(MISSING)"}, {Name: "B", Value: "$(A)"}}, want: Value{Value: "$(MISSING)", Known: true}},
		{name: "incomplete reference still unescapes trailing dollars", env: []corev1.EnvVar{{Name: "B", Value: "$(A$$"}}, want: Value{Value: "$(A$", Known: true}},
		{name: "shell syntax is not a Kubernetes lookup", env: []corev1.EnvVar{{Name: "B", Value: "$A/${A}/終$"}}, want: Value{Value: "$A/${A}/終$", Known: true}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Replay the ordered list without modifying authored values")
			original := (&corev1.Container{Env: test.env}).DeepCopy().Env
			values := Replay(test.env)
			require.Equal(t, test.want, values["B"])
			require.Contains(t, values, "B")
			require.NotContains(t, values, "ABSENT")
			require.Equal(t, original, test.env)
		})
	}
}

func TestReplayOpaqueValuesShadowEarlierDefinitions(t *testing.T) {
	for _, source := range []struct {
		name  string
		value corev1.EnvVarSource
	}{
		{name: "ConfigMap", value: corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "settings", Key: "port"}}},
		{name: "Secret", value: corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "port"}}},
		{name: "fieldRef", value: corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
		{name: "resourceFieldRef", value: corev1.EnvVarSource{ResourceFieldRef: &corev1.ResourceFieldSelector{Resource: "limits.cpu"}}},
	} {
		t.Run(source.name, func(t *testing.T) {
			t.Log("An unavailable definition replaces the earlier literal, including in aliases")
			env := []corev1.EnvVar{{Name: "A", Value: "30500"}, {Name: "A", ValueFrom: &source.value}, {Name: "B", Value: "$(A)"}}
			values := Replay(env)
			require.False(t, values["A"].Known)
			require.False(t, values["B"].Known)

			t.Log("A later literal can replace the unavailable source before a subsequent lookup")
			env = append(env, corev1.EnvVar{Name: "A", Value: "30600"}, corev1.EnvVar{Name: "B", Value: "$(A)"})
			require.Equal(t, Value{Value: "30600", Known: true}, Replay(env)["B"])
		})
	}
}
