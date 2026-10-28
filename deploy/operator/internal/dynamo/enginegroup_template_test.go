/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestGroveEngineGroupTopologyAliases(t *testing.T) {
	t.Log("declare a provider-independent launch with environment references that need topology first")
	main := corev1.Container{
		Name: consts.MainContainerName, Command: []string{"engine"},
		Args: []string{"--rank=$(DYNAMO_RANK)", "--address=$(DYNAMO_LEADER_ADDRESS)"},
		Env:  []corev1.EnvVar{{Name: "RENDEZVOUS", Value: "$(DYNAMO_LEADER_ADDRESS):29500"}},
	}
	expected := main.DeepCopy()
	expected.Env = append([]corev1.EnvVar{
		{Name: consts.DynamoRankEnvVar, Value: "$(GROVE_PCLQ_POD_INDEX)"},
		{Name: consts.DynamoLeaderAddressEnvVar, Value: "$(GROVE_PCLQ_NAME)-0.$(GROVE_HEADLESS_SERVICE)"},
	}, main.Env...)

	t.Log("supply only the two aliases before caller environment, without rewriting the launch")
	addGroveEngineGroupTopologyEnvVars(&main)
	assert.Equal(t, *expected, main)
}

func TestEngineGroupTopologyAliasesAreOperatorOwned(t *testing.T) {
	for _, name := range []string{consts.DynamoRankEnvVar, consts.DynamoLeaderAddressEnvVar} {
		t.Run(name, func(t *testing.T) {
			t.Log("attempt to shadow an allocation topology alias in the explicit template")
			template := &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
				RestartPolicy: corev1.RestartPolicyNever,
				Containers:    []corev1.Container{{Name: consts.MainContainerName, Env: []corev1.EnvVar{{Name: name, Value: "wrong"}}}},
			}}
			before := template.DeepCopy()

			t.Log("reject the conflicting declaration rather than silently overwriting it")
			require.ErrorContains(t, validateEngineGroupPodTemplate(template), "operator-owned topology")
			assert.Equal(t, before, template)
		})
	}
}

func TestEngineGroupIncarnationRequiresDownwardIdentity(t *testing.T) {
	for _, tc := range []struct {
		name      string
		value     string
		field     string
		version   string
		wantError bool
	}{
		{"implicit API version", "", "metadata.uid", "", false},
		{"explicit API version", "", "metadata.uid", "v1", false},
		{"caller supplied incarnation", "stale", "metadata.uid", "", true},
		{"name is not lifetime identity", "", "metadata.name", "", true},
		{"wrong API version", "", "metadata.uid", "v2", true},
		{"missing field", "", "", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("declare native process identity through Kubernetes metadata")
			main := &corev1.Container{Name: consts.MainContainerName, Env: []corev1.EnvVar{{
				Name: "PROCESS_UID", Value: tc.value,
				ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: tc.version, FieldPath: tc.field}},
			}}}
			before := main.DeepCopy()

			t.Log("accept only the exact Downward API identity without modifying the launch")
			err := requireEngineGroupFieldEnv(main, "PROCESS_UID", "metadata.uid")
			if tc.wantError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, before, main)
		})
	}
}

func TestGroveEngineGroupContainerProbeOwnership(t *testing.T) {
	for _, declared := range []bool{false, true} {
		name := "waiting joiner has no inferred HTTP probe"
		if declared {
			name = "explicit allocation probes are retained"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("start with the standard worker probes and an explicit native launch")
			main, err := NewWorkerDefaults().GetBaseContainer(ComponentContext{})
			require.NoError(t, err)
			require.NotNil(t, main.ReadinessProbe)
			main.Command, main.Args = []string{"engine"}, []string{"--rank=$(DYNAMO_RANK)"}
			template := &corev1.Container{Name: consts.MainContainerName}
			if declared {
				probe := &corev1.Probe{Exec: &corev1.ExecAction{Command: []string{"check-process"}}}
				template.LivenessProbe, template.ReadinessProbe, template.StartupProbe = probe.DeepCopy(), probe.DeepCopy(), probe.DeepCopy()
			}
			beforeTemplate := template.DeepCopy()
			beforeMain := main.DeepCopy()

			t.Log("apply the allocation probe contract without changing the engine command or shared defaults")
			configureGroveEngineGroupContainer(&main, template)
			assert.Equal(t, template.LivenessProbe, main.LivenessProbe)
			assert.Equal(t, template.ReadinessProbe, main.ReadinessProbe)
			assert.Equal(t, template.StartupProbe, main.StartupProbe)
			assert.Equal(t, beforeMain.Command, main.Command)
			assert.Equal(t, beforeMain.Args, main.Args)
			assert.Equal(t, beforeMain.Env, main.Env[2:])
			assert.Equal(t, beforeTemplate, template)
		})
	}
}
