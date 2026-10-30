/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strings"
	"testing"

	config "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestVLLMNativeTemplate(t *testing.T) {
	for _, worker := range []bool{false, true} {
		name := "native HTTP server"
		if worker {
			name = "Dynamo worker"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("declare the same native command for initial participants and later joiners")
			component := newVLLMComponentEngineGroupForTest()
			if worker {
				component.PodTemplate.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo.vllm", "--model", "model"}
			}
			before := component.DeepCopy()

			t.Log("validate the declared native bootstrap without changing commands, environment or lifecycle settings")
			source, profile, err := ResolveVLLMComponentEngineGroup(component, nil)
			require.NoError(t, err)
			assert.Equal(t, before, component)
			assert.Equal(t, int32(8), profile.InitialReplicas)
			assert.Equal(t, worker, IsVLLMWorkerLaunch(source))
			if worker {
				port, _, err := ResolveVLLMControlPorts(source)
				require.NoError(t, err)
				assert.Equal(t, int32(9090), port, "use the existing worker control-port default")
			}

			t.Log("reuse physical Pod rendering without adding or rewriting native command arguments")
			dgd := &api.DynamoGraphDeployment{Name: "world", Namespace: "test",
				Spec: api.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []api.DynamoComponentDeploymentSharedSpec{*component}}}
			pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &config.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, nil, nil, nil, false, nil)
			require.NoError(t, err)
			main := pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0]
			assert.Equal(t, before.PodTemplate.Spec.Containers[0].Command, main.Command)
			assert.Equal(t, before.PodTemplate.Spec.Containers[0].Args, main.Args)
			assert.Equal(t, before, component)
		})
	}
}

func TestVLLMPackedNativeTemplate(t *testing.T) {
	for _, test := range []struct {
		name                        string
		tp, local, initial, maximum int32
	}{
		{name: "EP2 seed on one two-GPU Pod", tp: 1, local: 2, initial: 1, maximum: 2},
		{name: "EP8 on two four-GPU Pods", tp: 1, local: 4, initial: 2, maximum: 3},
		{name: "two TP2 engines per Pod", tp: 2, local: 2, initial: 1, maximum: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("declare a single worker owning the Pod's complete native engine range")
			component := newVLLMComponentEngineGroupForTest()
			component.EngineGroup.InitialSize = test.initial
			component.EngineGroup.Policy = nil
			main := &component.PodTemplate.Spec.Containers[0]
			main.Command = []string{"python3", "-m", "dynamo.vllm", "--model", "model"}
			main.Args = append(withoutTestVLLMProfileOptions(main.Args, "--tensor-parallel-size", "--data-parallel-rank", "--data-parallel-size-local", "--data-parallel-external-lb", "--elastic-ep-initial-dp-size", "--elastic-ep-max-dp-size"),
				fmt.Sprintf("--tensor-parallel-size=%d", test.tp), fmt.Sprintf("--data-parallel-size-local=%d", test.local),
				fmt.Sprintf("--elastic-ep-initial-dp-size=%d", test.initial*test.local), fmt.Sprintf("--elastic-ep-max-dp-size=%d", test.maximum*test.local),
				"--elastic-ep-allocation-index="+engineGroupOrdinalRef)
			main.Resources.Limits["nvidia.com/gpu"] = resource.MustParse(fmt.Sprint(test.tp * test.local))
			main.Resources.Requests = nil
			before := component.DeepCopy()

			t.Log("resolve allocation counts separately from DP/EP geometry without altering the launch template")
			_, profile, err := ResolveVLLMComponentEngineGroup(component, nil)
			require.NoError(t, err)
			require.Equal(t, test.initial, profile.InitialReplicas)
			require.Equal(t, test.maximum, profile.MaximumReplicas)
			require.Equal(t, test.local, profile.NativeMembersPerReplica)
			require.Equal(t, int32(1), profile.Geometry.PodsPerReplica)
			require.Equal(t, before, component)
		})
	}
}

func TestVLLMNativeTemplateRequiresExplicitBootstrap(t *testing.T) {
	for _, option := range []string{"--elastic-ep-auto-bootstrap", "--elastic-ep-initial-dp-size", "--data-parallel-rank", "--data-parallel-address", "--data-parallel-external-lb", "--data-parallel-size-local"} {
		t.Run(option, func(t *testing.T) {
			t.Log("omit a required native bootstrap setting from an otherwise complete declaration")
			component := newVLLMComponentEngineGroupForTest()
			component.PodTemplate.Spec.Containers[0].Args = withoutTestVLLMProfileOptions(component.PodTemplate.Spec.Containers[0].Args, option)

			t.Log("reject rather than injecting the missing setting")
			_, _, err := ResolveVLLMComponentEngineGroup(component, nil)
			require.Error(t, err)
		})
	}
}

func TestVLLMAutomaticProfilePreservesGeometry(t *testing.T) {
	t.Log("resolve the immutable initial size using native automatic bootstrap")
	source := newTestVLLMProfileGeometrySource()
	explicit, err := ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	for i, arg := range source.Args {
		if strings.HasPrefix(arg, "--data-parallel-size=") {
			source.Args[i] = "--elastic-ep-initial-dp-size=8"
		}
	}
	source.Args = append(source.Args, "--elastic-ep-auto-bootstrap")
	automatic, err := ResolveVLLMElasticEPProfile(source)
	require.NoError(t, err)
	assert.Equal(t, explicit, automatic)
}
