package dynamo

import (
	"reflect"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestSGLangBackend_DirectFlagInjection(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name                    string
		numberOfNodes           int32
		role                    Role
		multinodeDeploymentType consts.MultinodeDeploymentType
		initialArgs             []string
		expectedArgs            []string
		description             string
	}{
		{
			name:                    "single node does not modify args",
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"python -m dynamo.sglang.worker"},
			expectedArgs:            []string{"python -m dynamo.sglang.worker"},
			description:             "Single node should not modify anything",
		},
		{
			name:                    "multinode adds flags to simple python command",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"python -m dynamo.sglang.worker"},
			expectedArgs:            []string{"python -m dynamo.sglang.worker --dist-init-addr ${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-test-service-ldr-0.${GROVE_HEADLESS_SERVICE}:29500 --nnodes 2 --node-rank 0"},
			description:             "Should add multinode flags directly to python command",
		},
		{
			name:                    "multinode with complex command",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"echo blah | wc -l && python -m dynamo.sglang.worker && ls -al"},
			expectedArgs:            []string{"echo blah | wc -l && python -m dynamo.sglang.worker --dist-init-addr ${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-test-service-ldr-0.${GROVE_HEADLESS_SERVICE}:29500 --nnodes 2 --node-rank 0 && ls -al"},
			description:             "Should add flags only to python command, not other commands",
		},
		{
			name:                    "multinode worker with Grove deployment",
			numberOfNodes:           3,
			role:                    RoleWorker,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"python -m dynamo.sglang.worker"},
			expectedArgs:            []string{"python -m dynamo.sglang.worker --dist-init-addr ${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-test-service-ldr-0.${GROVE_HEADLESS_SERVICE}:29500 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1))"},
			description:             "Worker should get correct node rank",
		},
		{
			name:                    "LWS deployment uses correct address",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeLWS,
			initialArgs:             []string{"python -m dynamo.sglang.worker"},
			expectedArgs:            []string{"python -m dynamo.sglang.worker --dist-init-addr ${LWS_LEADER_ADDRESS}:29500 --nnodes 2 --node-rank 0"},
			description:             "LWS deployment should use LWS_LEADER_ADDRESS",
		},
		{
			name:                    "command with pipes gets flags before pipe",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"python -m dynamo.sglang.worker | tee /tmp/log"},
			expectedArgs:            []string{"python -m dynamo.sglang.worker --dist-init-addr ${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-test-service-ldr-0.${GROVE_HEADLESS_SERVICE}:29500 --nnodes 2 --node-rank 0 | tee /tmp/log"},
			description:             "Should insert flags before pipe operator",
		},
		{
			name:                    "multiple args are flattened and processed together",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"echo start", "python -m dynamo.sglang.worker", "echo done"},
			expectedArgs:            []string{"echo start python -m dynamo.sglang.worker --dist-init-addr ${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-test-service-ldr-0.${GROVE_HEADLESS_SERVICE}:29500 --nnodes 2 --node-rank 0 echo done"},
			description:             "Multiple args should be flattened and python command gets flags",
		},
		{
			name:                    "no sglang command means flattened but no changes",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			initialArgs:             []string{"echo hello", "python -m some.other.module"},
			expectedArgs:            []string{"echo hello python -m some.other.module"},
			description:             "Non-sglang commands should be flattened but not modified",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{
				Args: append([]string{}, tt.initialArgs...),
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, &v1alpha1.DynamoComponentDeploymentOverridesSpec{}, tt.multinodeDeploymentType, "test-service")

			if !reflect.DeepEqual(container.Args, tt.expectedArgs) {
				t.Errorf("UpdateContainer() args = %v, want %v", container.Args, tt.expectedArgs)
			}

			// Verify no environment variables were added
			if len(container.Env) > 0 {
				t.Errorf("UpdateContainer() should not add environment variables, but added: %v", container.Env)
			}

			// Verify command was not changed
			if len(container.Command) > 0 {
				t.Errorf("UpdateContainer() should not modify command, but set: %v", container.Command)
			}
		})
	}
}

func TestSGLangBackend_ProbeRemoval(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name                    string
		numberOfNodes           int32
		role                    Role
		multinodeDeploymentType consts.MultinodeDeploymentType
		expectProbesRemoved     bool
	}{
		{
			name:                    "single node does not remove probes",
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectProbesRemoved:     false,
		},
		{
			name:                    "multinode leader removes probes",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectProbesRemoved:     true,
		},
		{
			name:                    "multinode worker removes probes",
			numberOfNodes:           2,
			role:                    RoleWorker,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectProbesRemoved:     true,
		},
		{
			name:                    "multinode main role does not remove probes",
			numberOfNodes:           2,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectProbesRemoved:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create initial probes
			livenessProbe := &corev1.Probe{InitialDelaySeconds: 30}
			readinessProbe := &corev1.Probe{InitialDelaySeconds: 10}
			startupProbe := &corev1.Probe{InitialDelaySeconds: 5}

			container := &corev1.Container{
				Args:           []string{"python -m dynamo.sglang.worker"},
				LivenessProbe:  livenessProbe,
				ReadinessProbe: readinessProbe,
				StartupProbe:   startupProbe,
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, &v1alpha1.DynamoComponentDeploymentOverridesSpec{}, tt.multinodeDeploymentType, "test-service")

			if tt.expectProbesRemoved {
				if container.LivenessProbe != nil {
					t.Errorf("Expected LivenessProbe to be removed, but it was not")
				}
				if container.ReadinessProbe != nil {
					t.Errorf("Expected ReadinessProbe to be removed, but it was not")
				}
				if container.StartupProbe != nil {
					t.Errorf("Expected StartupProbe to be removed, but it was not")
				}
			} else {
				if container.LivenessProbe == nil {
					t.Errorf("Expected LivenessProbe to be preserved, but it was removed")
				}
				if container.ReadinessProbe == nil {
					t.Errorf("Expected ReadinessProbe to be preserved, but it was removed")
				}
				if container.StartupProbe == nil {
					t.Errorf("Expected StartupProbe to be preserved, but it was removed")
				}
			}
		})
	}
}
