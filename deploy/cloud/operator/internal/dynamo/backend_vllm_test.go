package dynamo

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/onsi/gomega"
	ptr "k8s.io/utils/ptr"
)

func TestVLLMBackend_GenerateCommandAndArgs(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                    string
		componentType           string
		numberOfNodes           int32
		role                    Role
		component               *v1alpha1.DynamoComponentDeploymentOverridesSpec
		multinodeDeploymentType consts.MultinodeDeploymentType
		expectedCmd             []string
		expectedArgs            []string
		expectContains          []string
	}{
		{
			name:          "main component",
			componentType: commonconsts.ComponentTypeMain,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectedArgs:            []string{"python3 -m dynamo.frontend --http-port 8000"},
			expectContains:          []string{"python3 -m dynamo.frontend"},
		},
		{
			name:          "worker single node",
			componentType: commonconsts.ComponentTypeWorker,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.vllm"},
		},
		{
			name:          "worker multinode leader",
			componentType: commonconsts.ComponentTypeWorker,
			numberOfNodes: 3,
			role:          RoleLeader,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"ray start --head --port=6379", "python3 -m dynamo.vllm"},
		},
		{
			name:          "worker multinode worker",
			componentType: commonconsts.ComponentTypeWorker,
			numberOfNodes: 3,
			role:          RoleWorker,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectedArgs:            []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"},
			expectContains:          []string{"ray start --address"},
		},
		{
			name:          "prefill worker single node",
			componentType: commonconsts.ComponentTypePrefillWorker,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.vllm", "--is-prefill-worker"},
		},
		{
			name:          "prefill worker multinode leader",
			componentType: commonconsts.ComponentTypePrefillWorker,
			numberOfNodes: 3,
			role:          RoleLeader,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"ray start --head --port=6379", "python3 -m dynamo.vllm", "--is-prefill-worker"},
		},
		{
			name:          "prefill worker multinode worker",
			componentType: commonconsts.ComponentTypePrefillWorker,
			numberOfNodes: 3,
			role:          RoleWorker,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectedArgs:            []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"},
			expectContains:          []string{"ray start --address"},
		},
		{
			name:          "decode worker single node",
			componentType: commonconsts.ComponentTypeDecodeWorker,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.vllm"},
		},
		{
			name:          "decode worker multinode leader",
			componentType: commonconsts.ComponentTypeDecodeWorker,
			numberOfNodes: 3,
			role:          RoleLeader,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"ray start --head --port=6379", "python3 -m dynamo.vllm"},
		},
		{
			name:          "decode worker multinode worker",
			componentType: commonconsts.ComponentTypeDecodeWorker,
			numberOfNodes: 3,
			role:          RoleWorker,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectedArgs:            []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"},
			expectContains:          []string{"ray start --address"},
		},
		{
			name:          "unknown component type defaults to worker",
			componentType: "unknown",
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.vllm"},
		},
		{
			name:          "with flag overrides",
			componentType: commonconsts.ComponentTypeWorker,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						FlagOverrides: map[string]*string{
							"custom-flag": ptr.To("custom-value"),
						},
					},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"custom-flag", "custom-value"},
		},
		{
			name:          "with extra args",
			componentType: commonconsts.ComponentTypeWorker,
			numberOfNodes: 1,
			role:          RoleMain,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						ExtraArgs: []string{"--extra", "arg"},
					},
				},
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"--extra", "arg"},
		},
		{
			name:                    "nil dynamo config",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			component:               &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.vllm"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			cmd, args := backend.GenerateCommandAndArgs(tt.componentType, tt.numberOfNodes, tt.role, tt.component, tt.multinodeDeploymentType)

			// Check command
			g.Expect(cmd).To(gomega.Equal(tt.expectedCmd))

			// Check args
			if tt.expectedArgs != nil {
				g.Expect(args).To(gomega.Equal(tt.expectedArgs))
			}

			// Check that expected strings are contained in the result
			if tt.expectContains != nil {
				argsStr := strings.Join(args, " ")
				for _, expected := range tt.expectContains {
					if !strings.Contains(argsStr, expected) {
						t.Errorf("GenerateCommandAndArgs() args = %v, should contain %s", args, expected)
					}
				}
			}
		})
	}
}

func TestVLLMBackend_MergeArgs(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                    string
		defaultArgs             []string
		userArgs                []string
		multinode               bool
		role                    Role
		componentType           string
		numberOfNodes           int32
		multinodeDeploymentType consts.MultinodeDeploymentType
		component               *v1alpha1.DynamoComponentDeploymentOverridesSpec
		expectedResult          []string
	}{
		{
			name:           "no user args returns default",
			defaultArgs:    []string{"default", "args"},
			userArgs:       []string{},
			multinode:      false,
			role:           RoleMain,
			expectedResult: []string{"default", "args"},
		},
		{
			name:           "single node returns user args",
			defaultArgs:    []string{"default", "args"},
			userArgs:       []string{"user", "args"},
			multinode:      false,
			role:           RoleMain,
			expectedResult: []string{"user", "args"},
		},
		{
			name:                    "multinode leader prepends ray start",
			defaultArgs:             []string{"default", "args"},
			userArgs:                []string{"user", "args"},
			multinode:               true,
			role:                    RoleLeader,
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           3,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						TensorParallelSize: ptr.To(int32(2)),
						DataParallelSize:   ptr.To(int32(3)),
						FlagOverrides: map[string]*string{
							"custom-flag": ptr.To("custom-value"),
						},
						ExtraArgs: []string{"--extra", "arg"},
					},
				},
			},
			expectedResult: []string{"ray start --head --port=6379 && user args --custom-flag custom-value --data-parallel-size 3 --tensor-parallel-size 2 --extra arg"},
		},
		{
			name:                    "multinode worker returns ray start block",
			defaultArgs:             []string{"default", "args"},
			userArgs:                []string{"user", "args"},
			multinode:               true,
			role:                    RoleWorker,
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           3,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						TensorParallelSize: ptr.To(int32(2)),
						DataParallelSize:   ptr.To(int32(3)),
						FlagOverrides: map[string]*string{
							"custom-flag": ptr.To("custom-value"),
						},
						ExtraArgs: []string{"--extra", "arg"},
					},
				},
			},
			expectedResult: []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"},
		},
		{
			name:                    "multinode main role returns user args",
			defaultArgs:             []string{"default", "args"},
			userArgs:                []string{"user", "args"},
			multinode:               true,
			role:                    RoleMain,
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           3,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						TensorParallelSize: ptr.To(int32(2)),
						DataParallelSize:   ptr.To(int32(3)),
						FlagOverrides: map[string]*string{
							"custom-flag": ptr.To("custom-value"),
						},
						ExtraArgs: []string{"--extra", "arg"},
					},
				},
			},
			expectedResult: []string{"user", "args"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			result := backend.MergeArgs(tt.defaultArgs, tt.userArgs, tt.multinode, tt.role, tt.componentType, tt.numberOfNodes, tt.component, tt.multinodeDeploymentType)

			if tt.expectedResult != nil {
				g.Expect(result).To(gomega.Equal(tt.expectedResult))
			}
		})
	}
}

func TestBuildVLLMArgs(t *testing.T) {
	tests := []struct {
		name              string
		dynamoConfig      *v1alpha1.DynamoConfig
		expectContains    []string
		expectNotContains []string
	}{
		{
			name:         "nil configs",
			dynamoConfig: nil,
		},
		{
			name:              "empty model name",
			dynamoConfig:      &v1alpha1.DynamoConfig{},
			expectNotContains: []string{"model"},
		},
		{
			name: "with flag overrides",
			dynamoConfig: &v1alpha1.DynamoConfig{
				FlagOverrides: map[string]*string{
					"custom": ptr.To("value"),
				},
			},
			expectContains: []string{"custom", "value"},
		},
		{
			name: "with extra args",
			dynamoConfig: &v1alpha1.DynamoConfig{
				ExtraArgs: []string{"--extra", "arg"},
			},
			expectContains: []string{"--extra", "arg"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildVLLMArgs(tt.dynamoConfig)
			resultStr := strings.Join(result, " ")

			for _, expected := range tt.expectContains {
				if !strings.Contains(resultStr, expected) {
					t.Errorf("buildVLLMArgs() = %v, should contain %s", result, expected)
				}
			}

			for _, notExpected := range tt.expectNotContains {
				if strings.Contains(resultStr, notExpected) {
					t.Errorf("buildVLLMArgs() = %v, should NOT contain %s", result, notExpected)
				}
			}
		})
	}
}
