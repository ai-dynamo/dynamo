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

func TestSGLangBackend_GenerateCommandAndArgs(t *testing.T) {
	backend := &SGLangBackend{}

	tests := []struct {
		name                    string
		componentType           string
		numberOfNodes           int32
		role                    Role
		component               *v1alpha1.DynamoComponentDeploymentOverridesSpec
		multinodeDeploymentType consts.MultinodeDeploymentType
		expectedCmd             []string
		expectedArgs            []string
		expectContains          []string // For checking if specific strings are in the result
	}{
		{
			name:          "main component single node",
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
			expectedArgs:            []string{"python3 -m dynamo.frontend --http-port=8000"},
			expectContains:          []string{"python3 -m dynamo.frontend"},
		},
		{
			name:          "worker component single node",
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
			expectContains:          []string{"python3 -m dynamo.sglang.worker"},
		},
		{
			name:                    "decode worker component single node",
			componentType:           commonconsts.ComponentTypeDecodeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"python3 -m dynamo.sglang.decode_worker"},
		},
		{
			name:                    "worker component multinode leader",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           3,
			role:                    RoleLeader,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"python3 -m dynamo.sglang.worker", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name:                    "worker component multinode worker",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           3,
			role:                    RoleWorker,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"python3 -m dynamo.sglang.worker", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name:                    "with tensor parallel size",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						TensorParallelSize: ptr.To(int32(4)),
					},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"tp-size", "4"},
		},
		{
			name:                    "with data parallel size",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						DataParallelSize: ptr.To(int32(2)),
					},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"dp-size", "2"},
		},
		{
			name:                    "with flag overrides",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						FlagOverrides: map[string]*string{
							"custom-flag": ptr.To("custom-value"),
						},
					},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"custom-flag", "custom-value"},
		},
		{
			name:                    "with extra args",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					DynamoConfig: &v1alpha1.DynamoConfig{
						ExtraArgs: []string{"--extra", "arg"},
					},
				},
			},
			expectedCmd:    []string{"/bin/sh", "-c"},
			expectContains: []string{"--extra", "arg"},
		},
		{
			name:                    "nil dynamo config",
			componentType:           commonconsts.ComponentTypeWorker,
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			component:               &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expectedCmd:             []string{"/bin/sh", "-c"},
			expectContains:          []string{"python3 -m dynamo.sglang.worker"},
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

func TestSGLangBackend_MergeArgs(t *testing.T) {
	backend := &SGLangBackend{}

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
			name:                    "no user args returns default",
			defaultArgs:             []string{"default", "args"},
			userArgs:                []string{},
			multinode:               false,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedResult:          []string{"default", "args"},
		},
		{
			name:                    "single node returns user args",
			defaultArgs:             []string{"default", "args"},
			userArgs:                []string{"user", "args"},
			multinode:               false,
			role:                    RoleMain,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectedResult:          []string{"user", "args"},
		},
		{
			name:                    "multinode with user args",
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
			expectedResult: []string{"user args --custom-flag custom-value --dist-init-addr ${GROVE_HEADLESS_SERVICE}:29500 --dp-size 3 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1)) --tp-size 2 --extra arg"},
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

func TestBuildSGLangArgs(t *testing.T) {
	tests := []struct {
		name                    string
		numberOfNodes           int32
		role                    Role
		dynamoConfig            *v1alpha1.DynamoConfig
		multinodeDeploymentType consts.MultinodeDeploymentType
		expectContains          []string
		expectNotContains       []string
	}{
		{
			name:                    "single node no distributed flags",
			numberOfNodes:           1,
			role:                    RoleMain,
			dynamoConfig:            &v1alpha1.DynamoConfig{},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectNotContains:       []string{"dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name:                    "multinode leader",
			numberOfNodes:           3,
			role:                    RoleLeader,
			dynamoConfig:            &v1alpha1.DynamoConfig{},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectContains:          []string{"dist-init-addr", "nnodes", "node-rank", "0"},
		},
		{
			name:                    "multinode worker",
			numberOfNodes:           3,
			role:                    RoleWorker,
			dynamoConfig:            &v1alpha1.DynamoConfig{},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectContains:          []string{"dist-init-addr", "nnodes", "node-rank", "$((GROVE_PCLQ_POD_INDEX + 1))"},
		},
		{
			name:          "with tensor parallel",
			numberOfNodes: 1,
			role:          RoleMain,
			dynamoConfig: &v1alpha1.DynamoConfig{
				TensorParallelSize: ptr.To(int32(4)),
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectContains:          []string{"tp-size", "4"},
		},
		{
			name:          "with data parallel",
			numberOfNodes: 1,
			role:          RoleMain,
			dynamoConfig: &v1alpha1.DynamoConfig{
				DataParallelSize: ptr.To(int32(2)),
			},
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectContains:          []string{"dp-size", "2"},
		},
		{
			name:                    "nil config",
			numberOfNodes:           1,
			role:                    RoleMain,
			dynamoConfig:            nil,
			multinodeDeploymentType: consts.MultinodeDeploymentTypeGrove,
			expectNotContains:       []string{"tp-size", "dp-size"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := buildSGLangArgs(tt.numberOfNodes, tt.role, tt.dynamoConfig, tt.multinodeDeploymentType)
			resultStr := strings.Join(result, " ")

			for _, expected := range tt.expectContains {
				if !strings.Contains(resultStr, expected) {
					t.Errorf("buildSGLangArgs() = %v, should contain %s", result, expected)
				}
			}

			for _, notExpected := range tt.expectNotContains {
				if strings.Contains(resultStr, notExpected) {
					t.Errorf("buildSGLangArgs() = %v, should NOT contain %s", result, notExpected)
				}
			}
		})
	}
}
