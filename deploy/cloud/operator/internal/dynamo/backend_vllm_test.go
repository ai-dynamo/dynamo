package dynamo

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
)

func TestVLLMBackend_UpdateContainer(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                  string
		numberOfNodes         int32
		role                  Role
		component             *v1alpha1.DynamoComponentDeploymentOverridesSpec
		multinodeDeployer     MultinodeDeployer
		initialArgs           []string
		initialLivenessProbe  *corev1.Probe
		initialReadinessProbe *corev1.Probe
		initialStartupProbe   *corev1.Probe
		expectedArgs          []string
		expectContains        []string
		expectNotModified     bool // If true, container args should not change
		expectProbesRemoved   bool // If true, probes should be nil
	}{
		{
			name:              "single node does not modify args",
			numberOfNodes:     1,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.vllm"},
			expectNotModified: true,
		},
		{
			name:                "multinode leader prepends ray start --head",
			numberOfNodes:       3,
			role:                RoleLeader,
			component:           &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialArgs:         []string{"python3", "-m", "dynamo.vllm", "--model", "test"},
			expectContains:      []string{"ray start --head --port=6379 &&", "python3", "-m", "dynamo.vllm", "--model", "test"},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker replaces args with ray start --block",
			numberOfNodes:       3,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialArgs:         []string{"python3", "-m", "dynamo.vllm", "--model", "test"},
			expectedArgs:        []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker with LWS deployment type",
			numberOfNodes:       2,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer:   &LWSMultinodeDeployer{},
			initialArgs:         []string{"python3", "-m", "dynamo.vllm"},
			expectedArgs:        []string{"ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:              "multinode leader with no initial args",
			numberOfNodes:     2,
			role:              RoleLeader,
			component:         &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{},
			expectNotModified: true, // Should not modify empty args
		},
		{
			name:              "multinode main role (non-leader/worker) does not modify args",
			numberOfNodes:     3,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.frontend"},
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a container with initial state
			container := &corev1.Container{
				Args:           append([]string{}, tt.initialArgs...), // Copy slice to avoid modifying original
				LivenessProbe:  tt.initialLivenessProbe,
				ReadinessProbe: tt.initialReadinessProbe,
				StartupProbe:   tt.initialStartupProbe,
			}

			// Call UpdateContainer
			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, tt.component, "test-service", tt.multinodeDeployer)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(container.Args).To(gomega.Equal(tt.initialArgs))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(container.Args).To(gomega.Equal(tt.expectedArgs))
			} else if tt.expectContains != nil {
				// Check that expected strings are contained in the result
				argsStr := strings.Join(container.Args, " ")
				for _, expected := range tt.expectContains {
					if !strings.Contains(argsStr, expected) {
						t.Errorf("UpdateContainer() args = %v, should contain %s", container.Args, expected)
					}
				}
			}

			if tt.expectProbesRemoved {
				g.Expect(container.LivenessProbe).To(gomega.BeNil())
				g.Expect(container.ReadinessProbe).To(gomega.BeNil())
				g.Expect(container.StartupProbe).To(gomega.BeNil())
			}
		})
	}
}

func TestVLLMBackend_UpdateContainer_UseAsCompilationCache(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                  string
		component             *v1alpha1.DynamoComponentDeploymentOverridesSpec
		volumeMounts          []corev1.VolumeMount
		expectCacheEnvVar     bool
		expectCacheEnvVarName string
		expectCacheEnvVarVal  string
	}{
		{
			name: "VLLM backend with useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: []v1alpha1.VolumeMount{
						{
							Name:                  "vllm-cache",
							MountPoint:            "/root/.cache/vllm",
							UseAsCompilationCache: true,
						},
					},
				},
			},
			volumeMounts:          []corev1.VolumeMount{},
			expectCacheEnvVar:     true,
			expectCacheEnvVarName: "VLLM_CACHE_ROOT",
			expectCacheEnvVarVal:  "/root/.cache/vllm",
		},
		{
			name: "VLLM backend with useAsCompilationCache at custom mount point",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: []v1alpha1.VolumeMount{
						{
							Name:                  "custom-cache",
							MountPoint:            "/custom/cache/path",
							UseAsCompilationCache: true,
						},
					},
				},
			},
			volumeMounts:          []corev1.VolumeMount{},
			expectCacheEnvVar:     true,
			expectCacheEnvVarName: "VLLM_CACHE_ROOT",
			expectCacheEnvVarVal:  "/custom/cache/path",
		},
		{
			name: "VLLM backend without useAsCompilationCache",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: []v1alpha1.VolumeMount{
						{
							Name:       "regular-volume",
							MountPoint: "/data",
						},
					},
				},
			},
			volumeMounts:      []corev1.VolumeMount{},
			expectCacheEnvVar: false,
		},
		{
			name: "VLLM backend with no volume mounts",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: nil,
				},
			},
			volumeMounts:      []corev1.VolumeMount{},
			expectCacheEnvVar: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a container with initial state including volume mounts
			container := &corev1.Container{
				Env:          []corev1.EnvVar{},
				VolumeMounts: tt.volumeMounts,
			}

			// Call UpdateContainer
			backend.UpdateContainer(container, 1, RoleMain, tt.component, "test-service", &GroveMultinodeDeployer{})

			if tt.expectCacheEnvVar {
				// Check that the VLLM_CACHE_ROOT environment variable is set
				found := false
				for _, env := range container.Env {
					if env.Name == tt.expectCacheEnvVarName {
						found = true
						g.Expect(env.Value).To(gomega.Equal(tt.expectCacheEnvVarVal))
						break
					}
				}
				if !found {
					t.Errorf("Expected environment variable %s not found in container", tt.expectCacheEnvVarName)
				}
			} else {
				// Check that no cache environment variable is set
				for _, env := range container.Env {
					if env.Name == "VLLM_CACHE_ROOT" {
						t.Errorf("Unexpected environment variable VLLM_CACHE_ROOT found: %s", env.Value)
					}
				}
			}
		})
	}
}

func TestUpdateVLLMMultinodeArgs(t *testing.T) {
	tests := []struct {
		name              string
		role              Role
		multinodeDeployer MultinodeDeployer
		initialArgs       []string
		expectedArgs      []string
		expectContains    []string
		expectNotModified bool
	}{
		{
			name:              "leader prepends ray start --head",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.vllm"},
			expectContains:    []string{"ray start --head --port=6379 &&", "python3", "-m", "dynamo.vllm"},
		},
		{
			name:              "leader with empty args does not modify",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{},
			expectNotModified: true,
		},
		{
			name:              "worker with Grove deployment",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.vllm"},
			expectedArgs:      []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
		},
		{
			name:              "worker with LWS deployment",
			role:              RoleWorker,
			multinodeDeployer: &LWSMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.vllm"},
			expectedArgs:      []string{"ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"},
		},
		{
			name:              "main role does not modify args",
			role:              RoleMain,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialArgs:       []string{"python3", "-m", "dynamo.frontend"},
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a container with initial args
			container := &corev1.Container{
				Args: append([]string{}, tt.initialArgs...), // Copy slice to avoid modifying original
			}

			// Call updateVLLMMultinodeArgs
			updateVLLMMultinodeArgs(container, tt.role, "test-service", tt.multinodeDeployer)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(container.Args).To(gomega.Equal(tt.initialArgs))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(container.Args).To(gomega.Equal(tt.expectedArgs))
			} else if tt.expectContains != nil {
				// Check that expected strings are contained in the result
				argsStr := strings.Join(container.Args, " ")
				for _, expected := range tt.expectContains {
					if !strings.Contains(argsStr, expected) {
						t.Errorf("updateVLLMMultinodeArgs() args = %v, should contain %s", container.Args, expected)
					}
				}
			}
		})
	}
}
