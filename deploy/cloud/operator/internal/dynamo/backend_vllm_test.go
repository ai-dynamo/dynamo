package dynamo

import (
	"fmt"
	"reflect"
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
)

func TestVLLMBackend_UpdateContainer(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                string
		numberOfNodes       int32
		role                Role
		component           *v1alpha1.DynamoComponentDeploymentSharedSpec
		multinodeDeployer   MultinodeDeployer
		initialContainer    *corev1.Container
		gpuCount            int64 // GPU count for the test case
		expectedArgs        []string
		expectNotModified   bool // If true, container args should not change
		expectProbesRemoved bool // If true, probes should be nil
	}{
		{
			name:              "single node does not modify args",
			numberOfNodes:     1,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm"}},
			gpuCount:          0,
			expectNotModified: true,
		},
		{
			name:                "multinode leader prepends ray start --head",
			numberOfNodes:       3,
			role:                RoleLeader,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialContainer:    &corev1.Container{Command: []string{"python3", "-m", "dynamo.vllm"}, Args: []string{"--model", "test", tensorParallelSizeFlag, "8"}},
			gpuCount:            4,
			expectedArgs:        []string{fmt.Sprintf("ray start --head --port=%s && python3 -m dynamo.vllm --model test %s 8", VLLMPort, tensorParallelSizeFlag)},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker replaces args with ray start --block",
			numberOfNodes:       3,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialContainer:    &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", "--model", "test", tensorParallelSizeFlag, "8"}},
			gpuCount:            4,
			expectedArgs:        []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker with LWS deployment type",
			numberOfNodes:       2,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &LWSMultinodeDeployer{},
			initialContainer:    &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "8"}},
			gpuCount:            4,
			expectedArgs:        []string{"ray start --address=$LWS_LEADER_ADDRESS:6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:              "multinode leader with no initial args",
			numberOfNodes:     2,
			role:              RoleLeader,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{}},
			gpuCount:          0,
			expectNotModified: true, // Should not modify empty args
		},
		{
			name:              "multinode main role (non-leader/worker) does not modify args",
			numberOfNodes:     3,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.frontend"}},
			gpuCount:          0,
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			initialContainerArgs := append([]string{}, tt.initialContainer.Args...)

			// Create resources from GPU count and set in component
			if tt.gpuCount > 0 {
				tt.component.Resources = &v1alpha1.Resources{
					Limits: &v1alpha1.ResourceItem{
						GPU: strconv.FormatInt(tt.gpuCount, 10),
					},
				}
			}

			// Call UpdateContainer
			backend.UpdateContainer(tt.initialContainer, tt.numberOfNodes, tt.role, tt.component, "test-service", tt.multinodeDeployer)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(initialContainerArgs))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(tt.expectedArgs))
			}

			if tt.expectProbesRemoved {
				g.Expect(tt.initialContainer.LivenessProbe).To(gomega.BeNil())
				g.Expect(tt.initialContainer.ReadinessProbe).To(gomega.BeNil())
				g.Expect(tt.initialContainer.StartupProbe).To(gomega.BeNil())
			}
		})
	}
}

func TestVLLMBackend_ShellCommandInjection(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name              string
		numberOfNodes     int32
		role              Role
		multinodeDeployer MultinodeDeployer
		initialContainer  *corev1.Container
		gpuCount          int64 // GPU count for the test case
		expectedArgs      []string
		description       string
	}{
		{
			name:              "single node shell command not modified",
			numberOfNodes:     1,
			role:              RoleMain,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"sh", "-c"}, Args: []string{"python3 -m dynamo.vllm"}},
			gpuCount:          0,
			expectedArgs:      []string{"python3 -m dynamo.vllm"},
			description:       "Single node should not modify shell commands",
		},
		{
			name:              "multinode shell command with regex injection",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"sh", "-c"}, Args: []string{fmt.Sprintf("python3 -m dynamo.vllm %s 8", dataParallelSizeFlag)}},
			gpuCount:          4,
			expectedArgs:      []string{"python3 -m dynamo.vllm --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0 --data-parallel-size 8"},
			description:       "Shell commands should use regex injection for python commands",
		},
		{
			name:              "multinode shell command with complex pipeline",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"sh", "-c"}, Args: []string{fmt.Sprintf("echo blah | wc -l && python3 -m dynamo.vllm %s 8 && ls -al", dataParallelSizeFlag)}},
			gpuCount:          4,
			expectedArgs:      []string{"echo blah | wc -l && python3 -m dynamo.vllm --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0 --data-parallel-size 8 && ls -al"},
			description:       "Complex shell commands should inject flags only into python part",
		},
		{
			name:              "shell command with LWS deployer",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &LWSMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"sh", "-c"}, Args: []string{fmt.Sprintf("python3 -m dynamo.vllm %s 8", dataParallelSizeFlag)}},
			gpuCount:          4,
			expectedArgs:      []string{"python3 -m dynamo.vllm --data-parallel-address $LWS_LEADER_ADDRESS --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0 --data-parallel-size 8"},
			description:       "LWS shell commands should use LWS variables",
		},
		{
			name:              "shell command with pipes",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"sh", "-c"}, Args: []string{fmt.Sprintf("python3 -m dynamo.vllm %s 8 | tee /tmp/log", dataParallelSizeFlag)}},
			gpuCount:          4,
			expectedArgs:      []string{"python3 -m dynamo.vllm --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0 --data-parallel-size 8 | tee /tmp/log"},
			description:       "Shell commands with pipes should inject flags before pipe",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expectedCommand := append([]string{}, tt.initialContainer.Command...)

			// Create component with resources from GPU count
			component := &v1alpha1.DynamoComponentDeploymentSharedSpec{}
			if tt.gpuCount > 0 {
				component.Resources = &v1alpha1.Resources{
					Limits: &v1alpha1.ResourceItem{
						GPU: strconv.FormatInt(tt.gpuCount, 10),
					},
				}
			}

			backend.UpdateContainer(tt.initialContainer, tt.numberOfNodes, tt.role, component, "test-service", tt.multinodeDeployer)

			if !reflect.DeepEqual(tt.initialContainer.Args, tt.expectedArgs) {
				t.Errorf("UpdateContainer() args = %v, want %v", tt.initialContainer.Args, tt.expectedArgs)
			}

			if !reflect.DeepEqual(tt.initialContainer.Command, expectedCommand) {
				t.Errorf("UpdateContainer() should preserve shell command, got: %v, want: %v", tt.initialContainer.Command, expectedCommand)
			}
		})
	}
}

func TestVLLMBackend_UpdateContainer_UseAsCompilationCache(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                  string
		component             *v1alpha1.DynamoComponentDeploymentSharedSpec
		volumeMounts          []corev1.VolumeMount
		expectCacheEnvVar     bool
		expectCacheEnvVarName string
		expectCacheEnvVarVal  string
	}{
		{
			name: "VLLM backend with useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "vllm-cache",
						MountPoint:            "/root/.cache/vllm",
						UseAsCompilationCache: true,
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
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "custom-cache",
						MountPoint:            "/custom/cache/path",
						UseAsCompilationCache: true,
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
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:       "regular-volume",
						MountPoint: "/data",
					},
				},
			},
			volumeMounts:      []corev1.VolumeMount{},
			expectCacheEnvVar: false,
		},
		{
			name: "VLLM backend with no volume mounts",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: nil,
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
		initialContainer  *corev1.Container
		gpuCount          int64 // GPU count for the test case
		expectedArgs      []string
		expectNotModified bool
	}{
		{
			name:              "leader prepends ray start --head",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}},
			gpuCount:          8,
			expectedArgs:      []string{fmt.Sprintf("ray start --head --port=%s && python3 -m dynamo.vllm %s 16", VLLMPort, tensorParallelSizeFlag)},
		},
		{
			name:              "leader prepends distributed data parallel flags",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "16"}},
			gpuCount:          8,
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 16 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 8 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0", dataParallelSizeFlag)},
		},
		{
			name:              "leader with empty args does not modify",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{}},
			gpuCount:          0,
			expectNotModified: true,
		},
		{
			name:              "worker with ray distributed launch Grove",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}},
			gpuCount:          8,
			expectedArgs:      []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
		},
		{
			name:              "worker with data parallel launch Grove",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "16"}},
			gpuCount:          8,
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 16 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 8 --data-parallel-rpc-port 13445 --data-parallel-start-rank $(( 8 * $((GROVE_PCLQ_POD_INDEX + 1)) ))", dataParallelSizeFlag)},
		},
		{
			name:              "worker with data parallel launch Grove, tp > 1",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "8", tensorParallelSizeFlag, "2"}},
			gpuCount:          8,
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 8 %s 2 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank $(( 4 * $((GROVE_PCLQ_POD_INDEX + 1)) ))", dataParallelSizeFlag, tensorParallelSizeFlag)},
		},
		{
			name:              "worker with ray distributed launch LWS",
			role:              RoleWorker,
			multinodeDeployer: &LWSMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}},
			gpuCount:          8,
			expectedArgs:      []string{"ray start --address=$LWS_LEADER_ADDRESS:6379 --block"},
		},
		{
			name:              "main role does not modify args",
			role:              RoleMain,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.frontend"}},
			gpuCount:          0,
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			initialContainerArgs := append([]string{}, tt.initialContainer.Args...)

			// Create resources from GPU count
			var resources *v1alpha1.Resources
			if tt.gpuCount > 0 {
				resources = &v1alpha1.Resources{
					Limits: &v1alpha1.ResourceItem{
						GPU: strconv.FormatInt(tt.gpuCount, 10),
					},
				}
			}

			// Call updateVLLMMultinodeArgs
			updateVLLMMultinodeArgs(tt.initialContainer, tt.role, "test-service", tt.multinodeDeployer, resources)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(initialContainerArgs))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(tt.expectedArgs))
			}
		})
	}
}
