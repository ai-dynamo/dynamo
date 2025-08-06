package dynamo

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func TestTRTLLMBackend_UpdateContainer(t *testing.T) {
	tests := []struct {
		name                    string
		numberOfNodes           int32
		role                    Role
		multinodeDeploymentType commonconsts.MultinodeDeploymentType
		component               *v1alpha1.DynamoComponentDeploymentOverridesSpec
		expectedVolumeMounts    int
		expectedCommand         []string
		shouldContainMpirun     bool
		shouldContainSSHSetup   bool
		shouldContainSSHDaemon  bool
		shouldContainTotalGPUs  bool
	}{
		{
			name:                    "Single node - no changes",
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			component:               &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expectedVolumeMounts:    0,
			expectedCommand:         nil,
			shouldContainMpirun:     false,
			shouldContainSSHSetup:   false,
			shouldContainSSHDaemon:  false,
			shouldContainTotalGPUs:  false,
		},
		{
			name:                    "Multinode leader with GPU resources",
			numberOfNodes:           3,
			role:                    RoleLeader,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "2",
						},
					},
				},
			},
			expectedVolumeMounts:   1,
			expectedCommand:        []string{"/bin/sh", "-c"},
			shouldContainMpirun:    true,
			shouldContainSSHSetup:  true,
			shouldContainSSHDaemon: false,
			shouldContainTotalGPUs: true,
		},
		{
			name:                    "Multinode worker",
			numberOfNodes:           3,
			role:                    RoleWorker,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			component:               &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expectedVolumeMounts:    1,
			expectedCommand:         []string{"/bin/sh", "-c"},
			shouldContainMpirun:     false,
			shouldContainSSHSetup:   true,
			shouldContainSSHDaemon:  true,
			shouldContainTotalGPUs:  false,
		},
		{
			name:                    "Multinode leader with LWS deployment",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeLWS,
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			expectedVolumeMounts:   1,
			expectedCommand:        []string{"/bin/sh", "-c"},
			shouldContainMpirun:    true,
			shouldContainSSHSetup:  true,
			shouldContainSSHDaemon: false,
			shouldContainTotalGPUs: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			container := &corev1.Container{
				Args:           []string{"trtllm-llmapi-launch", "--model", "test"},
				LivenessProbe:  &corev1.Probe{},
				ReadinessProbe: &corev1.Probe{},
				StartupProbe:   &corev1.Probe{},
			}

			// Call UpdateContainer
			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, tt.component, tt.multinodeDeploymentType, "test-service")

			// Check volume mounts
			if len(container.VolumeMounts) != tt.expectedVolumeMounts {
				t.Errorf("UpdateContainer() volume mounts = %d, want %d", len(container.VolumeMounts), tt.expectedVolumeMounts)
			}

			if tt.expectedVolumeMounts > 0 {
				found := false
				for _, vm := range container.VolumeMounts {
					if vm.Name == "ssh-keypair" && vm.MountPath == "/ssh-pk" && vm.ReadOnly {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("UpdateContainer() should add ssh-keypair volume mount")
				}
			}

			// Check command
			if tt.expectedCommand != nil {
				if len(container.Command) != len(tt.expectedCommand) {
					t.Errorf("UpdateContainer() command length = %d, want %d", len(container.Command), len(tt.expectedCommand))
				} else {
					for i, cmd := range tt.expectedCommand {
						if container.Command[i] != cmd {
							t.Errorf("UpdateContainer() command[%d] = %s, want %s", i, container.Command[i], cmd)
						}
					}
				}
			}

			// Check args content
			if len(container.Args) > 0 {
				argsStr := strings.Join(container.Args, " ")

				if tt.shouldContainMpirun && !strings.Contains(argsStr, "mpirun") {
					t.Errorf("UpdateContainer() args should contain mpirun but got: %s", argsStr)
				}

				if tt.shouldContainSSHSetup && !strings.Contains(argsStr, "mkdir -p ~/.ssh") {
					t.Errorf("UpdateContainer() args should contain SSH setup but got: %s", argsStr)
				}

				if tt.shouldContainSSHDaemon && !strings.Contains(argsStr, "/usr/sbin/sshd") {
					t.Errorf("UpdateContainer() args should contain SSH daemon but got: %s", argsStr)
				}

				if tt.shouldContainTotalGPUs && !strings.Contains(argsStr, "-n ") {
					t.Errorf("UpdateContainer() args should contain GPU count (-n) but got: %s", argsStr)
				}

				if !tt.shouldContainMpirun && strings.Contains(argsStr, "mpirun") {
					t.Errorf("UpdateContainer() args should not contain mpirun but got: %s", argsStr)
				}
			}

			// Check that probes are removed for multinode
			if tt.numberOfNodes > 1 && (tt.role == RoleLeader || tt.role == RoleWorker) {
				if container.LivenessProbe != nil {
					t.Errorf("UpdateContainer() should remove LivenessProbe for multinode %s", tt.role)
				}
				if container.ReadinessProbe != nil {
					t.Errorf("UpdateContainer() should remove ReadinessProbe for multinode %s", tt.role)
				}
				if container.StartupProbe != nil {
					t.Errorf("UpdateContainer() should remove StartupProbe for multinode %s", tt.role)
				}
			}
		})
	}
}

func TestTRTLLMBackend_UpdatePodSpec(t *testing.T) {
	tests := []struct {
		name                    string
		numberOfNodes           int32
		role                    Role
		multinodeDeploymentType commonconsts.MultinodeDeploymentType
		initialVolumes          []corev1.Volume
		expectedVolumeCount     int
		shouldHaveSSHVolume     bool
	}{
		{
			name:                    "Single node - no SSH volume added",
			numberOfNodes:           1,
			role:                    RoleMain,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			initialVolumes:          []corev1.Volume{},
			expectedVolumeCount:     0,
			shouldHaveSSHVolume:     false,
		},
		{
			name:                    "Multinode leader - SSH volume added",
			numberOfNodes:           3,
			role:                    RoleLeader,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			initialVolumes:          []corev1.Volume{},
			expectedVolumeCount:     1,
			shouldHaveSSHVolume:     true,
		},
		{
			name:                    "Multinode worker - SSH volume added",
			numberOfNodes:           2,
			role:                    RoleWorker,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeLWS,
			initialVolumes:          []corev1.Volume{},
			expectedVolumeCount:     1,
			shouldHaveSSHVolume:     true,
		},
		{
			name:                    "Multinode with existing volumes",
			numberOfNodes:           2,
			role:                    RoleLeader,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			initialVolumes: []corev1.Volume{
				{Name: "existing-volume"},
			},
			expectedVolumeCount: 2,
			shouldHaveSSHVolume: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			podSpec := &corev1.PodSpec{
				Volumes: tt.initialVolumes,
			}
			component := &v1alpha1.DynamoComponentDeploymentOverridesSpec{}

			// Call UpdatePodSpec
			backend.UpdatePodSpec(podSpec, tt.numberOfNodes, tt.role, component, tt.multinodeDeploymentType, "test-service")

			// Check volume count
			if len(podSpec.Volumes) != tt.expectedVolumeCount {
				t.Errorf("UpdatePodSpec() volume count = %d, want %d", len(podSpec.Volumes), tt.expectedVolumeCount)
			}

			// Check for SSH volume
			hasSSHVolume := false
			for _, volume := range podSpec.Volumes {
				if volume.Name == "ssh-keypair" {
					hasSSHVolume = true
					// Verify volume configuration
					if volume.VolumeSource.Secret == nil {
						t.Errorf("UpdatePodSpec() SSH volume should use Secret volume source")
					} else {
						if volume.VolumeSource.Secret.SecretName != "ssh-keypair-secret" {
							t.Errorf("UpdatePodSpec() SSH volume secret name = %s, want ssh-keypair-secret", volume.VolumeSource.Secret.SecretName)
						}
						if volume.VolumeSource.Secret.DefaultMode == nil || *volume.VolumeSource.Secret.DefaultMode != 0644 {
							t.Errorf("UpdatePodSpec() SSH volume should have DefaultMode 0644")
						}
					}
					break
				}
			}

			if tt.shouldHaveSSHVolume && !hasSSHVolume {
				t.Errorf("UpdatePodSpec() should add SSH volume for multinode deployment")
			}

			if !tt.shouldHaveSSHVolume && hasSSHVolume {
				t.Errorf("UpdatePodSpec() should not add SSH volume for single node deployment")
			}
		})
	}
}

func TestTRTLLMBackend_generateWorkerHostnames(t *testing.T) {
	tests := []struct {
		name                    string
		numberOfNodes           int32
		multinodeDeploymentType commonconsts.MultinodeDeploymentType
		serviceName             string
		expectedContains        []string
		expectedNodeCount       int32
	}{
		{
			name:                    "Grove deployment with 3 nodes",
			numberOfNodes:           3,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			serviceName:             "test-service",
			expectedContains: []string{
				"test-service-ldr-0",
				"test-service-wkr-0",
				"test-service-wkr-1",
				"GROVE_PCSG_NAME",
				"GROVE_HEADLESS_SERVICE",
			},
			expectedNodeCount: 3,
		},
		{
			name:                    "LWS deployment with 2 nodes",
			numberOfNodes:           2,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeLWS,
			serviceName:             "test-service",
			expectedContains: []string{
				"${LWS_LEADER_ADDRESS}",
				"${LWS_WORKER_1_ADDRESS}",
			},
			expectedNodeCount: 2,
		},
		{
			name:                    "Grove deployment with 5 nodes",
			numberOfNodes:           5,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			serviceName:             "worker",
			expectedContains: []string{
				"worker-ldr-0",
				"worker-wkr-0",
				"worker-wkr-1",
				"worker-wkr-2",
				"worker-wkr-3",
			},
			expectedNodeCount: 5,
		},
		{
			name:                    "LWS deployment with 4 nodes",
			numberOfNodes:           4,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeLWS,
			serviceName:             "worker",
			expectedContains: []string{
				"${LWS_LEADER_ADDRESS}",
				"${LWS_WORKER_1_ADDRESS}",
				"${LWS_WORKER_2_ADDRESS}",
				"${LWS_WORKER_3_ADDRESS}",
			},
			expectedNodeCount: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			result := backend.generateWorkerHostnames(tt.numberOfNodes, tt.multinodeDeploymentType, tt.serviceName)

			for _, expected := range tt.expectedContains {
				if !strings.Contains(result, expected) {
					t.Errorf("generateWorkerHostnames() = %s, should contain %s", result, expected)
				}
			}

			// Check that result is comma-separated with correct count
			parts := strings.Split(result, ",")
			if int32(len(parts)) != tt.expectedNodeCount {
				t.Errorf("generateWorkerHostnames() should have %d hostnames, got %d: %v", tt.expectedNodeCount, len(parts), parts)
			}

			// Verify no empty parts
			for i, part := range parts {
				if strings.TrimSpace(part) == "" {
					t.Errorf("generateWorkerHostnames() has empty hostname at index %d", i)
				}
			}
		})
	}
}

func TestTRTLLMBackend_addSSHVolumeMount(t *testing.T) {
	tests := []struct {
		name                     string
		initialVolumeMounts      []corev1.VolumeMount
		expectedVolumeMountCount int
	}{
		{
			name:                     "Add SSH volume mount to empty container",
			initialVolumeMounts:      []corev1.VolumeMount{},
			expectedVolumeMountCount: 1,
		},
		{
			name: "Add SSH volume mount to container with existing mounts",
			initialVolumeMounts: []corev1.VolumeMount{
				{Name: "existing-mount", MountPath: "/existing"},
			},
			expectedVolumeMountCount: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			container := &corev1.Container{
				VolumeMounts: tt.initialVolumeMounts,
			}

			backend.addSSHVolumeMount(container)

			// Check volume mount count
			if len(container.VolumeMounts) != tt.expectedVolumeMountCount {
				t.Errorf("addSSHVolumeMount() volume mount count = %d, want %d", len(container.VolumeMounts), tt.expectedVolumeMountCount)
			}

			// Find and verify SSH volume mount
			found := false
			for _, vm := range container.VolumeMounts {
				if vm.Name == "ssh-keypair" {
					found = true
					if vm.MountPath != "/ssh-pk" {
						t.Errorf("addSSHVolumeMount() mount path = %s, want /ssh-pk", vm.MountPath)
					}
					if !vm.ReadOnly {
						t.Errorf("addSSHVolumeMount() ReadOnly should be true")
					}
					break
				}
			}

			if !found {
				t.Errorf("addSSHVolumeMount() should add ssh-keypair volume mount")
			}
		})
	}
}

func TestTRTLLMBackend_setupLeaderContainer(t *testing.T) {
	tests := []struct {
		name                    string
		numberOfNodes           int32
		multinodeDeploymentType commonconsts.MultinodeDeploymentType
		serviceName             string
		component               *v1alpha1.DynamoComponentDeploymentOverridesSpec
		initialArgs             []string
		initialCommand          []string
		expectedContains        []string
		expectedNotContains     []string
	}{
		{
			name:                    "Leader with args and GPU resources",
			numberOfNodes:           3,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			serviceName:             "test-service",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "2",
						},
					},
				},
			},
			initialArgs:    []string{"trtllm-llmapi-launch", "--model", "test"},
			initialCommand: []string{},
			expectedContains: []string{
				"mkdir -p ~/.ssh",
				"mpirun --oversubscribe",
				"-n 6", // 3 nodes * 2 GPUs
				"trtllm-llmapi-launch --model test",
				"test-service-ldr-0",
			},
			expectedNotContains: []string{},
		},
		{
			name:                    "Leader with command and no GPU resources",
			numberOfNodes:           2,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeLWS,
			serviceName:             "worker",
			component:               &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			initialArgs:             []string{},
			initialCommand:          []string{"python", "-m", "worker"},
			expectedContains: []string{
				"mkdir -p ~/.ssh",
				"mpirun --oversubscribe",
				"-n 0", // no GPU resources specified
				"python -m worker",
				"${LWS_LEADER_ADDRESS}",
			},
			expectedNotContains: []string{},
		},
		{
			name:                    "Leader with both command and args (args take precedence)",
			numberOfNodes:           2,
			multinodeDeploymentType: commonconsts.MultinodeDeploymentTypeGrove,
			serviceName:             "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"launch", "--config", "test.yaml"},
			initialCommand: []string{"ignored-command"},
			expectedContains: []string{
				"mpirun --oversubscribe",
				"-n 2", // 2 nodes * 1 GPU
				"launch --config test.yaml",
			},
			expectedNotContains: []string{
				"ignored-command",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			container := &corev1.Container{
				Args:    tt.initialArgs,
				Command: tt.initialCommand,
			}

			backend.setupLeaderContainer(container, tt.numberOfNodes, tt.multinodeDeploymentType, tt.serviceName, tt.component)

			// Check that command is set correctly
			expectedCommand := []string{"/bin/sh", "-c"}
			if len(container.Command) != len(expectedCommand) {
				t.Errorf("setupLeaderContainer() command = %v, want %v", container.Command, expectedCommand)
			} else {
				for i, cmd := range expectedCommand {
					if container.Command[i] != cmd {
						t.Errorf("setupLeaderContainer() command[%d] = %s, want %s", i, container.Command[i], cmd)
					}
				}
			}

			// Check args content
			if len(container.Args) != 1 {
				t.Errorf("setupLeaderContainer() should set exactly one arg, got %d", len(container.Args))
			} else {
				argsStr := container.Args[0]

				for _, expected := range tt.expectedContains {
					if !strings.Contains(argsStr, expected) {
						t.Errorf("setupLeaderContainer() args should contain %q but got: %s", expected, argsStr)
					}
				}

				for _, notExpected := range tt.expectedNotContains {
					if strings.Contains(argsStr, notExpected) {
						t.Errorf("setupLeaderContainer() args should not contain %q but got: %s", notExpected, argsStr)
					}
				}
			}
		})
	}
}

func TestTRTLLMBackend_setupWorkerContainer(t *testing.T) {
	tests := []struct {
		name             string
		initialArgs      []string
		initialCommand   []string
		expectedContains []string
	}{
		{
			name:           "Worker setup with initial args",
			initialArgs:    []string{"some", "args"},
			initialCommand: []string{},
			expectedContains: []string{
				"mkdir -p ~/.ssh",
				"cp /ssh-pk/private.key ~/.ssh/id_rsa",
				"chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys",
				"/usr/sbin/sshd -D",
				"# Start SSH daemon and keep container running",
			},
		},
		{
			name:           "Worker setup with initial command",
			initialArgs:    []string{},
			initialCommand: []string{"original", "command"},
			expectedContains: []string{
				"mkdir -p ~/.ssh",
				"cp /ssh-pk/private.key ~/.ssh/id_rsa",
				"chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys",
				"/usr/sbin/sshd -D",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			container := &corev1.Container{
				Args:    tt.initialArgs,
				Command: tt.initialCommand,
			}

			backend.setupWorkerContainer(container)

			// Check that command is set correctly
			expectedCommand := []string{"/bin/sh", "-c"}
			if len(container.Command) != len(expectedCommand) {
				t.Errorf("setupWorkerContainer() command = %v, want %v", container.Command, expectedCommand)
			} else {
				for i, cmd := range expectedCommand {
					if container.Command[i] != cmd {
						t.Errorf("setupWorkerContainer() command[%d] = %s, want %s", i, container.Command[i], cmd)
					}
				}
			}

			// Check args content
			if len(container.Args) != 1 {
				t.Errorf("setupWorkerContainer() should set exactly one arg, got %d", len(container.Args))
			} else {
				argsStr := container.Args[0]

				for _, expected := range tt.expectedContains {
					if !strings.Contains(argsStr, expected) {
						t.Errorf("setupWorkerContainer() args should contain %q but got: %s", expected, argsStr)
					}
				}

				// Verify that the command ends with SSH daemon
				if !strings.HasSuffix(argsStr, "/usr/sbin/sshd -D") {
					t.Errorf("setupWorkerContainer() should end with SSH daemon command, got: %s", argsStr)
				}
			}
		})
	}
}

func TestTRTLLMBackend_getGPUsPerNode(t *testing.T) {
	tests := []struct {
		name      string
		resources *common.Resources
		expected  int32
	}{
		{
			name:      "No resources - default to 0",
			resources: nil,
			expected:  0,
		},
		{
			name:      "Empty resources - default to 0",
			resources: &common.Resources{},
			expected:  0,
		},
		{
			name: "GPU in requests",
			resources: &common.Resources{
				Requests: &common.ResourceItem{
					GPU: "2",
				},
			},
			expected: 2,
		},
		{
			name: "GPU in limits",
			resources: &common.Resources{
				Limits: &common.ResourceItem{
					GPU: "4",
				},
			},
			expected: 4,
		},
		{
			name: "GPU in both requests and limits - requests takes precedence",
			resources: &common.Resources{
				Requests: &common.ResourceItem{
					GPU: "3",
				},
				Limits: &common.ResourceItem{
					GPU: "8",
				},
			},
			expected: 3,
		},
		{
			name: "Invalid GPU value - default to 0",
			resources: &common.Resources{
				Requests: &common.ResourceItem{
					GPU: "invalid",
				},
			},
			expected: 0,
		},
		{
			name: "Empty GPU string - default to 0",
			resources: &common.Resources{
				Requests: &common.ResourceItem{
					GPU: "",
				},
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getGPUsPerNode(tt.resources)
			if result != tt.expected {
				t.Errorf("getGPUsPerNode() = %d, want %d", result, tt.expected)
			}
		})
	}
}
