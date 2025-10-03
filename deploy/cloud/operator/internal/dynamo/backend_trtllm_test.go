package dynamo

import (
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

const (
	mpiRunSecretName = "mpi-run-ssh-secret"
)

func TestTRTLLMBackend_UpdateContainer(t *testing.T) {
	tests := []struct {
		name                   string
		numberOfNodes          int32
		role                   Role
		multinodeDeployer      MultinodeDeployer
		component              *v1alpha1.DynamoComponentDeploymentOverridesSpec
		expectedVolumeMounts   []corev1.VolumeMount
		expectedCommand        []string
		expectedArgs           []string
		expectedEnv            []corev1.EnvVar
		expectLivenessRemoved  bool
		expectReadinessRemoved bool
		expectStartupRemoved   bool
		expectedReadinessProbe *corev1.Probe
	}{
		{
			name:                   "Single node - no changes",
			numberOfNodes:          1,
			role:                   RoleMain,
			multinodeDeployer:      &GroveMultinodeDeployer{},
			component:              &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expectedVolumeMounts:   []corev1.VolumeMount{},
			expectedCommand:        []string{},
			expectedArgs:           []string{"python3", "--model", "test"},
			expectedEnv:            []corev1.EnvVar{},
			expectLivenessRemoved:  false,
			expectReadinessRemoved: false,
			expectStartupRemoved:   false,
			expectedReadinessProbe: nil,
		},
		{
			name:              "Multinode leader with GPU resources",
			numberOfNodes:     3,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "2",
						},
					},
				},
			},
			expectedVolumeMounts: []corev1.VolumeMount{
				{Name: mpiRunSecretName, MountPath: "/ssh-pk", ReadOnly: true},
			},
			expectedCommand: []string{"/bin/sh", "-c"},
			expectedArgs:    []string{"mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 6 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-wkr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-wkr-1.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x OMPI_MCA_orte_keep_fqdn_hostnames -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python3 --model test'"},
			expectedEnv: []corev1.EnvVar{
				{Name: "OMPI_MCA_orte_keep_fqdn_hostnames", Value: "1"},
			},
			expectLivenessRemoved:  false,
			expectReadinessRemoved: false,
			expectStartupRemoved:   false,
			expectedReadinessProbe: nil,
		},
		{
			name:              "Multinode worker",
			numberOfNodes:     3,
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			component:         &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			expectedVolumeMounts: []corev1.VolumeMount{
				{Name: mpiRunSecretName, MountPath: "/ssh-pk", ReadOnly: true},
			},
			expectedCommand: []string{"/bin/sh", "-c"},
			expectedArgs:    []string{"mkdir -p ~/.ssh ~/.ssh/host_keys ~/.ssh/run && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && ssh-keygen -t rsa -f ~/.ssh/host_keys/ssh_host_rsa_key -N '' && ssh-keygen -t ecdsa -f ~/.ssh/host_keys/ssh_host_ecdsa_key -N '' && ssh-keygen -t ed25519 -f ~/.ssh/host_keys/ssh_host_ed25519_key -N '' && printf 'Port 2222\\nHostKey ~/.ssh/host_keys/ssh_host_rsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ecdsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ed25519_key\\nPidFile ~/.ssh/run/sshd.pid\\nPermitRootLogin yes\\nPasswordAuthentication no\\nPubkeyAuthentication yes\\nAuthorizedKeysFile ~/.ssh/authorized_keys\\n' > ~/.ssh/sshd_config && mkdir -p /run/sshd && /usr/sbin/sshd -D -f ~/.ssh/sshd_config"},
			expectedEnv: []corev1.EnvVar{
				{Name: "OMPI_MCA_orte_keep_fqdn_hostnames", Value: "1"},
			},
			expectLivenessRemoved:  true,
			expectReadinessRemoved: false,
			expectStartupRemoved:   true,
			expectedReadinessProbe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					TCPSocket: &corev1.TCPSocketAction{
						Port: intstr.FromInt(commonconsts.MpiRunSshPort),
					},
				},
				InitialDelaySeconds: 20,
				PeriodSeconds:       20,
				TimeoutSeconds:      5,
				FailureThreshold:    10,
			},
		},
		{
			name:              "Multinode leader with LWS deployment",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &LWSMultinodeDeployer{},
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			expectedVolumeMounts: []corev1.VolumeMount{
				{Name: mpiRunSecretName, MountPath: "/ssh-pk", ReadOnly: true},
			},
			expectedCommand: []string{"/bin/sh", "-c"},
			expectedArgs:    []string{"mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(LWS_LEADER_ADDRESS),$(LWS_WORKER_1_ADDRESS) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x OMPI_MCA_orte_keep_fqdn_hostnames -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python3 --model test'"},
			expectedEnv: []corev1.EnvVar{
				{Name: "OMPI_MCA_orte_keep_fqdn_hostnames", Value: "1"},
			},
			expectLivenessRemoved:  false,
			expectReadinessRemoved: false,
			expectStartupRemoved:   false,
			expectedReadinessProbe: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{
				MpiRunSecretName: mpiRunSecretName,
			}
			container := &corev1.Container{
				Args:           []string{"python3", "--model", "test"},
				LivenessProbe:  &corev1.Probe{},
				ReadinessProbe: &corev1.Probe{},
				StartupProbe:   &corev1.Probe{},
			}

			backend.UpdateContainer(container, tt.numberOfNodes, tt.role, tt.component, "test-service", tt.multinodeDeployer)

			validateVolumeMounts(t, container, tt.expectedVolumeMounts)
			validateCommand(t, container, tt.expectedCommand)
			validateArgs(t, container, tt.expectedArgs)
			validateEnvironmentVariables(t, container, tt.expectedEnv)
			validateLivenessProbe(t, container, tt.expectLivenessRemoved, tt.role)
			validateStartupProbe(t, container, tt.expectStartupRemoved, tt.role)
			validateReadinessProbe(t, container, tt.expectReadinessRemoved, tt.expectedReadinessProbe, tt.role)
		})
	}
}

// Helper functions to reduce cyclomatic complexity of the main test

func validateVolumeMounts(t *testing.T, container *corev1.Container, expected []corev1.VolumeMount) {
	if len(container.VolumeMounts) != len(expected) {
		t.Errorf("UpdateContainer() volume mounts count = %d, want %d", len(container.VolumeMounts), len(expected))
		return
	}

	for i, expectedVolumeMount := range expected {
		actualVolumeMount := container.VolumeMounts[i]
		if actualVolumeMount.Name != expectedVolumeMount.Name {
			t.Errorf("UpdateContainer() volume mount[%d].Name = %s, want %s", i, actualVolumeMount.Name, expectedVolumeMount.Name)
		}
		if actualVolumeMount.MountPath != expectedVolumeMount.MountPath {
			t.Errorf("UpdateContainer() volume mount[%d].MountPath = %s, want %s", i, actualVolumeMount.MountPath, expectedVolumeMount.MountPath)
		}
		if actualVolumeMount.ReadOnly != expectedVolumeMount.ReadOnly {
			t.Errorf("UpdateContainer() volume mount[%d].ReadOnly = %t, want %t", i, actualVolumeMount.ReadOnly, expectedVolumeMount.ReadOnly)
		}
	}
}

func validateCommand(t *testing.T, container *corev1.Container, expected []string) {
	if len(container.Command) != len(expected) {
		t.Errorf("UpdateContainer() command length = %d, want %d", len(container.Command), len(expected))
		return
	}

	for i, expectedCmd := range expected {
		if container.Command[i] != expectedCmd {
			t.Errorf("UpdateContainer() command[%d] = %s, want %s", i, container.Command[i], expectedCmd)
		}
	}
}

func validateArgs(t *testing.T, container *corev1.Container, expected []string) {
	if len(container.Args) != len(expected) {
		t.Errorf("UpdateContainer() args length = %d, want %d", len(container.Args), len(expected))
		return
	}

	for i, expectedArg := range expected {
		if container.Args[i] != expectedArg {
			t.Errorf("UpdateContainer() args[%d] = %s, want %s", i, container.Args[i], expectedArg)
		}
	}
}

func validateEnvironmentVariables(t *testing.T, container *corev1.Container, expected []corev1.EnvVar) {
	if len(container.Env) != len(expected) {
		t.Errorf("UpdateContainer() env count = %d, want %d", len(container.Env), len(expected))
		return
	}

	for i, expectedEnv := range expected {
		actualEnv := container.Env[i]
		if actualEnv.Name != expectedEnv.Name {
			t.Errorf("UpdateContainer() env[%d].Name = %s, want %s", i, actualEnv.Name, expectedEnv.Name)
		}
		if actualEnv.Value != expectedEnv.Value {
			t.Errorf("UpdateContainer() env[%d].Value = %s, want %s", i, actualEnv.Value, expectedEnv.Value)
		}
	}
}

func validateLivenessProbe(t *testing.T, container *corev1.Container, expectRemoved bool, role Role) {
	if expectRemoved {
		if container.LivenessProbe != nil {
			t.Errorf("UpdateContainer() should remove LivenessProbe for %s", role)
		}
	} else {
		if container.LivenessProbe == nil {
			t.Errorf("UpdateContainer() should not remove LivenessProbe for %s", role)
		}
	}
}

func validateStartupProbe(t *testing.T, container *corev1.Container, expectRemoved bool, role Role) {
	if expectRemoved {
		if container.StartupProbe != nil {
			t.Errorf("UpdateContainer() should remove StartupProbe for %s", role)
		}
	} else {
		if container.StartupProbe == nil {
			t.Errorf("UpdateContainer() should not remove StartupProbe for %s", role)
		}
	}
}

func validateReadinessProbe(t *testing.T, container *corev1.Container, expectRemoved bool, expected *corev1.Probe, role Role) {
	if expectRemoved {
		if container.ReadinessProbe != nil {
			t.Errorf("UpdateContainer() should remove ReadinessProbe for %s", role)
		}
	} else if expected != nil {
		// Check that readiness probe matches expected
		if container.ReadinessProbe == nil {
			t.Errorf("UpdateContainer() should set ReadinessProbe for %s", role)
		} else {
			validateProbeDetails(t, container.ReadinessProbe, expected)
		}
	} else {
		// No specific readiness probe expected, should remain as originally set
		if container.ReadinessProbe == nil {
			t.Errorf("UpdateContainer() should not remove ReadinessProbe for %s", role)
		}
	}
}

func validateProbeDetails(t *testing.T, actual, expected *corev1.Probe) {
	// Compare probe details
	if actual.TCPSocket == nil {
		t.Errorf("UpdateContainer() ReadinessProbe should have TCPSocket")
	} else if actual.TCPSocket.Port.IntVal != expected.TCPSocket.Port.IntVal {
		t.Errorf("UpdateContainer() ReadinessProbe port = %d, want %d", actual.TCPSocket.Port.IntVal, expected.TCPSocket.Port.IntVal)
	}
	if actual.InitialDelaySeconds != expected.InitialDelaySeconds {
		t.Errorf("UpdateContainer() ReadinessProbe InitialDelaySeconds = %d, want %d", actual.InitialDelaySeconds, expected.InitialDelaySeconds)
	}
	if actual.PeriodSeconds != expected.PeriodSeconds {
		t.Errorf("UpdateContainer() ReadinessProbe PeriodSeconds = %d, want %d", actual.PeriodSeconds, expected.PeriodSeconds)
	}
	if actual.TimeoutSeconds != expected.TimeoutSeconds {
		t.Errorf("UpdateContainer() ReadinessProbe TimeoutSeconds = %d, want %d", actual.TimeoutSeconds, expected.TimeoutSeconds)
	}
	if actual.FailureThreshold != expected.FailureThreshold {
		t.Errorf("UpdateContainer() ReadinessProbe FailureThreshold = %d, want %d", actual.FailureThreshold, expected.FailureThreshold)
	}
}

func TestTRTLLMBackend_UpdatePodSpec(t *testing.T) {
	tests := []struct {
		name                string
		numberOfNodes       int32
		role                Role
		multinodeDeployer   MultinodeDeployer
		initialVolumes      []corev1.Volume
		expectedVolumeCount int
		shouldHaveSSHVolume bool
	}{
		{
			name:                "Single node - no SSH volume added",
			numberOfNodes:       1,
			role:                RoleMain,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialVolumes:      []corev1.Volume{},
			expectedVolumeCount: 0,
			shouldHaveSSHVolume: false,
		},
		{
			name:                "Multinode leader - SSH volume added",
			numberOfNodes:       3,
			role:                RoleLeader,
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialVolumes:      []corev1.Volume{},
			expectedVolumeCount: 1,
			shouldHaveSSHVolume: true,
		},
		{
			name:                "Multinode worker - SSH volume added",
			numberOfNodes:       2,
			role:                RoleWorker,
			multinodeDeployer:   &LWSMultinodeDeployer{},
			initialVolumes:      []corev1.Volume{},
			expectedVolumeCount: 1,
			shouldHaveSSHVolume: true,
		},
		{
			name:              "Multinode with existing volumes",
			numberOfNodes:     2,
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialVolumes: []corev1.Volume{
				{Name: "existing-volume"},
			},
			expectedVolumeCount: 2,
			shouldHaveSSHVolume: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{
				MpiRunSecretName: mpiRunSecretName,
			}
			podSpec := &corev1.PodSpec{
				Volumes: tt.initialVolumes,
				Containers: []corev1.Container{
					{
						Name: commonconsts.MainContainerName,
						Env:  []corev1.EnvVar{},
					},
				},
			}
			component := &v1alpha1.DynamoComponentDeploymentOverridesSpec{}

			// Call UpdatePodSpec
			backend.UpdatePodSpec(podSpec, tt.numberOfNodes, tt.role, component, "test-service")

			// Check volume count
			if len(podSpec.Volumes) != tt.expectedVolumeCount {
				t.Errorf("UpdatePodSpec() volume count = %d, want %d", len(podSpec.Volumes), tt.expectedVolumeCount)
			}

			// Check for SSH volume
			hasSSHVolume := false
			for _, volume := range podSpec.Volumes {
				if volume.Name == mpiRunSecretName {
					hasSSHVolume = true
					// Verify volume configuration
					if volume.VolumeSource.Secret == nil {
						t.Errorf("UpdatePodSpec() SSH volume should use Secret volume source")
					} else {
						if volume.VolumeSource.Secret.SecretName != mpiRunSecretName {
							t.Errorf("UpdatePodSpec() SSH volume secret name = %s, want %s", volume.VolumeSource.Secret.SecretName, mpiRunSecretName)
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
		name              string
		numberOfNodes     int32
		multinodeDeployer MultinodeDeployer
		serviceName       string
		expectedContains  []string
		expectedNodeCount int32
	}{
		{
			name:              "Grove deployment with 3 nodes",
			numberOfNodes:     3,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test-service",
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
			name:              "LWS deployment with 2 nodes",
			numberOfNodes:     2,
			multinodeDeployer: &LWSMultinodeDeployer{},
			serviceName:       "test-service",
			expectedContains: []string{
				"$(LWS_LEADER_ADDRESS)",
				"$(LWS_WORKER_1_ADDRESS)",
			},
			expectedNodeCount: 2,
		},
		{
			name:              "Grove deployment with 5 nodes",
			numberOfNodes:     5,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "worker",
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
			name:              "LWS deployment with 4 nodes",
			numberOfNodes:     4,
			multinodeDeployer: &LWSMultinodeDeployer{},
			serviceName:       "worker",
			expectedContains: []string{
				"$(LWS_LEADER_ADDRESS)",
				"$(LWS_WORKER_1_ADDRESS)",
				"$(LWS_WORKER_2_ADDRESS)",
				"$(LWS_WORKER_3_ADDRESS)",
			},
			expectedNodeCount: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			result := backend.generateWorkerHostnames(tt.numberOfNodes, tt.serviceName, tt.multinodeDeployer)

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
	expectedSSHVolumeMount := corev1.VolumeMount{
		Name:      mpiRunSecretName,
		MountPath: "/ssh-pk",
		ReadOnly:  true,
	}

	tests := []struct {
		name                 string
		initialVolumeMounts  []corev1.VolumeMount
		expectedVolumeMounts []corev1.VolumeMount
	}{
		{
			name:                 "Add SSH volume mount to empty container",
			initialVolumeMounts:  []corev1.VolumeMount{},
			expectedVolumeMounts: []corev1.VolumeMount{expectedSSHVolumeMount},
		},
		{
			name: "Add SSH volume mount to container with existing mounts",
			initialVolumeMounts: []corev1.VolumeMount{
				{Name: "existing-mount", MountPath: "/existing"},
			},
			expectedVolumeMounts: []corev1.VolumeMount{
				{Name: "existing-mount", MountPath: "/existing"},
				expectedSSHVolumeMount,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{
				MpiRunSecretName: mpiRunSecretName,
			}
			container := &corev1.Container{
				VolumeMounts: tt.initialVolumeMounts,
			}

			backend.addSSHVolumeMount(container)

			// Check that volume mounts match expected
			if len(container.VolumeMounts) != len(tt.expectedVolumeMounts) {
				t.Errorf("addSSHVolumeMount() volume mount count = %d, want %d", len(container.VolumeMounts), len(tt.expectedVolumeMounts))
				return
			}

			for i, expected := range tt.expectedVolumeMounts {
				actual := container.VolumeMounts[i]
				if actual.Name != expected.Name {
					t.Errorf("addSSHVolumeMount() volume mount[%d].Name = %s, want %s", i, actual.Name, expected.Name)
				}
				if actual.MountPath != expected.MountPath {
					t.Errorf("addSSHVolumeMount() volume mount[%d].MountPath = %s, want %s", i, actual.MountPath, expected.MountPath)
				}
				if actual.ReadOnly != expected.ReadOnly {
					t.Errorf("addSSHVolumeMount() volume mount[%d].ReadOnly = %t, want %t", i, actual.ReadOnly, expected.ReadOnly)
				}
			}
		})
	}
}

func TestTRTLLMBackend_setupLeaderContainer(t *testing.T) {
	tests := []struct {
		name              string
		numberOfNodes     int32
		multinodeDeployer MultinodeDeployer
		serviceName       string
		component         *v1alpha1.DynamoComponentDeploymentOverridesSpec
		initialArgs       []string
		initialCommand    []string
		expected          string
	}{
		{
			name:              "Leader with args and GPU resources",
			numberOfNodes:     3,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test-service",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "2",
						},
					},
				},
			},
			initialArgs:    []string{"python3", "--model", "test"},
			initialCommand: []string{},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 6 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-wkr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-wkr-1.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python3 --model test'",
		},
		{
			name:              "Leader with command and no GPU resources",
			numberOfNodes:     2,
			multinodeDeployer: &LWSMultinodeDeployer{},
			serviceName:       "worker",
			component:         &v1alpha1.DynamoComponentDeploymentOverridesSpec{},
			initialArgs:       []string{},
			initialCommand:    []string{"python", "-m", "worker"},
			expected:          "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 0 -H $(LWS_LEADER_ADDRESS),$(LWS_WORKER_1_ADDRESS) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python -m worker'",
		},
		{
			name:              "Leader with both command and args (shell command - args take precedence)",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
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
			initialCommand: []string{"sh", "-c"},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch launch --config test.yaml'",
		},
		{
			name:              "Leader with python command and args (combined)",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"-m", "dynamo.trtllm", "--model-path", "test"},
			initialCommand: []string{"python3"},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path test'",
		},
		{
			name:              "Leader with python module command and separate args",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"--model-path", "Qwen/Qwen3-0.6B", "--served-model-name", "Qwen/Qwen3-0.6B", "--disaggregation-mode", "prefill"},
			initialCommand: []string{"python3", "-m", "dynamo.trtllm"},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --served-model-name Qwen/Qwen3-0.6B --disaggregation-mode prefill'",
		},
		{
			name:              "Leader with absolute path python command",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Limits: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"-m", "dynamo.trtllm", "--model-path", "test"},
			initialCommand: []string{"/usr/bin/python3.8"},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch /usr/bin/python3.8 -m dynamo.trtllm --model-path test'",
		},
		{
			name:              "Leader with all environment variables forwarded",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"serve", "--model", "test"},
			initialCommand: []string{},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch serve --model test'",
		},
		{
			name:              "Leader with overlapping environment variables (deduplication test)",
			numberOfNodes:     2,
			multinodeDeployer: &GroveMultinodeDeployer{},
			serviceName:       "test",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					Resources: &common.Resources{
						Requests: &common.ResourceItem{
							GPU: "1",
						},
					},
				},
			},
			initialArgs:    []string{"serve", "--model", "test"},
			initialCommand: []string{},
			expected:       "mkdir -p ~/.ssh && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && mpirun --oversubscribe -n 2 -H $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-ldr-0.$(GROVE_HEADLESS_SERVICE),$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-wkr-0.$(GROVE_HEADLESS_SERVICE) --mca pml ob1 --mca plm_rsh_args \"-p 2222 -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" -x CUDA_VISIBLE_DEVICES -x CUSTOM_VAR -x HF_DATASETS_CACHE -x HF_ENDPOINT -x HF_HOME -x HF_TOKEN -x HOME -x HUGGING_FACE_HUB_TOKEN -x LD_LIBRARY_PATH -x MODEL_PATH -x NCCL_DEBUG -x NCCL_IB_DISABLE -x NCCL_P2P_DISABLE -x PATH -x PYTHONPATH -x TENSORRT_LLM_CACHE_DIR -x TOKENIZERS_PARALLELISM -x TRANSFORMERS_CACHE -x USER bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch serve --model test'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backend := &TRTLLMBackend{}
			container := &corev1.Container{
				Args:    tt.initialArgs,
				Command: tt.initialCommand,
			}

			// Add test environment variables for the deduplication test
			if tt.name == "Leader with overlapping environment variables (deduplication test)" {
				container.Env = []corev1.EnvVar{
					{Name: "CUDA_VISIBLE_DEVICES", Value: "0,1"}, // This should NOT be duplicated
					{Name: "CUSTOM_VAR", Value: "test_value"},    // This should be added
					{Name: "PATH", Value: "/custom/path"},        // This should NOT be duplicated
				}
			}

			backend.setupLeaderContainer(container, tt.numberOfNodes, tt.serviceName, tt.component, tt.multinodeDeployer)

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
				if argsStr != tt.expected {
					t.Errorf("setupLeaderContainer() args = %q, want %q", argsStr, tt.expected)
				}
			}
		})
	}
}

func TestTRTLLMBackend_setupWorkerContainer(t *testing.T) {
	tests := []struct {
		name           string
		initialArgs    []string
		initialCommand []string
		expected       string
	}{
		{
			name:           "Worker setup with initial args",
			initialArgs:    []string{"some", "args"},
			initialCommand: []string{},
			expected:       "mkdir -p ~/.ssh ~/.ssh/host_keys ~/.ssh/run && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && ssh-keygen -t rsa -f ~/.ssh/host_keys/ssh_host_rsa_key -N '' && ssh-keygen -t ecdsa -f ~/.ssh/host_keys/ssh_host_ecdsa_key -N '' && ssh-keygen -t ed25519 -f ~/.ssh/host_keys/ssh_host_ed25519_key -N '' && printf 'Port 2222\\nHostKey ~/.ssh/host_keys/ssh_host_rsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ecdsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ed25519_key\\nPidFile ~/.ssh/run/sshd.pid\\nPermitRootLogin yes\\nPasswordAuthentication no\\nPubkeyAuthentication yes\\nAuthorizedKeysFile ~/.ssh/authorized_keys\\n' > ~/.ssh/sshd_config && mkdir -p /run/sshd && /usr/sbin/sshd -D -f ~/.ssh/sshd_config",
		},
		{
			name:           "Worker setup with initial command",
			initialArgs:    []string{},
			initialCommand: []string{"original", "command"},
			expected:       "mkdir -p ~/.ssh ~/.ssh/host_keys ~/.ssh/run && ls -la /ssh-pk/ && cp /ssh-pk/private.key ~/.ssh/id_rsa && cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub && cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys && chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys && chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys && printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort 2222\\n' > ~/.ssh/config && ssh-keygen -t rsa -f ~/.ssh/host_keys/ssh_host_rsa_key -N '' && ssh-keygen -t ecdsa -f ~/.ssh/host_keys/ssh_host_ecdsa_key -N '' && ssh-keygen -t ed25519 -f ~/.ssh/host_keys/ssh_host_ed25519_key -N '' && printf 'Port 2222\\nHostKey ~/.ssh/host_keys/ssh_host_rsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ecdsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ed25519_key\\nPidFile ~/.ssh/run/sshd.pid\\nPermitRootLogin yes\\nPasswordAuthentication no\\nPubkeyAuthentication yes\\nAuthorizedKeysFile ~/.ssh/authorized_keys\\n' > ~/.ssh/sshd_config && mkdir -p /run/sshd && /usr/sbin/sshd -D -f ~/.ssh/sshd_config",
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
				if argsStr != tt.expected {
					t.Errorf("setupWorkerContainer() args = %q, want %q", argsStr, tt.expected)
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

func TestTRTLLMBackend_UpdateContainer_UseAsCompilationCache(t *testing.T) {
	backend := &TRTLLMBackend{}

	tests := []struct {
		name                       string
		component                  *v1alpha1.DynamoComponentDeploymentOverridesSpec
		volumeMounts               []corev1.VolumeMount
		expectNoEnvVarChanges      bool
		expectLoggedPartialSupport bool
	}{
		{
			name: "TensorRT-LLM backend with useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: []v1alpha1.VolumeMount{
						{
							Name:                  "trtllm-cache",
							MountPoint:            "/cache/trtllm",
							UseAsCompilationCache: true,
						},
					},
				},
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true, // TensorRT-LLM doesn't set env vars yet
			expectLoggedPartialSupport: true,
		},
		{
			name: "TensorRT-LLM backend with useAsCompilationCache at custom mount point",
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
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true, // TensorRT-LLM doesn't set env vars yet
			expectLoggedPartialSupport: true,
		},
		{
			name: "TensorRT-LLM backend without useAsCompilationCache",
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
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true,
			expectLoggedPartialSupport: false,
		},
		{
			name: "TensorRT-LLM backend with no volume mounts",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					VolumeMounts: nil,
				},
			},
			volumeMounts:               []corev1.VolumeMount{},
			expectNoEnvVarChanges:      true,
			expectLoggedPartialSupport: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a container with initial state including volume mounts
			container := &corev1.Container{
				Env:          []corev1.EnvVar{},
				VolumeMounts: tt.volumeMounts,
			}

			// Store original env vars for comparison
			originalEnvCount := len(container.Env)

			// Call UpdateContainer (single node to avoid multinode logic)
			backend.UpdateContainer(container, 1, RoleMain, tt.component, "test-service", &GroveMultinodeDeployer{})

			if tt.expectNoEnvVarChanges {
				// Check that no new environment variables were added
				if len(container.Env) != originalEnvCount {
					t.Errorf("Expected no environment variable changes, but env count changed from %d to %d", originalEnvCount, len(container.Env))
				}
			}
		})
	}
}
