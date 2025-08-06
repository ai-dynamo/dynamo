package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

type TRTLLMBackend struct{}

func (b *TRTLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	// For single node, nothing to do
	if numberOfNodes <= 1 {
		return
	}

	// Remove probes for multinode leader and worker
	if role == RoleLeader || role == RoleWorker {
		container.LivenessProbe = nil
		container.ReadinessProbe = nil
		container.StartupProbe = nil
	}

	// Add SSH keypair volume mount for multinode deployments
	b.addSSHVolumeMount(container)

	// Update container command based on role
	switch role {
	case RoleLeader:
		b.setupLeaderContainer(container, numberOfNodes, multinodeDeploymentType, serviceName, component)
	case RoleWorker:
		b.setupWorkerContainer(container)
	}
}

func (b *TRTLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	// Add SSH keypair volume for TRTLLM multinode deployments
	if numberOfNodes > 1 {
		sshVolume := corev1.Volume{
			Name: "ssh-keypair",
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName:  "ssh-keypair-secret",
					DefaultMode: func() *int32 { mode := int32(0644); return &mode }(),
				},
			},
		}
		podSpec.Volumes = append(podSpec.Volumes, sshVolume)
	}
}

// addSSHVolumeMount adds the SSH keypair secret volume mount to the container
func (b *TRTLLMBackend) addSSHVolumeMount(container *corev1.Container) {
	sshVolumeMount := corev1.VolumeMount{
		Name:      "ssh-keypair",
		MountPath: "/ssh-pk",
		ReadOnly:  true,
	}
	container.VolumeMounts = append(container.VolumeMounts, sshVolumeMount)
}

// setupLeaderContainer configures the leader node with SSH setup and mpirun command
func (b *TRTLLMBackend) setupLeaderContainer(container *corev1.Container, numberOfNodes int32, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string, component *v1alpha1.DynamoComponentDeploymentOverridesSpec) {
	// Generate the list of worker hostnames
	workerHosts := b.generateWorkerHostnames(numberOfNodes, multinodeDeploymentType, serviceName)

	// Store original command/args for later use
	var originalCommand string
	if len(container.Args) > 0 {
		originalCommand = strings.Join(container.Args, " ")
	} else if len(container.Command) > 0 {
		originalCommand = strings.Join(container.Command, " ")
	}

	// Setup SSH and run mpirun command
	sshSetupCommands := []string{
		"mkdir -p ~/.ssh",
		"ls -la /ssh-pk/", // Debug: list files in ssh-pk directory
		"cp /ssh-pk/private.key ~/.ssh/id_rsa",
		"cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub",
		"cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys",
		"chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys",
		"chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys",
		"printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\n' > ~/.ssh/config",
	}

	// Calculate total number of GPUs across all nodes
	gpusPerNode := getGPUsPerNode(component.Resources)
	totalGPUs := numberOfNodes * gpusPerNode

	// Build mpirun command
	mpirunCmd := fmt.Sprintf("mpirun --oversubscribe -n %d -H %s %s",
		totalGPUs,
		workerHosts,
		originalCommand)

	// Combine SSH setup and mpirun command
	fullCommand := strings.Join(append(sshSetupCommands, mpirunCmd), " && ")

	// Update container to use bash with the full command
	container.Command = []string{"/bin/sh", "-c"}
	container.Args = []string{fullCommand}
}

// setupWorkerContainer configures worker nodes with SSH setup and daemon
func (b *TRTLLMBackend) setupWorkerContainer(container *corev1.Container) {
	// Setup SSH for worker nodes
	sshSetupCommands := []string{
		"mkdir -p ~/.ssh ~/.ssh/host_keys",
		"ls -la /ssh-pk/", // Debug: list files in ssh-pk directory
		"cp /ssh-pk/private.key ~/.ssh/id_rsa",
		"cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub",
		"cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys",
		"chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys",
		"chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys",
		"printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\n' > ~/.ssh/config",
		// Generate host keys in user writable directory
		"ssh-keygen -t rsa -f ~/.ssh/host_keys/ssh_host_rsa_key -N ''",
		"ssh-keygen -t ecdsa -f ~/.ssh/host_keys/ssh_host_ecdsa_key -N ''",
		"ssh-keygen -t ed25519 -f ~/.ssh/host_keys/ssh_host_ed25519_key -N ''",
		// Create SSH daemon config to use custom host keys location
		"printf 'Port 22\\nHostKey ~/.ssh/host_keys/ssh_host_rsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ecdsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ed25519_key\\nPermitRootLogin yes\\nPasswordAuthentication no\\nPubkeyAuthentication yes\\nAuthorizedKeysFile ~/.ssh/authorized_keys\\n' > ~/.ssh/sshd_config",
		"/usr/sbin/sshd -D -f ~/.ssh/sshd_config",
	}

	fullCommand := strings.Join(sshSetupCommands, " && ")

	// Update container to use bash with the SSH setup and daemon
	container.Command = []string{"/bin/sh", "-c"}
	container.Args = []string{fullCommand}
}

// generateWorkerHostnames creates a comma-separated list of worker hostnames
func (b *TRTLLMBackend) generateWorkerHostnames(numberOfNodes int32, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) string {
	var hostnames []string

	// Add leader hostname first
	if multinodeDeploymentType == commonconsts.MultinodeDeploymentTypeGrove {
		leaderHostname := generateGroveLeaderHostname(serviceName)
		hostnames = append(hostnames, leaderHostname)

		// Add worker hostnames
		for i := int32(0); i < numberOfNodes-1; i++ {
			workerHostname := fmt.Sprintf("${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}-%s-%s-%d.${GROVE_HEADLESS_SERVICE}",
				serviceName, commonconsts.GroveRoleSuffixWorker, i)
			hostnames = append(hostnames, workerHostname)
		}
	} else {
		// For LWS deployment type - using environment variables
		hostnames = append(hostnames, "${LWS_LEADER_ADDRESS}")
		for i := int32(1); i < numberOfNodes; i++ {
			hostnames = append(hostnames, fmt.Sprintf("${LWS_WORKER_%d_ADDRESS}", i))
		}
	}

	return strings.Join(hostnames, ",")
}

// getGPUsPerNode extracts the number of GPUs per node from resources
func getGPUsPerNode(resources *common.Resources) int32 {
	if resources != nil && resources.Requests != nil && resources.Requests.GPU != "" {
		if gpus, err := strconv.ParseInt(resources.Requests.GPU, 10, 32); err == nil {
			return int32(gpus)
		}
	}
	if resources != nil && resources.Limits != nil && resources.Limits.GPU != "" {
		if gpus, err := strconv.ParseInt(resources.Limits.GPU, 10, 32); err == nil {
			return int32(gpus)
		}
	}
	return 0 // Default to 0 GPUs if not specified
}
