package dynamo

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type TRTLLMBackend struct {
	MpiRunSecretName string
}

func (b *TRTLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// Check for volumeMounts with useAsCompilationCache=true
	for _, volumeMount := range component.VolumeMounts {
		if volumeMount.UseAsCompilationCache {
			logger := log.Log.WithName("trtllm-backend")
			logger.Info("Compilation cache configured for TensorRT-LLM but not yet fully supported",
				"backend", "trtllm",
				"status", "partial-support",
				"use-as-compilation-cache", true,
				"env-vars-set", false,
				"next-steps", "upstream TensorRT-LLM changes needed")
		}
	}

	// For single node, nothing to do
	if numberOfNodes <= 1 {
		return
	}

	// Configure probes for multinode deployments
	if role == RoleWorker {
		// For workers: remove liveness and startup probes, set readiness to check SSH port
		container.LivenessProbe = nil
		container.StartupProbe = nil
		container.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(commonconsts.MpiRunSshPort),
				},
			},
			InitialDelaySeconds: 20,
			PeriodSeconds:       20,
			TimeoutSeconds:      5,
			FailureThreshold:    10,
		}
	}
	// For leaders: leave all probes untouched

	// Add SSH keypair volume mount for multinode deployments
	b.addSSHVolumeMount(container)

	// Add OpenMPI environment variable to keep FQDN hostnames
	envVar := corev1.EnvVar{
		Name:  "OMPI_MCA_orte_keep_fqdn_hostnames",
		Value: "1",
	}
	container.Env = append(container.Env, envVar)

	// Update container command based on role
	switch role {
	case RoleLeader:
		b.setupLeaderContainer(container, numberOfNodes, serviceName, component, multinodeDeployer)
	case RoleWorker:
		b.setupWorkerContainer(container)
	}
}

func (b *TRTLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentSharedSpec, serviceName string) {
	// Add SSH keypair volume for TRTLLM multinode deployments
	if numberOfNodes > 1 {
		sshVolume := corev1.Volume{
			Name: b.MpiRunSecretName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{
					SecretName:  b.MpiRunSecretName,
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
		Name:      b.MpiRunSecretName,
		MountPath: "/ssh-pk",
		ReadOnly:  true,
	}
	container.VolumeMounts = append(container.VolumeMounts, sshVolumeMount)
}

// setupLeaderContainer configures the leader node with SSH setup and mpirun command
func (b *TRTLLMBackend) setupLeaderContainer(container *corev1.Container, numberOfNodes int32, serviceName string, component *v1alpha1.DynamoComponentDeploymentSharedSpec, multinodeDeployer MultinodeDeployer) {
	// Generate the list of worker hostnames
	workerHosts := b.generateWorkerHostnames(numberOfNodes, serviceName, multinodeDeployer)

	// Store original command/args for later use
	var originalCommand string

	if len(container.Command) > 0 && isPythonCommand(container.Command[0]) {
		// Direct Python command: combine command + args
		var parts []string
		parts = append(parts, container.Command...)
		if len(container.Args) > 0 {
			parts = append(parts, container.Args...)
		}
		originalCommand = strings.Join(parts, " ")
	} else if len(container.Args) > 0 {
		// Shell command (sh -c): args contains the full command
		originalCommand = strings.Join(container.Args, " ")
	} else if len(container.Command) > 0 {
		// Fallback: just command
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
		fmt.Sprintf("printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort %d\\n' > ~/.ssh/config", commonconsts.MpiRunSshPort),
	}

	// Calculate total number of GPUs across all nodes
	gpusPerNode := getGPUsPerNode(component.Resources)
	totalGPUs := numberOfNodes * gpusPerNode

	// Build mpirun command with explicit SSH configuration and environment variables
	// Wrap the entire command (trtllm-llmapi-launch + original command) in bash -c for proper shell interpretation
	wrappedCommand := fmt.Sprintf("bash -c 'source /opt/dynamo/venv/bin/activate && trtllm-llmapi-launch %s'", originalCommand)

	// Generate environment variable flags for mpirun
	envVarsStr := generateEnvVarFlags(container.Env)

	mpirunCmd := fmt.Sprintf("mpirun --oversubscribe -n %d -H %s --mca pml ob1 --mca plm_rsh_args \"-p %d -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa\" %s %s",
		totalGPUs,
		workerHosts,
		commonconsts.MpiRunSshPort,
		envVarsStr,
		wrappedCommand)

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
		"mkdir -p ~/.ssh ~/.ssh/host_keys ~/.ssh/run",
		"ls -la /ssh-pk/", // Debug: list files in ssh-pk directory
		"cp /ssh-pk/private.key ~/.ssh/id_rsa",
		"cp /ssh-pk/private.key.pub ~/.ssh/id_rsa.pub",
		"cp /ssh-pk/private.key.pub ~/.ssh/authorized_keys",
		"chmod 600 ~/.ssh/id_rsa ~/.ssh/authorized_keys",
		"chmod 644 ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys",
		fmt.Sprintf("printf 'Host *\\nIdentityFile ~/.ssh/id_rsa\\nStrictHostKeyChecking no\\nPort %d\\n' > ~/.ssh/config", commonconsts.MpiRunSshPort),
		// Generate host keys in user writable directory
		"ssh-keygen -t rsa -f ~/.ssh/host_keys/ssh_host_rsa_key -N ''",
		"ssh-keygen -t ecdsa -f ~/.ssh/host_keys/ssh_host_ecdsa_key -N ''",
		"ssh-keygen -t ed25519 -f ~/.ssh/host_keys/ssh_host_ed25519_key -N ''",
		// Create SSH daemon config to use custom host keys location and non-privileged port
		fmt.Sprintf("printf 'Port %d\\nHostKey ~/.ssh/host_keys/ssh_host_rsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ecdsa_key\\nHostKey ~/.ssh/host_keys/ssh_host_ed25519_key\\nPidFile ~/.ssh/run/sshd.pid\\nPermitRootLogin yes\\nPasswordAuthentication no\\nPubkeyAuthentication yes\\nAuthorizedKeysFile ~/.ssh/authorized_keys\\n' > ~/.ssh/sshd_config", commonconsts.MpiRunSshPort),
		"mkdir -p /run/sshd",
		"/usr/sbin/sshd -D -f ~/.ssh/sshd_config",
	}

	fullCommand := strings.Join(sshSetupCommands, " && ")

	// Update container to use bash with the SSH setup and daemon
	container.Command = []string{"/bin/sh", "-c"}
	container.Args = []string{fullCommand}
}

// generateWorkerHostnames creates a comma-separated list of worker hostnames
func (b *TRTLLMBackend) generateWorkerHostnames(numberOfNodes int32, serviceName string, multinodeDeployer MultinodeDeployer) string {
	return strings.Join(multinodeDeployer.GetHostNames(serviceName, numberOfNodes), ",")
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

// getCommonTRTLLMEnvVars returns a map of common environment variables for TRTLLM deployments
func getCommonTRTLLMEnvVars() map[string]bool {
	return map[string]bool{
		"CUDA_VISIBLE_DEVICES": true, "MODEL_PATH": true, "HF_TOKEN": true, "HUGGING_FACE_HUB_TOKEN": true, "HF_ENDPOINT": true,
		"TOKENIZERS_PARALLELISM": true, "NCCL_DEBUG": true, "NCCL_IB_DISABLE": true, "NCCL_P2P_DISABLE": true,
		"TENSORRT_LLM_CACHE_DIR": true, "HF_HOME": true, "TRANSFORMERS_CACHE": true, "HF_DATASETS_CACHE": true,
		"PATH": true, "LD_LIBRARY_PATH": true, "PYTHONPATH": true, "HOME": true, "USER": true,
	}
}

// collectAllEnvVars combines explicit container env vars with common TRTLLM env vars, removing duplicates
func collectAllEnvVars(containerEnvVars []corev1.EnvVar) []string {
	// Initialize set with common environment variables
	envVarSet := getCommonTRTLLMEnvVars()

	// Add explicit environment variables from container
	for _, env := range containerEnvVars {
		envVarSet[env.Name] = true
	}

	// Convert set to sorted slice for consistent output
	envVarNames := make([]string, 0, len(envVarSet))
	for envVar := range envVarSet {
		envVarNames = append(envVarNames, envVar)
	}
	sort.Strings(envVarNames)

	return envVarNames
}

// formatEnvVarFlags converts environment variable names to mpirun -x flags
func formatEnvVarFlags(envVarNames []string) string {
	envVars := make([]string, 0, len(envVarNames))
	for _, envVar := range envVarNames {
		envVars = append(envVars, fmt.Sprintf("-x %s", envVar))
	}
	return strings.Join(envVars, " ")
}

// generateEnvVarFlags generates the complete environment variable flags string for mpirun
func generateEnvVarFlags(containerEnvVars []corev1.EnvVar) string {
	envVarNames := collectAllEnvVars(containerEnvVars)
	return formatEnvVarFlags(envVarNames)
}
