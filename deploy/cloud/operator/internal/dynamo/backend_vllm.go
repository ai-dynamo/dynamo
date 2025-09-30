package dynamo

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	VLLMPort = "6379"
)

type VLLMBackend struct{}

func (b *VLLMBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	isMultinode := numberOfNodes > 1

	if isMultinode {
		// Apply multinode-specific argument modifications
		updateVLLMMultinodeArgs(container, role, serviceName, multinodeDeployer)

		// Remove probes for multinode worker and leader
		if role == RoleWorker {
			container.LivenessProbe = nil
			container.ReadinessProbe = nil
			container.StartupProbe = nil
		}
	}

	// Set compilation cache environment variables for VLLM
	cacheDir := ""

	// Check for volumeMounts with useAsCompilationCache=true
	for _, volumeMount := range component.VolumeMounts {
		if volumeMount.UseAsCompilationCache {
			cacheDir = volumeMount.MountPoint
			break
		}
	}

	if cacheDir != "" {
		// Set VLLM cache directory using the environment variable
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "VLLM_CACHE_ROOT",
			Value: cacheDir,
		})

		// Log confirmation that compilation cache is configured for VLLM
		logger := log.Log.WithName("vllm-backend")
		logger.Info("Compilation cache configured and enabled for VLLM backend",
			"backend", "vllm",
			"status", "fully-supported",
			"cache-dir", cacheDir,
			"use-as-compilation-cache", true,
			"env-vars-set", true,
			"env-vars", "VLLM_CACHE_ROOT")
	}
}

func (b *VLLMBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string) {
	// do nothing
}

// updateVLLMMultinodeArgs applies Ray-specific modifications for multinode deployments
func updateVLLMMultinodeArgs(container *corev1.Container, role Role, serviceName string, multinodeDeployer MultinodeDeployer) {
	switch role {
	case RoleLeader:
		if len(container.Args) > 0 {
			// Prepend ray start --head command to existing args
			container.Args = []string{fmt.Sprintf("ray start --head --port=%s && %s", VLLMPort, strings.Join(container.Args, " "))}
		}
	case RoleWorker:
		// Worker nodes only run Ray, completely replace args
		leaderHostname := multinodeDeployer.GetLeaderHostname(serviceName)
		container.Args = []string{fmt.Sprintf("ray start --address=%s:%s --block", leaderHostname, VLLMPort)}
	}
}
