package dynamo

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

type SGLangBackend struct{}

func (b *SGLangBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
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

	// Generate the flags to add
	flags := b.getMultinodeFlags(numberOfNodes, role, multinodeDeploymentType, serviceName)
	if flags == "" {
		return
	}

	// Flatten all args into a single command and inject flags
	if len(container.Args) > 0 {
		fullCommand := strings.Join(container.Args, " ")
		modifiedCommand := b.injectFlagsIntoPythonCommand(fullCommand, flags)
		container.Args = []string{modifiedCommand}
	}
}

func (b *SGLangBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) {
	// do nothing
}

// getMultinodeFlags returns the multinode flags as a single string
func (b *SGLangBackend) getMultinodeFlags(numberOfNodes int32, role Role, multinodeDeploymentType commonconsts.MultinodeDeploymentType, serviceName string) string {
	var distInitAddr, nodeRank string

	// Determine dist-init-addr
	if multinodeDeploymentType == commonconsts.MultinodeDeploymentTypeGrove {
		leaderHostname := generateGroveLeaderHostname(serviceName)
		distInitAddr = fmt.Sprintf("%s:29500", leaderHostname)
	} else {
		distInitAddr = "${LWS_LEADER_ADDRESS}:29500"
	}

	// Determine node-rank
	if role == RoleLeader {
		nodeRank = "0"
	} else {
		if multinodeDeploymentType == commonconsts.MultinodeDeploymentTypeGrove {
			nodeRank = "$((GROVE_PCLQ_POD_INDEX + 1))"
		} else {
			nodeRank = "${LWS_WORKER_INDEX}"
		}
	}

	return fmt.Sprintf("--dist-init-addr %s --nnodes %d --node-rank %s", distInitAddr, numberOfNodes, nodeRank)
}

// injectFlagsIntoPythonCommand finds python sglang commands and adds flags after them
func (b *SGLangBackend) injectFlagsIntoPythonCommand(arg, flags string) string {
	// Regex to match python commands that contain sglang
	// Matches: python, python3, python3.11, etc. followed by sglang-related modules
	pattern := `(python[0-9.]*\s+[^|&;]*sglang[^|&;]*?)(\s|$|[|&;])`

	re := regexp.MustCompile(pattern)

	// Replace with the command + flags + whatever comes after
	result := re.ReplaceAllStringFunc(arg, func(match string) string {
		// Extract the python command part and the delimiter
		submatches := re.FindStringSubmatch(match)
		if len(submatches) >= 3 {
			pythonCmd := submatches[1]
			delimiter := submatches[2]
			return pythonCmd + " " + flags + delimiter
		}
		return match
	})

	return result
}
