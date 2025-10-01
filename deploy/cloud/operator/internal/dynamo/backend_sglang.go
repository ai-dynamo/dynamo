package dynamo

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	SglangPort = "29500"
)

type SGLangBackend struct{}

// isPythonCommand checks if the command is a Python interpreter
func isPythonCommand(cmd string) bool {
	if cmd == "python" || cmd == "python3" {
		return true
	}
	// Match python with version numbers like python3.11, python2.7, etc.
	matched, _ := regexp.MatchString(`^python\d+(\.\d+)*$`, cmd)
	return matched
}

func (b *SGLangBackend) UpdateContainer(container *corev1.Container, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string, multinodeDeployer MultinodeDeployer) {
	// Check for volumeMounts with useAsCompilationCache=true
	for _, volumeMount := range component.VolumeMounts {
		if volumeMount.UseAsCompilationCache {
			logger := log.Log.WithName("sglang-backend")
			logger.Info("Compilation cache configured for SGLang but not yet fully supported",
				"backend", "sglang",
				"status", "partial-support",
				"cache-dir", volumeMount.MountPoint,
				"use-as-compilation-cache", true,
				"env-vars-set", false,
				"next-steps", "upstream SGLang changes needed")
		}
	}

	// For single node, nothing to do
	if numberOfNodes <= 1 {
		return
	}

	// Remove probes for multinode worker
	if role == RoleWorker {
		container.LivenessProbe = nil
		container.ReadinessProbe = nil
		container.StartupProbe = nil
	}

	// Generate the flags to add
	flags, needsShell := b.getMultinodeFlags(numberOfNodes, role, serviceName, multinodeDeployer)
	if flags == "" {
		return
	}

	/*
	 * Flag Injection Strategy for Multinode SGLang Deployments
	 *
	 * This code handles the injection of distributed training flags (--dist-init-addr, --nnodes, --node-rank)
	 * into container commands for multinode SGLang deployments. The complexity arises from supporting multiple
	 * container command patterns and ensuring proper environment variable interpretation.
	 *
	 * Two main scenarios are handled:
	 *
	 * 1. Direct Python Command (e.g., Command: ["python3"], Args: ["-m", "sglang", "..."])
	 *    - If shell interpretation is needed (for env vars): Wrap in "sh -c" with exec
	 *    - If no shell needed: Simply append flags to the Args array
	 *
	 * 2. Non-Python Command (e.g., Command: ["sh"], Args: ["-c", "python3 -m sglang ..."])
	 *    - Use regex-based injection to find embedded Python+SGLang commands within args
	 *    - Insert flags after the Python command but before any shell operators (|, &, ;)
	 *
	 * The needsShell flag indicates when environment variables require shell interpretation
	 */
	if len(container.Command) > 0 && isPythonCommand(container.Command[0]) {
		// Direct python command case
		if needsShell {
			// Transform to shell wrapper for env var interpretation
			fullCommand := strings.Join(container.Command, " ")
			originalArgs := strings.Join(container.Args, " ")
			var shellCommand string
			if len(container.Args) > 0 {
				// Use exec to ensure PID 1 is given to the python command
				shellCommand = fmt.Sprintf("exec %s %s %s", fullCommand, originalArgs, flags)
			} else {
				// Use exec to ensure PID 1 is given to the python command
				shellCommand = fmt.Sprintf("exec %s %s", fullCommand, flags)
			}
			container.Command = []string{"sh", "-c"}
			container.Args = []string{shellCommand}
		} else {
			// Simple append to args
			flagsSlice := strings.Fields(flags)
			container.Args = append(container.Args, flagsSlice...)
		}
	} else {
		// Non-python command case - try injection on each arg individually
		for i, arg := range container.Args {
			modifiedArg := b.injectFlagsIntoPythonCommand(arg, flags)
			if modifiedArg != arg { // flags were successfully injected
				container.Args[i] = modifiedArg
				break // stop after first successful injection
			}
		}
	}
}

func (b *SGLangBackend) UpdatePodSpec(podSpec *corev1.PodSpec, numberOfNodes int32, role Role, component *v1alpha1.DynamoComponentDeploymentOverridesSpec, serviceName string) {
	// do nothing
}

// getMultinodeFlags returns the multinode flags and whether shell interpretation is needed
func (b *SGLangBackend) getMultinodeFlags(numberOfNodes int32, role Role, serviceName string, multinodeDeployer MultinodeDeployer) (string, bool) {
	distInitAddr := fmt.Sprintf("%s:%s", multinodeDeployer.GetLeaderHostname(serviceName), SglangPort)

	var nodeRank string
	var needsShell bool

	if role == RoleLeader {
		nodeRank = "0"
		needsShell = false
	} else {
		nodeRank, needsShell = multinodeDeployer.GetNodeRank()
	}

	flags := fmt.Sprintf("--dist-init-addr %s --nnodes %d --node-rank %s", distInitAddr, numberOfNodes, nodeRank)
	return flags, needsShell
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
