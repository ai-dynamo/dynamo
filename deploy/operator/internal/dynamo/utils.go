package dynamo

import (
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

/*
 * Flag Injection Strategy for Multinode
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
func injectFlagsIntoContainerCommand(container *corev1.Container, flags string, needsShell bool, framework string) {
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
			modifiedArg := injectFlagsIntoPythonCommand(arg, flags, framework)
			if modifiedArg != arg { // flags were successfully injected
				container.Args[i] = modifiedArg
				break // stop after first successful injection
			}
		}
	}
}

func injectFlagsIntoPythonCommand(arg, flags string, framework string) string {
	// Regex to match python commands that contain sglang
	// Matches: python, python3, python3.11, etc. followed by sglang-related modules
	pattern := fmt.Sprintf(`(python[0-9.]*\s+[^|&;]*%s[^|&;]*?)(\s|$|[|&;])`, framework)

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
