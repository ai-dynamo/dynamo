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
 * All MultinodeDeployer implementations MUST return Kubernetes env-var
 * expansion syntax ("$(VAR)") from GetLeaderHostname / GetNodeRank. The
 * kubelet substitutes those references in container Args/Command before the
 * container starts, so plain $(VAR) references never require a shell wrapper.
 * Shell wrapping (`sh -c`) is only needed for shell-only constructs that the
 * kubelet does not evaluate - e.g. arithmetic expansion `$(( ... ))` or
 * command substitution - which is signaled by the `needsShell` bool returned
 * from GetNodeRank (Grove's `$((GROVE_PCLQ_POD_INDEX + 1))` is the canonical
 * example).
 *
 * Two main scenarios are handled:
 *
 * 1. Direct Python Command (e.g., Command: ["python3"], Args: ["-m", "sglang", "..."])
 *    - If needsShell is true (shell-only expression such as arithmetic): wrap
 *      the command in "sh -c" with exec so the shell evaluates the expression.
 *    - Otherwise: simply append flags to the Args array; the kubelet expands
 *      any $(VAR) references itself.
 *
 * 2. Non-Python Command (e.g., Command: ["sh"], Args: ["-c", "python3 -m sglang ..."])
 *    - Use regex-based injection to find embedded Python+SGLang commands within args
 *    - Insert flags after the Python command but before any shell operators (|, &, ;)
 */

// shellQuoteForBashC quotes a string so it survives shell interpretation inside sh -c.
// Simple args (flags, paths) pass through unchanged; args containing special characters
// (JSON, env vars, spaces, quotes) are wrapped in double quotes with inner escaping.
func shellQuoteForBashC(s string) string {
	if strings.ContainsAny(s, " \t\n'\"\\{}[]$`!") {
		escaped := s
		escaped = strings.ReplaceAll(escaped, `\`, `\\`) // must be first
		escaped = strings.ReplaceAll(escaped, `"`, `\"`)
		escaped = strings.ReplaceAll(escaped, `$`, `\$`)
		escaped = strings.ReplaceAll(escaped, "`", "\\`")
		escaped = strings.ReplaceAll(escaped, "'", `'"'"'`)
		return `"` + escaped + `"`
	}
	return s
}

// containerHasArg reports whether the container already carries the given
// flag/value pair in its Args (either as adjacent tokens "flag", "value" or
// as a single token "flag=value" or "flag value" embedded inside a shell
// string). It is used to make flag injection idempotent.
func containerHasArg(container *corev1.Container, flag, value string) bool {
	if container == nil {
		return false
	}
	joined := flag + " " + value
	equals := flag + "=" + value
	for i, arg := range container.Args {
		if strings.Contains(arg, joined) || strings.Contains(arg, equals) {
			return true
		}
		if arg == flag && i+1 < len(container.Args) && container.Args[i+1] == value {
			return true
		}
	}
	return false
}

func injectFlagsIntoContainerCommand(container *corev1.Container, flags string, needsShell bool, framework string) {
	if len(container.Command) > 0 && isPythonCommand(container.Command[0]) {
		// Direct python command case
		if needsShell {
			// Transform to shell wrapper for env var interpretation.
			// Quote each token individually so paths with spaces or special
			// characters survive shell interpretation.
			quotedCmd := make([]string, len(container.Command))
			for i, tok := range container.Command {
				quotedCmd[i] = shellQuoteForBashC(tok)
			}
			fullCommand := strings.Join(quotedCmd, " ")
			quotedArgs := make([]string, len(container.Args))
			for i, arg := range container.Args {
				quotedArgs[i] = shellQuoteForBashC(arg)
			}
			originalArgs := strings.Join(quotedArgs, " ")
			var shellCommand string
			if len(container.Args) > 0 {
				shellCommand = fmt.Sprintf("exec %s %s %s", fullCommand, originalArgs, flags)
			} else {
				shellCommand = fmt.Sprintf("exec %s %s", fullCommand, flags)
			}
			container.Command = []string{"sh", "-c"}
			container.Args = []string{shellCommand}
		} else {
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
