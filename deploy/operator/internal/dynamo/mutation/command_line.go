/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

var pythonCommandPattern = regexp.MustCompile(`^(.*/)?(python\d*(\.\d+)*)$`)

// IsPythonCommand reports whether command invokes a Python interpreter.
func IsPythonCommand(command string) bool {
	return pythonCommandPattern.MatchString(command)
}

// ShellQuote quotes a string so it survives shell interpretation inside sh -c.
// Simple args (flags, paths) pass through unchanged; args containing special
// characters (JSON, env vars, spaces, quotes) are wrapped in double quotes with
// inner escaping.
func ShellQuote(value string) string {
	if !strings.ContainsAny(value, " \t\n'\"\\{}[]$`!") {
		return value
	}
	escaped := strings.ReplaceAll(value, `\`, `\\`)
	escaped = strings.ReplaceAll(escaped, `"`, `\"`)
	escaped = strings.ReplaceAll(escaped, `$`, `\$`)
	escaped = strings.ReplaceAll(escaped, "`", "\\`")
	escaped = strings.ReplaceAll(escaped, "'", `'"'"'`)
	return `"` + escaped + `"`
}

// ContainerHasArg reports whether the container already carries the given
// flag/value pair in Args, either as adjacent tokens "flag", "value" or as a
// single token "flag=value" or "flag value" embedded inside a shell string.
// It is used to make flag injection idempotent.
func ContainerHasArg(container *corev1.Container, flag, value string) bool {
	return container != nil && hasArg(container.Args, flag, value)
}

// ContainerCommandLineHasArg reports whether Command or Args contains an exact
// flag/value pair, including pairs embedded in a shell command string.
func ContainerCommandLineHasArg(container *corev1.Container, flag, value string) bool {
	if container == nil {
		return false
	}
	commandLine := make([]string, 0, len(container.Command)+len(container.Args))
	commandLine = append(commandLine, container.Command...)
	commandLine = append(commandLine, container.Args...)
	if hasArg(commandLine, flag, value) {
		return true
	}

	expandedCommandLine := make([]string, 0, len(commandLine))
	for _, arg := range commandLine {
		expandedCommandLine = append(expandedCommandLine, strings.Fields(arg)...)
	}
	return hasArg(expandedCommandLine, flag, value)
}

func hasArg(args []string, flag, value string) bool {
	joined := strings.TrimSpace(flag + " " + value)
	equals := flag + "=" + value
	for i, arg := range args {
		if strings.Contains(arg, joined) || value != "" && strings.Contains(arg, equals) {
			return true
		}
		if arg == flag && (value == "" || i+1 < len(args) && args[i+1] == value) {
			return true
		}
	}
	return false
}

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
func injectFlagsIntoContainerCommand(container *corev1.Container, flags string, needsShell bool, framework, executable, subcommand string) {
	if len(container.Command) > 0 && IsPythonCommand(container.Command[0]) {
		if needsShell {
			quotedCommand := make([]string, len(container.Command))
			for i, token := range container.Command {
				quotedCommand[i] = ShellQuote(token)
			}
			quotedArgs := make([]string, len(container.Args))
			for i, arg := range container.Args {
				quotedArgs[i] = ShellQuote(arg)
			}
			commandLine := strings.Join(quotedCommand, " ")
			if len(quotedArgs) > 0 {
				commandLine += " " + strings.Join(quotedArgs, " ")
			}
			container.Command = []string{"sh", "-c"}
			container.Args = []string{fmt.Sprintf("exec %s %s", commandLine, flags)}
			return
		}
		container.Args = append(container.Args, strings.Fields(flags)...)
		return
	}

	for i, arg := range container.Args {
		modified := arg
		if framework != "" {
			modified = injectFlagsIntoPythonCommand(arg, flags, framework)
		}
		if modified == arg && executable != "" {
			modified = injectFlagsIntoDirectCommand(arg, flags, executable, subcommand)
		}
		if modified != arg {
			container.Args[i] = modified
			return
		}
	}
}

func injectFlagsIntoDirectCommand(command, flags, executable, subcommand string) string {
	directCommand := strings.TrimSpace(executable + " " + subcommand)
	pattern := fmt.Sprintf(`(^|\s)((?:exec\s+)?%s(?:\s+[^|&;]*)?)(\s|$|[|&;])`, regexp.QuoteMeta(directCommand))
	re := regexp.MustCompile(pattern)
	return re.ReplaceAllStringFunc(command, func(match string) string {
		submatches := re.FindStringSubmatch(match)
		if len(submatches) < 4 {
			return match
		}
		prefix := submatches[1]
		engineCommand := strings.TrimSpace(submatches[2])
		separator := submatches[3]
		return fmt.Sprintf("%s%s %s%s", prefix, engineCommand, flags, separator)
	})
}

func isShellCommand(command string) bool {
	command = strings.TrimPrefix(command, "/bin/")
	return command == "sh" || command == "bash"
}

func injectFlagsIntoPythonCommand(command, flags, framework string) string {
	pattern := fmt.Sprintf(`(python[0-9.]*\s+[^|&;]*%s[^|&;]*?)(\s|$|[|&;])`, framework)
	re := regexp.MustCompile(pattern)
	return re.ReplaceAllStringFunc(command, func(match string) string {
		submatches := re.FindStringSubmatch(match)
		if len(submatches) < 3 {
			return match
		}
		pythonCommand := strings.TrimSpace(submatches[1])
		separator := submatches[2]
		return fmt.Sprintf("%s %s%s", pythonCommand, flags, separator)
	})
}
