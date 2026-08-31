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

// ShellQuote quotes one command token for a generated sh -c command.
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

// ContainerHasArg reports whether Args contains an exact flag/value pair.
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

func injectFlagsIntoContainerCommand(container *corev1.Container, flags string, needsShell bool, framework string) {
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
		modified := injectFlagsIntoPythonCommand(arg, flags, framework)
		if modified != arg {
			container.Args[i] = modified
			return
		}
	}
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
