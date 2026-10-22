package dynamo

import (
	"fmt"
	"regexp"
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"mvdan.cc/sh/v3/expand"
	"mvdan.cc/sh/v3/syntax"
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

// shellSafeToken matches tokens that are literal to the shell in every context
// and therefore need no quoting inside sh -c.
var shellSafeToken = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

// shellQuotePOSIX renders s as exactly one argv token that survives `sh -c`
// unchanged. Tokens built only from shell-neutral characters pass through
// unquoted for readability; everything else — whitespace, quotes, $, ;, |, &,
// globs, and the empty string — is wrapped in single quotes, inside which every
// byte is literal except the single quote itself, which is closed and re-opened
// via the '\” idiom. Unlike shellQuoteForBashC this is argv-preserving: it
// round-trips arbitrary tokens (including empty ones and embedded quotes)
// through the shell without splitting, dropping, or reinterpreting them.
func shellQuotePOSIX(s string) string {
	if shellSafeToken.MatchString(s) {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// findEnvVar returns the named environment variable entry, or nil when absent. The
// entry, not its value, so a valueFrom variable is distinguishable from an absent one.
func findEnvVar(env []corev1.EnvVar, name string) *corev1.EnvVar {
	for i := range env {
		if env[i].Name == name {
			return &env[i]
		}
	}
	return nil
}

func findContainerPort(container *corev1.Container, name string) *corev1.ContainerPort {
	for i := range container.Ports {
		if container.Ports[i].Name == name {
			return &container.Ports[i]
		}
	}
	return nil
}

// containerHasArg reports whether the container already carries the given
// flag/value pair in its Args (either as adjacent tokens "flag", "value" or
// as a single token "flag=value" or "flag value" embedded inside a shell
// string). It is used to make flag injection idempotent.
func containerHasArg(container *corev1.Container, flag, value string) bool {
	if container == nil {
		return false
	}
	return hasArg(container.Args, flag, value)
}

func hasArg(args []string, flag, value string) bool {
	joined := flag + " " + value
	equals := flag + "=" + value
	for i, arg := range args {
		if strings.Contains(arg, joined) || strings.Contains(arg, equals) {
			return true
		}
		if arg == flag && i+1 < len(args) && args[i+1] == value {
			return true
		}
	}
	return false
}

// launchArgument preserves whether a shell word can become multiple engine arguments.
type launchArgument struct {
	Value     string
	MayExpand bool
}

// pythonModuleArgs reads a direct Python module launch or a single sh/bash -c command.
// container must not be nil. Shell commands are parsed, never executed; dynamic words
// remain opaque so the backend can decide whether they affect its topology.
func pythonModuleArgs(container *corev1.Container, module string) ([]launchArgument, error) {
	commandLine := append(slices.Clone(container.Command), container.Args...)
	command := make([]launchArgument, 0, len(commandLine))
	for _, arg := range commandLine {
		command = append(command, launchArgument{Value: arg})
	}

	// Preserve argv boundaries for direct launches; only a shell script needs tokenization.
	if len(command) >= 3 && isShellCommand(command[0].Value) && strings.HasPrefix(command[1].Value, "-") && strings.Contains(command[1].Value, "c") {
		var err error
		command, _, err = shellPythonCommand(command[2].Value)
		if err != nil {
			return nil, err
		}
	}

	// Match the module itself rather than text in a wrapper argument or shell comment.
	if len(command) < 3 || !isPythonCommand(command[0].Value) || command[1].Value != "-m" || command[2].Value != module {
		return nil, fmt.Errorf("expected python -m %s", module)
	}
	return command[3:], nil
}

// shellPythonCommand reads one Python module invocation, optionally prefixed by exec.
// It returns the call node so injection can insert flags without rewriting user text.
func shellPythonCommand(script string) ([]launchArgument, *syntax.CallExpr, error) {
	file, err := syntax.NewParser().Parse(strings.NewReader(script), "")
	if err != nil {
		return nil, nil, fmt.Errorf("invalid shell launch: %w", err)
	}
	if len(file.Stmts) != 1 || file.Stmts[0].Background || file.Stmts[0].Negated {
		return nil, nil, fmt.Errorf("expected a single Python module launch in the shell command")
	}
	call, ok := file.Stmts[0].Cmd.(*syntax.CallExpr)
	if !ok {
		return nil, nil, fmt.Errorf("expected a Python module launch, not shell control flow")
	}

	// Preserve unresolved words; unquote only literals, without reading the host environment.
	command := make([]launchArgument, 0, len(call.Args))
	for _, word := range call.Args {
		dynamic := false
		mayExpand := false
		for _, part := range word.Parts {
			switch part := part.(type) {
			case *syntax.SglQuoted, *syntax.DblQuoted:
			case *syntax.Lit:
				mayExpand = mayExpand || strings.ContainsAny(part.Value, "*?[")
			default:
				mayExpand = true
			}
		}
		syntax.Walk(word, func(node syntax.Node) bool {
			// Quoted arrays can still produce multiple arguments (for example "$@").
			if param, ok := node.(*syntax.ParamExp); ok && (param.Index != nil || param.Param.Value == "@" || param.Param.Value == "*" || param.Names != 0) {
				mayExpand = true
			}
			switch node.(type) {
			case nil, *syntax.Word, *syntax.Lit, *syntax.SglQuoted, *syntax.DblQuoted:
				return true
			default:
				dynamic = true
				return false
			}
		})
		value := script[word.Pos().Offset():word.End().Offset()]
		if !dynamic {
			value, err = expand.Literal(&expand.Config{Env: expand.ListEnviron()}, word)
			if err != nil {
				return nil, nil, err
			}
		}
		command = append(command, launchArgument{Value: value, MayExpand: mayExpand})
	}

	// A wrapper argument mentioning Python is not an engine invocation.
	if len(command) > 0 && command[0].Value == "exec" {
		command = command[1:]
	}
	if len(command) < 3 || !isPythonCommand(command[0].Value) || command[1].Value != "-m" {
		return nil, nil, fmt.Errorf("expected a Python module launch")
	}
	return command, call, nil
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
		// Shell scripts can be authored in Args or Command; keep the existing Args-first behavior.
		tokenLists := [][]string{container.Args}
		if framework == "sglang" {
			tokenLists = append(tokenLists, container.Command)
		}
		for _, tokens := range tokenLists {
			for i, arg := range tokens {
				// Native SGLang placement and injection must read the same shell invocation.
				if framework == "sglang" {
					command, call, err := shellPythonCommand(arg)
					if err == nil && command[2].Value == "sglang.launch_server" {
						// Preserve existing shell precedence: authored flags follow injected flags.
						moduleIndex := len(call.Args) - len(command) + 2
						end := call.Args[moduleIndex].End().Offset()
						tokens[i] = arg[:end] + " " + flags + arg[end:]
						return
					}
				}
				modifiedArg := injectFlagsIntoPythonCommand(arg, flags, framework)
				if modifiedArg != arg {
					tokens[i] = modifiedArg
					return
				}
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
