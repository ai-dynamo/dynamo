package dynamo

import (
	"regexp"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
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
	return mutation.ShellQuote(s)
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

// containerHasArg reports whether the container already carries the given
// flag/value pair in its Args (either as adjacent tokens "flag", "value" or
// as a single token "flag=value" or "flag value" embedded inside a shell
// string). It is used to make flag injection idempotent.
func containerHasArg(container *corev1.Container, flag, value string) bool {
	return mutation.ContainerHasArg(container, flag, value)
}

func containerCommandLineHasArg(container *corev1.Container, flag, value string) bool {
	return mutation.ContainerCommandLineHasArg(container, flag, value)
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
