package dynamo

import (
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// shellSafeToken matches tokens that are literal to the shell in every context
// and therefore need no quoting inside sh -c.
var shellSafeToken = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

// shellQuotePOSIX renders s as exactly one argv token that survives `sh -c`
// unchanged. Tokens built only from shell-neutral characters pass through
// unquoted for readability; everything else — whitespace, quotes, $, ;, |, &,
// globs, and the empty string — is wrapped in single quotes, inside which every
// byte is literal except the single quote itself, which is closed and re-opened
// via the '\” idiom. Unlike mutation.ShellQuote this is argv-preserving: it
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
