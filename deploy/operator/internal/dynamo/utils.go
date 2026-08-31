package dynamo

import (
	"regexp"
	"strings"
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
