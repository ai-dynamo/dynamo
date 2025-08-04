package dynamo

import (
	"fmt"
	"sort"
)

// Shared helper for applying flag overrides and extra args
func applyFlagOverridesAndExtraArgs(
	baseFlags map[string]*string,
	flagOverrides map[string]*string,
	extraArgs []string,
) []string {
	// Apply overrides/removals
	for flag, value := range flagOverrides {
		if value == nil {
			delete(baseFlags, flag)
		} else {
			baseFlags[flag] = value
		}
	}
	// Convert to args (sorted for deterministic order)
	args := []string{}
	flagNames := make([]string, 0, len(baseFlags))
	for flag := range baseFlags {
		flagNames = append(flagNames, flag)
	}
	sort.Strings(flagNames)
	for _, flag := range flagNames {
		if value := baseFlags[flag]; value != nil {
			args = append(args, fmt.Sprintf("--%s %s", flag, *value))
		}
	}
	// Append extraArgs
	args = append(args, extraArgs...)
	return args
}
