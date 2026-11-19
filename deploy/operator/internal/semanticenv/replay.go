/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package semanticenv

import (
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// Value is an expansion-time value. Known is false when it depends on an
// unavailable runtime input; a known empty string is distinct from an unknown.
type Value struct {
	Value string
	Known bool
}

// Bound total materialized output: repeated aliases can otherwise expand
// exponentially even when the authored Kubernetes object is small.
const maxReplayBytes = 1 << 20

// Replay evaluates an already composed env list in Kubernetes order without
// mutating it. Missing references and valueFrom remain unknown; no process,
// image, envFrom, or provider environment is assumed. Map presence distinguishes
// an absent name from a declared unknown value. Inserted values are not re-expanded.
// Values exceeding the shared 1 MiB output budget also remain unknown.
func Replay(env []corev1.EnvVar) map[string]Value {
	values := make(map[string]Value, len(env))
	remaining := maxReplayBytes
	for _, variable := range env {
		// An unreadable definition must shadow any earlier known value.
		if variable.ValueFrom != nil {
			values[variable.Name] = Value{}
			continue
		}

		// Resolve against the state at this entry, never the final name map.
		known := true
		value, complete := expand(variable.Value, remaining, func(name string) string {
			if previous, exists := values[name]; exists {
				known = known && previous.Known
				return previous.Value
			}
			known = false
			return "$(" + name + ")"
		})
		remaining -= len(value)
		values[variable.Name] = Value{Value: value, Known: known && complete}
	}
	return values
}

// expand recognizes Kubernetes dollar escaping and parenthesized references.
func expand(value string, limit int, lookup func(string) string) (string, bool) {
	var result strings.Builder
	lastCloser := strings.LastIndexByte(value, ')')
	for i := 0; i < len(value); {
		part := value[i : i+1]
		if value[i] != '$' || i+1 == len(value) {
			i++
		} else {
			switch value[i+1] {
			case '$':
				i += 2
			case '(':
				// Avoid rescanning an unterminated suffix for every nested opener.
				if i >= lastCloser {
					part = "$("
					i += 2
				} else {
					end := strings.IndexByte(value[i+2:], ')')
					part = lookup(value[i+2 : i+2+end])
					i += end + 3
				}
			default:
				i++
			}
		}

		// An oversized expansion is an unavailable runtime value, not an error.
		if len(part) > limit-result.Len() {
			return result.String(), false
		}
		result.WriteString(part)
	}
	return result.String(), true
}
