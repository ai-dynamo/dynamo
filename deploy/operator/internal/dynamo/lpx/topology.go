/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var topologyRe = regexp.MustCompile(
	`^(.+?)__(.+?)__([0-9]+)C__G_([0-9_]+?)__(.+?)__GHZ_([0-9_]+?)__(.+?)` +
		`(?:__G_([0-9_]+?))?(?:__(.+?))?$`,
)

var legacyTopologyRe = regexp.MustCompile(`(^|_)(-?\d+)_CHIP(_|$)`)
var topologyChipCountRe = regexp.MustCompile(`__([0-9]+)C__`)

// Topology is a parsed compiler topology with its scheduler-facing chip count.
type Topology struct {
	// ChipCount is the number of LPU chips represented by the topology.
	ChipCount int
	// Raw is the original compiler topology string.
	Raw              string
	compatibilityKey string
}

// Replicas returns the number of replicas (ChipCount / 8, one per node, floor of 1).
// The receiver must be non-nil and is not mutated.
func (t *Topology) Replicas() int {
	return max(1, t.ChipCount/lpuChipsPerNode)
}

// compatibleWith reports whether two topologies share the V2 prop-sync compatibility identity.
func (t Topology) compatibleWith(other Topology) bool {
	return t.compatibilityKey == other.compatibilityKey
}

func (t Topology) withChipCount(chipCount int) (Topology, error) {
	if chipCount <= 0 {
		return Topology{}, fmt.Errorf("chip count must be positive")
	}

	raw := t.Raw
	switch {
	case strings.Contains(raw, "__"):
		raw = topologyChipCountRe.ReplaceAllString(raw, fmt.Sprintf("__%dC__", chipCount))
	case legacyTopologyRe.MatchString(raw):
		raw = legacyTopologyRe.ReplaceAllString(raw, fmt.Sprintf("${1}%d_CHIP${3}", chipCount))
	default:
		return Topology{}, fmt.Errorf("topology %q does not contain a replaceable chip count", raw)
	}

	return parse(raw)
}

func parse(raw string) (Topology, error) {
	if strings.Contains(raw, "__") {
		matches := topologyRe.FindStringSubmatchIndex(raw)
		if matches == nil {
			return Topology{}, fmt.Errorf("topology %q does not match expected pattern", raw)
		}

		chipCount, err := strconv.Atoi(raw[matches[6]:matches[7]])
		if err != nil {
			return Topology{}, fmt.Errorf("parsing chip count from %q: %w", raw, err)
		}

		return Topology{
			ChipCount:        chipCount,
			Raw:              raw,
			compatibilityKey: raw[:matches[6]] + raw[matches[7]:],
		}, nil
	}

	match := strings.Trim(legacyTopologyRe.FindString(raw), "_")
	if match == "" {
		return Topology{}, fmt.Errorf("topology %q does not match expected pattern", raw)
	}

	chipCount, err := strconv.Atoi(strings.TrimSuffix(match, "_CHIP"))
	if err != nil {
		return Topology{}, fmt.Errorf("parsing chip count from %q: %w", raw, err)
	}

	return Topology{
		ChipCount: chipCount,
		Raw:       raw,
	}, nil
}
