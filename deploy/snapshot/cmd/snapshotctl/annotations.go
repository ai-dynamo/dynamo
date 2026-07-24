// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

// reconcileTargetContainers returns the normalized target-container list. A flag value and
// manifest annotation may both be present only when they match. Callers format the result for
// the target-containers annotation (restore) or set it on PodReference.Containers (capture).
func reconcileTargetContainers(annotations map[string]string, flagValue string, minCount, maxCount int) ([]string, error) {
	flagNames, flagErr := snapshotprotocol.ParseTargetContainers(flagValue)
	if flagErr != nil {
		return nil, fmt.Errorf("--container(s) flag: %w", flagErr)
	}

	manifestRaw := ""
	if annotations != nil {
		manifestRaw = annotations[snapshotprotocol.TargetContainersAnnotation]
	}
	manifestNames, manifestErr := snapshotprotocol.ParseTargetContainers(manifestRaw)
	if manifestErr != nil {
		return nil, fmt.Errorf("manifest %s annotation: %w", snapshotprotocol.TargetContainersAnnotation, manifestErr)
	}

	chosen := flagNames
	if len(flagNames) == 0 {
		chosen = manifestNames
	} else if len(manifestNames) > 0 {
		if snapshotprotocol.FormatTargetContainers(flagNames) != snapshotprotocol.FormatTargetContainers(manifestNames) {
			return nil, fmt.Errorf(
				"--container(s) flag %q does not match manifest %s %q; pass one or the other",
				snapshotprotocol.FormatTargetContainers(flagNames),
				snapshotprotocol.TargetContainersAnnotation,
				snapshotprotocol.FormatTargetContainers(manifestNames),
			)
		}
	}

	if len(chosen) == 0 {
		return nil, fmt.Errorf("target containers are required: pass --container(s) or set %s on the manifest", snapshotprotocol.TargetContainersAnnotation)
	}
	if minCount > 0 && len(chosen) < minCount {
		return nil, fmt.Errorf("expected at least %d target container(s), got %d", minCount, len(chosen))
	}
	if maxCount > 0 && len(chosen) > maxCount {
		return nil, fmt.Errorf("expected at most %d target container(s), got %d", maxCount, len(chosen))
	}
	return chosen, nil
}
