// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package injection

// These path constants are mirrored in cmd/ns-bind-mount/main.c as
// ALLOWED_SRC_PREFIX and ALLOWED_DST_PREFIX. Keep both in sync when either changes.
const (
	agentBinDir = "/snapshot-binaries" // mirrored in main.c as ALLOWED_SRC_PREFIX

	SnapshotBinDir = "/tmp" + agentBinDir // mirrored in main.c as ALLOWED_DST_PREFIX
)
