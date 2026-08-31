/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package render

import corev1 "k8s.io/api/core/v1"

// AddFlagsMutation injects an ordered, atomically applied flag set into an
// engine command line.
type AddFlagsMutation struct {
	ContainerName string
	Flags         string
	NeedsShell    bool
	Framework     string
}

func (m AddFlagsMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	if m.Flags == "" {
		return nil
	}
	injectFlagsIntoContainerCommand(container, m.Flags, m.NeedsShell, m.Framework)
	return nil
}
