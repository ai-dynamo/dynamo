/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

// Flag is one declarative command-line flag. Omit allows a resolved backend
// condition to suppress a row without rebuilding the ordered flag set.
type Flag struct {
	Name  string
	Value string
	Omit  bool
}

// AddFlagsMutation atomically injects an ordered flag set into an engine
// command line.
type AddFlagsMutation struct {
	ContainerName string
	Flags         []Flag
	NeedsShell    bool
	Framework     string
}

func (m AddFlagsMutation) Apply(container *corev1.Container) error {
	if err := ValidateContainer(container, m.ContainerName); err != nil {
		return err
	}
	flags, err := FormatFlags(m.Flags)
	if err != nil {
		return err
	}
	if flags == "" {
		return nil
	}
	injectFlagsIntoContainerCommand(container, flags, m.NeedsShell, m.Framework)
	return nil
}

// FormatFlags formats an ordered declarative flag set for command injection.
func FormatFlags(declarations []Flag) (string, error) {
	flags := make([]string, 0, len(declarations))
	for _, flag := range declarations {
		if flag.Omit {
			continue
		}
		if flag.Name == "" {
			return "", fmt.Errorf("flag is empty")
		}
		flags = append(flags, flagAndValue(flag.Name, flag.Value))
	}
	return strings.Join(flags, " "), nil
}

// AddFlagMutation injects one flag/value pair into an engine command line.
type AddFlagMutation struct {
	ContainerName string
	Flag          string
	Value         string
	NeedsShell    bool
	Framework     string
}

func (m AddFlagMutation) Apply(container *corev1.Container) error {
	return (AddFlagsMutation{
		ContainerName: m.ContainerName,
		Flags:         []Flag{{Name: m.Flag, Value: m.Value}},
		NeedsShell:    m.NeedsShell,
		Framework:     m.Framework,
	}).Apply(container)
}

// EnsureFlagMutation adds one flag/value pair unless that exact pair exists.
type EnsureFlagMutation struct {
	ContainerName string
	Flag          string
	Value         string
	NeedsShell    bool
	Framework     string
}

func (m EnsureFlagMutation) Apply(container *corev1.Container) error {
	if ContainerCommandLineHasArg(container, m.Flag, m.Value) {
		return nil
	}
	return (AddFlagMutation{
		ContainerName: m.ContainerName,
		Flag:          m.Flag,
		Value:         m.Value,
		NeedsShell:    m.NeedsShell,
		Framework:     m.Framework,
	}).Apply(container)
}

// EnsureArgsFlagMutation adds one flag/value pair unless that exact pair
// already exists in Args. It preserves the legacy engine mutation behavior
// for backends that intentionally do not inspect Command.
type EnsureArgsFlagMutation struct {
	ContainerName string
	Flag          string
	Value         string
	NeedsShell    bool
	Framework     string
}

func (m EnsureArgsFlagMutation) Apply(container *corev1.Container) error {
	if ContainerHasArg(container, m.Flag, m.Value) {
		return nil
	}
	return (AddFlagMutation{
		ContainerName: m.ContainerName,
		Flag:          m.Flag,
		Value:         m.Value,
		NeedsShell:    m.NeedsShell,
		Framework:     m.Framework,
	}).Apply(container)
}

func flagAndValue(flag, value string) string {
	return strings.TrimSpace(flag + " " + value)
}
