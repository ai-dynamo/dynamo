/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package render

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

func validateContainerMutation(container *corev1.Container, containerName string) error {
	if container == nil {
		return fmt.Errorf("container is nil")
	}
	if containerName != "" && container.Name != "" && container.Name != containerName {
		return fmt.Errorf("mutation targets container %q, got %q", containerName, container.Name)
	}
	return nil
}

// AppendEnvMutation appends an environment variable to a container.
type AppendEnvMutation struct {
	ContainerName string
	Env           corev1.EnvVar
}

func (m AppendEnvMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	container.Env = append(container.Env, *m.Env.DeepCopy())
	return nil
}

// RemoveProbesMutation removes selected probes from a container.
type RemoveProbesMutation struct {
	ContainerName string
	Liveness      bool
	Readiness     bool
	Startup       bool
}

func (m RemoveProbesMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	if m.Liveness {
		container.LivenessProbe = nil
	}
	if m.Readiness {
		container.ReadinessProbe = nil
	}
	if m.Startup {
		container.StartupProbe = nil
	}
	return nil
}

// SetReadinessProbeMutation replaces a container's readiness probe.
type SetReadinessProbeMutation struct {
	ContainerName string
	Probe         *corev1.Probe
}

func (m SetReadinessProbeMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	container.ReadinessProbe = m.Probe.DeepCopy()
	return nil
}

// AppendVolumeMountMutation appends a volume mount to a container.
type AppendVolumeMountMutation struct {
	ContainerName string
	VolumeMount   corev1.VolumeMount
}

func (m AppendVolumeMountMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	container.VolumeMounts = append(container.VolumeMounts, *m.VolumeMount.DeepCopy())
	return nil
}

// SetCommandMutation replaces a container's command and arguments.
type SetCommandMutation struct {
	ContainerName string
	Command       []string
	Args          []string
}

func (m SetCommandMutation) Apply(container *corev1.Container) error {
	if err := validateContainerMutation(container, m.ContainerName); err != nil {
		return err
	}
	container.Command = append([]string(nil), m.Command...)
	container.Args = append([]string(nil), m.Args...)
	return nil
}
