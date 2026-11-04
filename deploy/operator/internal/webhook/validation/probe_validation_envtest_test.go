/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type probeAdmissionCase struct {
	name          string
	deployment    runtime.Object
	oldDeployment runtime.Object
	wantCELErr    string
}

// containerProbeAdmissionCases mutates the container in the caller-owned fixture,
// returning independent snapshots for the resource's existing admission table.
func containerProbeAdmissionCases(deployment runtime.Object, container *corev1.Container, path *field.Path) []probeAdmissionCase {
	// Exercise omission, both invalid handler counts, and every supported handler.
	handlers := []struct {
		name    string
		probe   *corev1.Probe
		invalid bool
	}{
		{name: "omitted"},
		{name: "timing only", probe: &corev1.Probe{PeriodSeconds: 5}, invalid: true},
		{name: "multiple handlers", probe: &corev1.Probe{
			Exec: &corev1.ExecAction{Command: []string{"true"}}, GRPC: &corev1.GRPCAction{Port: 8000}}, invalid: true},
		{name: "exec", probe: &corev1.Probe{Exec: &corev1.ExecAction{Command: []string{"true"}}}},
		{name: "http", probe: &corev1.Probe{HTTPGet: &corev1.HTTPGetAction{Path: "/live", Port: intstr.FromInt(8000)}}},
		{name: "tcp", probe: &corev1.Probe{TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(8000)}}},
		{name: "grpc", probe: &corev1.Probe{GRPC: &corev1.GRPCAction{Port: 8000}}},
	}

	// Apply each probe independently, with an omitted probe as the update seed.
	baseline := deployment.DeepCopyObject()
	var tests []probeAdmissionCase
	for _, probe := range []struct {
		name  string
		value **corev1.Probe
	}{
		{name: "startupProbe", value: &container.StartupProbe},
		{name: "livenessProbe", value: &container.LivenessProbe},
		{name: "readinessProbe", value: &container.ReadinessProbe},
	} {
		for _, handler := range handlers {
			*probe.value = handler.probe.DeepCopy()
			for _, operation := range []string{"create", "update"} {
				test := probeAdmissionCase{
					name: fmt.Sprintf("%s %s %s %s", deployment.GetObjectKind().GroupVersionKind().Version,
						path.Child(probe.name), operation, handler.name),
					deployment: deployment.DeepCopyObject(),
				}
				if operation == "update" {
					test.oldDeployment = baseline.DeepCopyObject()
				}
				if handler.invalid {
					test.wantCELErr = fmt.Sprintf("%s: Invalid value: %s must define exactly one handler", path.Child(probe.name), probe.name)
				}
				tests = append(tests, test)
			}
		}
		*probe.value = nil
	}
	return tests
}
