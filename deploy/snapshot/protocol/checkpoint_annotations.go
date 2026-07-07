// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

const (
	linkerdInjectAnnotation = "linkerd.io/inject"
	linkerdInjectDisabled   = "disabled"

	istioSidecarInjectAnnotation = "sidecar.istio.io/inject"
	istioSidecarInjectDisabled   = "false"
)

func DisableCheckpointJobSidecarInjection(annotations map[string]string) map[string]string {
	if annotations == nil {
		annotations = map[string]string{}
	}

	// Checkpoint Jobs complete when the target container exits after the
	// snapshot-agent captures it. Injected service-mesh sidecars can keep the
	// pod active after the checkpoint is written, leaving the Job unfinished.
	annotations[linkerdInjectAnnotation] = linkerdInjectDisabled
	annotations[istioSidecarInjectAnnotation] = istioSidecarInjectDisabled
	return annotations
}
