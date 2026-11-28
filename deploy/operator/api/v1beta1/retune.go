// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

// RetuneSpec requests operator-owned semantic startup configuration for one Planner.
// Search domains, intervals, scoring, and replica scaling remain Planner configuration.
type RetuneSpec struct {
	// enabled configures the frontend selected by type in the same DGD and the
	// Planner with shared trace storage, a reachable frontend admin listener, and
	// KV routing with replica synchronization and the supplied startup values.
	// It does not enable replica scaling or choose an optimization strategy.
	// Omission or false leaves ordinary component rendering unchanged. Enabling,
	// disabling, or changing semantic settings changes the affected Pod templates.
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// traceStorage references the user-provided trace PVC. Required when enabled.
	// The operator owns the mounts, trace layout, and associated environment
	// bindings. Component status acknowledges the semantic target contract's hash.
	// +optional
	TraceStorage *RetuneTraceStorageSpec `json:"traceStorage,omitempty"`

	// router supplies initial serving values, not Planner search domains. The
	// operator injects these through existing frontend environment variables.
	// Explicit CLI arguments and router-policy configuration retain their normal
	// precedence; effective runtime validation is not provided by this setting.
	// Omission uses overlapScoreCredit 1, prefillLoadScale 1, and temperature 0.
	// +optional
	Router *RetuneRouterSpec `json:"router,omitempty"`
}

// RetuneRouterSpec supplies the supported startup settings for Retune.
type RetuneRouterSpec struct {
	// overlapScoreCredit is the initial KV overlap credit. Defaults to 1.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	OverlapScoreCredit *float64 `json:"overlapScoreCredit,omitempty"`

	// prefillLoadScale is the initial prefill-load weight. Defaults to 1.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	PrefillLoadScale *float64 `json:"prefillLoadScale,omitempty"`

	// temperature is the initial router sampling temperature. Defaults to 0.
	// +optional
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	Temperature *float64 `json:"temperature,omitempty"`
}

// RetuneTraceStorageSpec references storage shared by the frontend and Planner.
type RetuneTraceStorageSpec struct {
	// claimName is an existing writable ReadWriteMany PVC in the DGD namespace.
	// The operator does not create the claim or wait for it during admission.
	// +required
	// +kubebuilder:validation:MinLength=1
	ClaimName string `json:"claimName"`
}

// RetuneStatus acknowledges synchronized semantic target configuration for a Planner.
// It is associated with the resource's status.observedGeneration and does not
// assert runtime readiness or the effective configuration of running processes.
type RetuneStatus struct {
	// configHash is the opaque binding of the operator-generated launch contract.
	// Publication acknowledges that the frontend and Planner target lowerings
	// have been applied, independently of readiness or optional role projections.
	// The Planner also matches its immutable launch binding before starting
	// tuning windows; this hash does not acknowledge effective runtime settings.
	// +required
	// +kubebuilder:validation:MinLength=1
	ConfigHash string `json:"configHash"`
}
