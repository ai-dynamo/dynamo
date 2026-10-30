// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

// ErrorDomain is the google.rpc.ErrorInfo domain of every registry error.
const ErrorDomain = "mcs.dynamo.nvidia.com"

// google.rpc.ErrorInfo reasons. common.proto documents the gRPC code of each.
const (
	ReasonStaleGeneration      = "STALE_GENERATION"
	ReasonLeaseEpochMismatch   = "LEASE_EPOCH_MISMATCH"
	ReasonClusterNotRegistered = "CLUSTER_NOT_REGISTERED"
	ReasonTooManyEndpoints     = "TOO_MANY_ENDPOINTS"
	ReasonTooManyWatchers      = "TOO_MANY_WATCHERS"
	ReasonStoreUnavailable     = "STORE_UNAVAILABLE"
	ReasonShuttingDown         = "SHUTTING_DOWN"

	// ReasonInvalidArgument accompanies INVALID_ARGUMENT errors other than
	// TOO_MANY_ENDPOINTS; a google.rpc.BadRequest detail names the fields.
	ReasonInvalidArgument = "INVALID_ARGUMENT"
	// ReasonNotFound accompanies NOT_FOUND errors for objects other than
	// clusters, for example a ServiceImport.
	ReasonNotFound = "NOT_FOUND"
	// ReasonInternal accompanies INTERNAL errors.
	ReasonInternal = "INTERNAL"
)

// MetadataCurrentGeneration is the ErrorInfo metadata key that carries the
// stored generation, in decimal, of a STALE_GENERATION error.
const MetadataCurrentGeneration = "current_generation"

// ServiceExport condition types, as in sigs.k8s.io/mcs-api.
const (
	ConditionValid    = "Valid"
	ConditionReady    = "Ready"
	ConditionConflict = "Conflict"
)

// Condition reasons. The conflict reasons and Valid, Ready, Pending,
// NoService and InvalidServiceType are the sigs.k8s.io/mcs-api constants;
// IPFamilyConflict and ClusterLeaseExpired are registry additions.
const (
	ConditionReasonValid              = "Valid"
	ConditionReasonReady              = "Ready"
	ConditionReasonPending            = "Pending"
	ConditionReasonNoService          = "NoService"
	ConditionReasonInvalidServiceType = "InvalidServiceType"

	ConditionReasonNoConflicts                   = "NoConflicts"
	ConditionReasonPortConflict                  = "PortConflict"
	ConditionReasonTypeConflict                  = "TypeConflict"
	ConditionReasonSessionAffinityConflict       = "SessionAffinityConflict"
	ConditionReasonSessionAffinityConfigConflict = "SessionAffinityConfigConflict"
	ConditionReasonAnnotationsConflict           = "AnnotationsConflict"
	ConditionReasonLabelsConflict                = "LabelsConflict"
	ConditionReasonInternalTrafficPolicyConflict = "InternalTrafficPolicyConflict"
	ConditionReasonTrafficDistributionConflict   = "TrafficDistributionConflict"
	ConditionReasonIPFamilyConflict              = "IPFamilyConflict"
	ConditionReasonClusterLeaseExpired           = "ClusterLeaseExpired"

	// ConditionReasonFailed is the mcs-api reason of Ready=False after a
	// registry write failed or the export is invalid.
	ConditionReasonFailed = "Failed"
	// ConditionReasonApplyFailed is the reason of Ready=False on a
	// ServiceImport whose objects the controller could not apply.
	ConditionReasonApplyFailed = "ApplyFailed"
	// ConditionReasonUnknown replaces a registry reason the Kubernetes API
	// would reject.
	ConditionReasonUnknown = "Unknown"
)

// API limits. Requests over a limit fail with INVALID_ARGUMENT, except
// endpoint sets, which fail with reason TOO_MANY_ENDPOINTS.
const (
	MaxLabels             = 64
	MaxAnnotations        = 32
	MaxAnnotationBytes    = 16 << 10
	MaxPorts              = 64
	MaxEndpointsPerExport = 1000
	MaxEndpointAttributes = 16
	MaxMessageBytes       = 4 << 20
)
