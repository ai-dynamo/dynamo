/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package compatibility

import (
	semver "github.com/Masterminds/semver/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Gated exposes the metadata and runtime version used to evaluate a
// compatibility gate.
//
// A nil runtime version means that it could not be resolved.
type Gated interface {
	metav1.Object
	RuntimeVersion() *runtimeversion.Version
}

// Predicate decides whether a resource explicitly opts in to a feature.
type Predicate[T any] func(T) bool

// DecisionStatus describes the result of evaluating a compatibility gate.
type DecisionStatus string

const (
	// DecisionEnabled means that the feature may be enabled.
	DecisionEnabled DecisionStatus = "Enabled"
	// DecisionPending means that the runtime supports the feature, but an
	// existing resource has not opted in to the new default.
	DecisionPending DecisionStatus = "Pending"
	// DecisionDisabled means that the runtime cannot be proven to support the feature.
	DecisionDisabled DecisionStatus = "Disabled"
)

// DecisionReason explains why a compatibility gate produced its status.
type DecisionReason string

const (
	// ReasonConstraintsSatisfied means that all configured constraints passed.
	ReasonConstraintsSatisfied DecisionReason = "ConstraintsSatisfied"
	// ReasonExplicitOptIn means that opt-in bypassed the origin constraint.
	ReasonExplicitOptIn DecisionReason = "ExplicitOptIn"
	// ReasonOriginVersionUnsupported means that the origin version did not
	// satisfy the default-enablement constraint.
	ReasonOriginVersionUnsupported DecisionReason = "OriginVersionUnsupported"
	// ReasonRuntimeVersionUnsupported means that the runtime version was
	// unknown or below the hard capability constraint.
	ReasonRuntimeVersionUnsupported DecisionReason = "RuntimeVersionUnsupported"
)

// Decision is the typed result of evaluating a compatibility gate.
type Decision struct {
	Status DecisionStatus
	Reason DecisionReason
}

// Enabled reports whether the feature may be enabled.
func (d Decision) Enabled() bool {
	return d.Status == DecisionEnabled
}

// Pending reports whether the feature is waiting for explicit opt-in.
func (d Decision) Pending() bool {
	return d.Status == DecisionPending
}

// Gate combines the runtime capability and origin default-enablement
// constraints for one feature.
//
// The runtime constraint is a hard requirement. OptIn may bypass an absent or
// older origin version, but it never bypasses the runtime constraint.
type Gate[T Gated] struct {
	Name              string
	MinOriginVersion  *semver.Version
	MinRuntimeVersion *runtimeversion.Version
	OptIn             Predicate[T]
}

// Evaluate applies runtime capability, origin default, and explicit opt-in
// rules in that order.
func (g Gate[T]) Evaluate(resource T) Decision {
	logger := log.Log.WithName("compatibility").WithValues("feature", g.Name)

	// Reject resources whose runtime cannot be proven to support the feature.
	if g.MinRuntimeVersion != nil {
		runtimeVersion := resource.RuntimeVersion()

		// Stop before origin or opt-in evaluation when runtime support is unknown.
		if runtimeVersion == nil || !runtimeVersion.AtLeast(*g.MinRuntimeVersion) {
			logger.V(1).Info("Runtime compatibility constraint not satisfied",
				"runtimeVersion", runtimeVersion,
				"threshold", g.MinRuntimeVersion)

			return Decision{
				Status: DecisionDisabled,
				Reason: ReasonRuntimeVersionUnsupported,
			}
		}
	}

	// Enable resources created after the new default was introduced.
	if g.MinOriginVersion != nil {
		originVersion := g.originVersion(resource)

		// Apply the new default when the resource was created after its introduction.
		if originVersion != nil && originVersion.Compare(g.MinOriginVersion) >= 0 {
			logger.V(1).Info("Compatibility gate constraints satisfied",
				"originVersion", originVersion,
				"originThreshold", g.MinOriginVersion)

			return Decision{
				Status: DecisionEnabled,
				Reason: ReasonConstraintsSatisfied,
			}
		}

		// Allow an explicit opt-in to bypass only the origin constraint.
		if g.OptIn != nil && g.OptIn(resource) {
			logger.V(1).Info("Compatibility gate enabled by explicit opt-in")

			return Decision{
				Status: DecisionEnabled,
				Reason: ReasonExplicitOptIn,
			}
		}

		// Record why a compatible legacy resource is waiting for explicit opt-in.
		logger.V(1).Info("Origin compatibility constraint not satisfied",
			"originVersion", originVersion,
			"threshold", g.MinOriginVersion)

		return Decision{
			Status: DecisionPending,
			Reason: ReasonOriginVersionUnsupported,
		}
	}

	// Enable the feature after every configured constraint has passed.
	return Decision{
		Status: DecisionEnabled,
		Reason: ReasonConstraintsSatisfied,
	}
}

// originVersion resolves the resource's durable operator origin annotation.
func (g Gate[T]) originVersion(resource T) *semver.Version {
	// Preserve the legacy default when no durable origin version is available.
	originValue, exists := resource.GetAnnotations()[consts.KubeAnnotationDynamoOperatorOriginVersion]
	if !exists {
		return nil
	}

	// Treat malformed legacy annotations as an unknown origin version.
	originVersion, err := semver.NewVersion(originValue)
	if err != nil {
		log.Log.WithName("compatibility").WithValues("feature", g.Name).Info(
			"Invalid operator origin version",
			"version", originValue,
			"error", err.Error())

		return nil
	}

	return originVersion
}

// VersionedResource adapts resolved metadata and runtime information to Gated.
type VersionedResource struct {
	metav1.Object
	Runtime *runtimeversion.Version
}

// RuntimeVersion returns the resolved Dynamo runtime compatibility version.
func (r VersionedResource) RuntimeVersion() *runtimeversion.Version {
	return r.Runtime
}

// NewVersionedResource creates a lightweight Gated adapter for legacy call sites.
func NewVersionedResource(object metav1.Object, runtimeVersion *runtimeversion.Version) VersionedResource {
	return VersionedResource{
		Object:  object,
		Runtime: runtimeVersion,
	}
}
