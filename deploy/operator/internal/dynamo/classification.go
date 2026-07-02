// Package: internal/dynamo/classification.go

package dynamo

import "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"

// ComponentReadyClassification is the internal, per-component diagnosis
// computed from Grove PodClique / PodCliqueScalingGroup status fields.
//
// It mirrors the public v1beta1.DGDReadyReason* constants (InsufficientCapacity,
// PodsNotReady, etc.) but is kept as a separate internal type for two reasons:
//
// raw Grove counters (scheduledReplicas, scheduleGatedReplicas) are
// used to *produce* the classification but not copied onto v1beta1.ComponentReplicaStatus
type ComponentReadyClassification string

const (
	// componentReady: the component is fully ready — replicas, scheduling, and
	// rollout all satisfy the desired state.
	componentReady ComponentReadyClassification = "Ready"

	// componentInsufficientCapacity: the component is blocked *before* runtime
	// readiness. Triggered when Grove reports any of:
	//   • status.scheduledReplicas < spec.replicas              (PodClique or PCSG)
	//   • status.scheduleGatedReplicas > 0                      (PodClique only)
	//   • ConditionTypePodCliqueScheduled=False with an
	//     insufficient-capacity reason                          (PodClique)
	//   • ConditionTypeMinAvailableBreached=True with an
	//     insufficient-scheduled-replicas reason                (PCSG)
	componentInsufficientCapacity ComponentReadyClassification = "InsufficientCapacity"

	// componentUpdating: scheduling is sufficient (capacity is available), but
	// the component has not yet finished rolling out updated replicas:
	//   • status.updatedReplicas < spec.replicas
	//   • status.replicas != spec.replicas  (rolling update in progress)
	componentUpdating ComponentReadyClassification = "Updating"

	// componentPodsNotReady: scheduling is sufficient and the rollout is
	// complete, but the component does not yet have enough ready/available
	// replicas at runtime:
	//   • PodClique:  status.readyReplicas < spec.replicas
	//   • PCSG:       status.availableReplicas < spec.replicas
	componentPodsNotReady ComponentReadyClassification = "PodsNotReady"

	// componentUnclassified: the component is not ready, but the controller
	// cannot map the cause into one of the above buckets. Typical causes:
	//   • the Grove resource was not found (not yet created)
	//   • status.observedGeneration is nil or stale (spec not yet processed)
	//   • a client.Get error prevented reading status
	//
	// At the DGD level, componentUnclassified aggregates to
	// v1beta1.DGDReadyReasonSomeResourcesNotReady
	componentUnclassified ComponentReadyClassification = "Unclassified"
)

// classificationToReadyReason converts a single component's internal
// classification to the corresponding public DGD Ready condition reason.
//
// This is only called when *all* not-ready components share the same
// classification (len(classifications) == 1 in aggregateReadyReason). When
// not-ready components have different classifications, the caller uses
// DGDReadyReasonMixedNotReadyReasons directly without going through this
// function.
func classificationToReadyReason(c ComponentReadyClassification) string {
	switch c {
	case componentInsufficientCapacity:
		return v1beta1.DGDReadyReasonInsufficientCapacity
	case componentUpdating:
		return v1beta1.DGDReadyReasonUpdating
	case componentPodsNotReady:
		return v1beta1.DGDReadyReasonPodsNotReady
	case componentReady:
		// Shouldn't be reached: aggregateReadyReason only calls this for
		// not-ready components. Treat as a no-op safe fallback.
		return v1beta1.DGDReadyReasonAllResourcesReady
	default:
		// componentUnclassified, or any future value not yet handled.
		return v1beta1.DGDReadyReasonSomeResourcesNotReady
	}
}

// aggregateReadyReason converts the *set* of distinct per-component
// classifications collected across a whole DGD into a single DGD-level
// Ready condition reason, following the proposal's aggregation rules:
//
//	all components ready             → AllResourcesReady   (caller guards this case)
//	one distinct not-ready class     → that class's reason
//	multiple distinct not-ready      → MixedNotReadyReasons
//	unclassifiable (or empty set)    → SomeResourcesNotReady
//
// It is only called when len(notReadyComponents) > 0, so the empty-set
// branch is a defensive fallback, not a normal code path.
func aggregateReadyReason(classifications map[ComponentReadyClassification]bool) string {
	switch len(classifications) {
	case 0:
		// Defensive: should not happen when there is at least one
		// not-ready component, but guard rather than panic.
		return v1beta1.DGDReadyReasonSomeResourcesNotReady

	case 1:
		for c := range classifications {
			if c == componentUnclassified {
				// A single unclassified component maps to the
				// SomeResourcesNotReady fallback, not to a specific
				// named reason, since the controller has no enough
				// information to be more precise.
				return v1beta1.DGDReadyReasonSomeResourcesNotReady
			}
			return classificationToReadyReason(c)
		}
		// Unreachable, but satisfies the compiler.
		return v1beta1.DGDReadyReasonSomeResourcesNotReady

	default:
		// More than one distinct classification across this DGD's
		// not-ready components: some may be capacity-blocked while
		// others are merely pods-not-ready, for example. Report
		// MixedNotReadyReasons so the operator knows to look at the
		// per-component message for the breakdown.
		return v1beta1.DGDReadyReasonMixedNotReadyReasons
	}
}
