// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package merge computes a ServiceImport from the exports of one Service in
// every cluster of a clusterset. It implements the conflict resolution of
// KEP-1645 (the oldest export wins, one reason per conflicting field) as
// expressed by the condition reasons of sigs.k8s.io/mcs-api, and runs it once,
// server side, so every cluster sees the same answer.
//
// Rules:
//
//   - Only exports of clusters whose lease is active take part. They are
//     ordered oldest first by export creation timestamp, then by cluster
//     name; the first one wins.
//   - Ports are the union by (port, protocol) in priority order. A name or
//     app protocol disagreement resolves to the oldest export; a port whose
//     name is already used by another (port, protocol) is skipped.
//   - IP families are intersected; an export with no family in common with
//     the running intersection is skipped.
//   - Type, session affinity, its client IP timeout, annotations, labels,
//     internal traffic policy and traffic distribution come from the winner.
//
// A conflict is reported on the import and, as KEP-1645 asks, in the Conflict
// condition of every participating cluster. ClusterResult.Conflict
// additionally names the clusters whose own export disagrees with the winner.
package merge

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/store"
)

// Export is the export of the Service from one cluster.
type Export struct {
	Cluster string
	// LeaseActive is false when the cluster's lease has expired or the
	// cluster is not registered. Such an export does not take part in
	// conflict resolution and contributes no endpoints.
	LeaseActive bool
	Spec        store.ExportSpec
}

// Conflict describes a disagreement between exports.
type Conflict struct {
	// Reason is one of the api.ConditionReason*Conflict constants.
	Reason  string
	Message string
	// WinningCluster is the cluster whose export won.
	WinningCluster string
}

// ConditionStatus is a Kubernetes condition status.
type ConditionStatus string

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

// Condition is a ServiceExport condition without its transition time; the
// caller tracks transitions.
type Condition struct {
	Type    string
	Status  ConditionStatus
	Reason  string
	Message string
}

// ClusterResult is the outcome of the merge for one exporting cluster.
type ClusterResult struct {
	Cluster string
	// Participating is true when the cluster's export took part in the
	// merge (its lease is active).
	Participating bool
	// Conflict is set when this cluster's own export disagrees with the
	// winning export. It is never set for the winner.
	Conflict *Conflict
	// Conditions are Valid, Ready and Conflict, in that order.
	Conditions []Condition
}

// Result is a merged ServiceImport.
type Result struct {
	// Winner is the cluster of the oldest participating export, or empty
	// when no export participates.
	Winner string
	// Spec is the merged spec. When no export participates it is merged
	// from every export, so the shape of an import whose clusters all
	// expired stays stable until compaction removes the records.
	Spec store.ExportSpec
	// Conflict is set when any participating export conflicts.
	Conflict *Conflict
	// Clusters has one entry per input export, sorted by cluster name.
	Clusters []ClusterResult
}

// Merge merges the exports of one Service. Exports must name distinct
// clusters. An empty input yields an empty Result.
func Merge(exports []Export) Result {
	if len(exports) == 0 {
		return Result{}
	}
	ordered := Participants(exports)
	participating := len(ordered) > 0
	if !participating {
		ordered = slices.Clone(exports)
		OrderByPriority(ordered)
	}
	winner := ordered[0]

	ports, reason, msg := MergePorts(ordered)
	ipFamilies, ipReason, ipMsg := IntersectIPFamilies(ordered)
	if reason == api.ConditionReasonNoConflicts {
		reason, msg = ipReason, ipMsg
	}
	if reason == api.ConditionReasonNoConflicts {
		reason, msg = CheckConflictExport(ordered)
	}

	res := Result{Spec: winner.Spec.Clone()}
	res.Spec.Ports = ports
	res.Spec.IPFamilies = ipFamilies
	if participating {
		res.Winner = winner.Cluster
		if reason != api.ConditionReasonNoConflicts {
			res.Conflict = &Conflict{Reason: reason, Message: msg, WinningCluster: winner.Cluster}
		}
	}

	skippedFamilies := ipFamilySkips(ordered)
	for _, e := range exports {
		cr := ClusterResult{Cluster: e.Cluster, Participating: participating && e.LeaseActive}
		if cr.Participating && e.Cluster != winner.Cluster {
			cr.Conflict = clusterConflict(winner, e, skippedFamilies[e.Cluster])
		}
		cr.Conditions = conditions(e.Cluster, cr.Participating, res.Conflict)
		res.Clusters = append(res.Clusters, cr)
	}
	slices.SortFunc(res.Clusters, func(a, b ClusterResult) int { return cmp.Compare(a.Cluster, b.Cluster) })
	return res
}

// Participants returns the exports of clusters with an active lease, ordered
// by priority (see OrderByPriority): the same ordering with expired clusters
// skipped.
func Participants(exports []Export) []Export {
	var out []Export
	for _, e := range exports {
		if e.LeaseActive {
			out = append(out, e)
		}
	}
	OrderByPriority(out)
	return out
}

// OrderByPriority sorts exports oldest first by export creation timestamp,
// breaking ties by cluster name.
func OrderByPriority(exports []Export) {
	slices.SortStableFunc(exports, func(a, b Export) int {
		return cmp.Or(
			a.Spec.ExportCreationTimestamp.Compare(b.Spec.ExportCreationTimestamp),
			cmp.Compare(a.Cluster, b.Cluster),
		)
	})
}

type portMerge struct {
	store.ServicePort
	cluster string
}

func protocolOf(p store.Protocol) store.Protocol {
	if p == "" {
		return store.ProtocolTCP
	}
	return p
}

func isSimilarPort(a, b store.ServicePort) bool {
	return a.Port == b.Port && protocolOf(a.Protocol) == protocolOf(b.Protocol)
}

func formatPort(p store.ServicePort) string {
	s := fmt.Sprintf("%d/%s", p.Port, protocolOf(p.Protocol))
	if p.Name != "" {
		s += fmt.Sprintf(" name %q", p.Name)
	}
	if p.AppProtocol != "" {
		s += fmt.Sprintf(" appProtocol %q", p.AppProtocol)
	}
	return s
}

// duplicatedPortNameConflict reports a port whose name is already used by
// another (port, protocol) of an older export.
func duplicatedPortNameConflict(port portMerge, byName map[string]portMerge) string {
	older, ok := byName[port.Name]
	if ok && !isSimilarPort(port.ServicePort, older.ServicePort) {
		return fmt.Sprintf(
			"Duplicated port name %q. Using port definition %q from oldest service export in cluster %q.",
			port.Name, formatPort(older.ServicePort), older.cluster,
		)
	}
	return ""
}

func portConflict(port, older portMerge) string {
	if port.Name != older.Name {
		return fmt.Sprintf(
			"Conflicting port name on %q. Using the port name %q from oldest service export in cluster %q.",
			formatPort(port.ServicePort), older.Name, older.cluster,
		)
	}
	if port.AppProtocol != older.AppProtocol {
		return fmt.Sprintf(
			"Conflicting appProtocol on %q. Using the appProtocol %q from oldest service export in cluster %q.",
			formatPort(port.ServicePort), older.AppProtocol, older.cluster,
		)
	}
	return ""
}

// MergePorts returns the union of the ports of exports ordered by priority,
// keyed by (port, protocol), with the conflict reason and message of the
// first conflict found. Name and app protocol conflicts resolve to the
// oldest export; a port with a name already used by another (port,
// protocol) is skipped. When no other conflict exists, an export whose port
// set differs from the union is a PortConflict. That last comparison ignores
// port order.
func MergePorts(ordered []Export) ([]store.ServicePort, string, string) {
	conflictMsg := ""
	var merged []portMerge
	byName := map[string]portMerge{}
	for _, e := range ordered {
		for _, port := range e.Spec.Ports {
			pm := portMerge{ServicePort: port, cluster: e.Cluster}
			if msg := duplicatedPortNameConflict(pm, byName); msg != "" {
				if conflictMsg == "" {
					conflictMsg = msg
				}
				continue
			}
			i := slices.IndexFunc(merged, func(m portMerge) bool { return isSimilarPort(port, m.ServicePort) })
			if i == -1 {
				// byName is never overwritten so that it keeps the oldest
				// export of every port name.
				byName[port.Name] = pm
				merged = append(merged, pm)
			} else if conflictMsg == "" {
				conflictMsg = portConflict(pm, merged[i])
			}
		}
	}

	ports := make([]store.ServicePort, 0, len(merged))
	for _, m := range merged {
		ports = append(ports, m.ServicePort)
	}
	if conflictMsg == "" {
		for _, e := range ordered {
			if !samePortSet(ports, e.Spec.Ports) {
				conflictMsg = fmt.Sprintf(
					"Ports from cluster %q do not match ports of oldest service export in cluster %q.",
					e.Cluster, ordered[0].Cluster,
				)
				break
			}
		}
	}
	if conflictMsg != "" {
		return ports, api.ConditionReasonPortConflict, conflictMsg
	}
	return ports, api.ConditionReasonNoConflicts, ""
}

func samePort(a, b store.ServicePort) bool {
	return isSimilarPort(a, b) && a.Name == b.Name && a.AppProtocol == b.AppProtocol
}

func samePortSet(a, b []store.ServicePort) bool {
	if len(a) != len(b) {
		return false
	}
	for _, p := range a {
		if !slices.ContainsFunc(b, func(q store.ServicePort) bool { return samePort(p, q) }) {
			return false
		}
	}
	return true
}

// IntersectIPFamilies returns the intersection of the IP families of exports
// ordered by priority, keeping the order of the oldest export. An export with
// no family in common with the running intersection is skipped, so that the
// import never ends up without a family, and reported as an
// IPFamilyConflict. An export that lists no IP family (a writer that does not
// know them) does not constrain the intersection.
func IntersectIPFamilies(ordered []Export) ([]store.IPFamily, string, string) {
	families, skipped := intersect(ordered)
	if len(skipped) > 0 {
		return families, api.ConditionReasonIPFamilyConflict,
			fmt.Sprintf("IPFamilies conflict. Cluster %q has no IPFamilies in common.", skipped[0])
	}
	return families, api.ConditionReasonNoConflicts, ""
}

func intersect(ordered []Export) ([]store.IPFamily, []string) {
	var families []store.IPFamily
	var skipped []string
	for _, e := range ordered {
		if len(e.Spec.IPFamilies) == 0 {
			continue
		}
		if families == nil {
			families = slices.Clone(e.Spec.IPFamilies)
			continue
		}
		var next []store.IPFamily
		for _, f := range families {
			if slices.Contains(e.Spec.IPFamilies, f) {
				next = append(next, f)
			}
		}
		if len(next) == 0 {
			skipped = append(skipped, e.Cluster)
			continue
		}
		families = next
	}
	return families, skipped
}

func ipFamilySkips(ordered []Export) map[string]bool {
	_, skipped := intersect(ordered)
	out := map[string]bool{}
	for _, c := range skipped {
		out[c] = true
	}
	return out
}

type field struct {
	name   string
	reason string
	value  func(store.ExportSpec) string
	equal  func(a, b store.ExportSpec) bool
}

func sessionAffinity(s store.ExportSpec) store.SessionAffinity {
	if s.SessionAffinity == "" {
		return store.SessionAffinityNone
	}
	return s.SessionAffinity
}

func clientIPTimeout(s store.ExportSpec) (int32, bool) {
	if s.SessionAffinityConfig == nil || s.SessionAffinityConfig.ClientIP == nil {
		return 0, false
	}
	return s.SessionAffinityConfig.ClientIP.TimeoutSeconds, true
}

func internalTrafficPolicy(s store.ExportSpec) store.InternalTrafficPolicy {
	if s.InternalTrafficPolicy == "" {
		return store.InternalTrafficPolicyCluster
	}
	return s.InternalTrafficPolicy
}

func serviceType(s store.ExportSpec) store.ServiceType {
	if s.Type == "" {
		return store.ServiceTypeClusterSetIP
	}
	return s.Type
}

func formatMap(m map[string]string) string {
	keys := slices.Sorted(maps.Keys(m))
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+m[k])
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

// fields are the properties compared by CheckConflictExport, in the order
// that picks the reported reason.
var fields = []field{
	{
		name:   "type",
		reason: api.ConditionReasonTypeConflict,
		value:  func(s store.ExportSpec) string { return string(serviceType(s)) },
		equal:  func(a, b store.ExportSpec) bool { return serviceType(a) == serviceType(b) },
	},
	{
		name:   "sessionAffinity",
		reason: api.ConditionReasonSessionAffinityConflict,
		value:  func(s store.ExportSpec) string { return string(sessionAffinity(s)) },
		equal:  func(a, b store.ExportSpec) bool { return sessionAffinity(a) == sessionAffinity(b) },
	},
	{
		name:   "sessionAffinityConfig.clientIP",
		reason: api.ConditionReasonSessionAffinityConfigConflict,
		value: func(s store.ExportSpec) string {
			if t, ok := clientIPTimeout(s); ok {
				return fmt.Sprint(t)
			}
			return ""
		},
		equal: func(a, b store.ExportSpec) bool {
			ta, oka := clientIPTimeout(a)
			tb, okb := clientIPTimeout(b)
			return oka == okb && ta == tb
		},
	},
	{
		name:   "annotations",
		reason: api.ConditionReasonAnnotationsConflict,
		value:  func(s store.ExportSpec) string { return formatMap(s.Annotations) },
		equal:  func(a, b store.ExportSpec) bool { return maps.Equal(a.Annotations, b.Annotations) },
	},
	{
		name:   "labels",
		reason: api.ConditionReasonLabelsConflict,
		value:  func(s store.ExportSpec) string { return formatMap(s.Labels) },
		equal:  func(a, b store.ExportSpec) bool { return maps.Equal(a.Labels, b.Labels) },
	},
	{
		name:   "internalTrafficPolicy",
		reason: api.ConditionReasonInternalTrafficPolicyConflict,
		value:  func(s store.ExportSpec) string { return string(internalTrafficPolicy(s)) },
		equal:  func(a, b store.ExportSpec) bool { return internalTrafficPolicy(a) == internalTrafficPolicy(b) },
	},
	{
		name:   "trafficDistribution",
		reason: api.ConditionReasonTrafficDistributionConflict,
		value:  func(s store.ExportSpec) string { return s.TrafficDistribution },
		equal:  func(a, b store.ExportSpec) bool { return a.TrafficDistribution == b.TrafficDistribution },
	},
}

// CheckConflictExport compares every export ordered by priority with the
// oldest one on type, session affinity, its client IP timeout, annotations,
// labels, internal traffic policy and traffic distribution, and returns the
// reason and message of the first disagreement. Ports and IP families are
// checked by MergePorts and IntersectIPFamilies.
func CheckConflictExport(ordered []Export) (string, string) {
	if len(ordered) == 0 {
		return api.ConditionReasonNoConflicts, ""
	}
	oldest := ordered[0]
	for i, e := range ordered[1:] {
		for _, f := range fields {
			if f.equal(oldest.Spec, e.Spec) {
				continue
			}
			disagree := 1
			for _, other := range ordered[i+2:] {
				if !f.equal(oldest.Spec, other.Spec) {
					disagree++
				}
			}
			return f.reason, fmt.Sprintf(
				"Conflicting %s. %d/%d clusters disagree. Using %q from oldest service export in cluster %q.",
				f.name, disagree, len(ordered), f.value(oldest.Spec), oldest.Cluster,
			)
		}
	}
	return api.ConditionReasonNoConflicts, ""
}

// clusterConflict returns how the export e disagrees with the winner, or nil.
func clusterConflict(winner, e Export, familySkipped bool) *Conflict {
	if !samePortSet(winner.Spec.Ports, e.Spec.Ports) {
		return &Conflict{
			Reason: api.ConditionReasonPortConflict,
			Message: fmt.Sprintf("Ports of cluster %q differ from the ports of the oldest service export in cluster %q.",
				e.Cluster, winner.Cluster),
			WinningCluster: winner.Cluster,
		}
	}
	if familySkipped {
		return &Conflict{
			Reason:         api.ConditionReasonIPFamilyConflict,
			Message:        fmt.Sprintf("IPFamilies conflict. Cluster %q has no IPFamilies in common.", e.Cluster),
			WinningCluster: winner.Cluster,
		}
	}
	for _, f := range fields {
		if !f.equal(winner.Spec, e.Spec) {
			return &Conflict{
				Reason: f.reason,
				Message: fmt.Sprintf("Conflicting %s. Cluster %q has %q; using %q from oldest service export in cluster %q.",
					f.name, e.Cluster, f.value(e.Spec), f.value(winner.Spec), winner.Cluster),
				WinningCluster: winner.Cluster,
			}
		}
	}
	return nil
}

func conditions(cluster string, participating bool, conflict *Conflict) []Condition {
	valid := Condition{
		Type: api.ConditionValid, Status: ConditionTrue,
		Reason: api.ConditionReasonValid, Message: "ServiceExport is valid",
	}
	if !participating {
		msg := fmt.Sprintf("The lease of cluster %q is not active; its export is excluded from the ServiceImport", cluster)
		return []Condition{
			valid,
			{Type: api.ConditionReady, Status: ConditionFalse, Reason: api.ConditionReasonClusterLeaseExpired, Message: msg},
			{Type: api.ConditionConflict, Status: ConditionUnknown, Reason: api.ConditionReasonClusterLeaseExpired, Message: msg},
		}
	}
	ready := Condition{
		Type: api.ConditionReady, Status: ConditionTrue,
		Reason: api.ConditionReasonReady, Message: "ServiceExport is ready",
	}
	if conflict != nil {
		return []Condition{valid, ready, {
			Type: api.ConditionConflict, Status: ConditionTrue, Reason: conflict.Reason, Message: conflict.Message,
		}}
	}
	return []Condition{valid, ready, {
		Type: api.ConditionConflict, Status: ConditionFalse,
		Reason: api.ConditionReasonNoConflicts, Message: "ServiceExport has no conflicts",
	}}
}
