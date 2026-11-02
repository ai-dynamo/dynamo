// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package importer

import (
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	discoveryv1ac "k8s.io/client-go/applyconfigurations/discovery/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/convert"
)

// importAPIVersion and importKind identify ServiceImports in owner references
// and apply requests.
var (
	importAPIVersion = mcsv1beta1.GroupVersion.String()
	importKind       = mcsv1beta1.ServiceImportKindName
)

// managedLabels are the labels of every object the importer writes.
func managedLabels(clusterset string) map[string]string {
	return map[string]string{api.LabelManagedBy: api.ManagedByController, api.LabelClusterSet: clusterset}
}

// isManaged reports whether the importer created an object.
func isManaged(labels map[string]string) bool {
	return labels[api.LabelManagedBy] == api.ManagedByController
}

// validLabels drops entries that are not valid Kubernetes labels, which an
// explicit spec.exportedLabels may contain.
func validLabels(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		if len(validation.IsQualifiedName(k)) == 0 && len(validation.IsValidLabelValue(v)) == 0 {
			out[k] = v
		}
	}
	return out
}

// validAnnotations drops entries whose keys are not qualified names.
func validAnnotations(in map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range in {
		if len(validation.IsQualifiedName(strings.ToLower(k))) == 0 {
			out[k] = v
		}
	}
	return out
}

// InvalidPorts reports why the merged ports of an import cannot be the ports
// of a Service: more than one port with an unnamed one among them, or a
// name used twice. The registry's merge does not produce such a set, but an
// older registry may; applying it would fail on every reconcile.
func InvalidPorts(imp *mcsv1.ServiceImport) error {
	ports := imp.GetPorts()
	names := make(map[string]bool, len(ports))
	for _, p := range ports {
		if p.GetName() == "" && len(ports) > 1 {
			return fmt.Errorf("port %d/%s has no name although the import has %d ports", p.GetPort(), convert.ProtocolFromRegistry(p.GetProtocol()), len(ports))
		}
		if names[p.GetName()] {
			return fmt.Errorf("port name %q is used twice", p.GetName())
		}
		names[p.GetName()] = true
	}
	return nil
}

// importPorts maps the merged ports of an import.
func importPorts(imp *mcsv1.ServiceImport) []mcsv1beta1.ServicePort {
	ports := make([]mcsv1beta1.ServicePort, 0, len(imp.GetPorts()))
	for _, p := range imp.GetPorts() {
		sp := mcsv1beta1.ServicePort{Name: p.GetName(), Protocol: convert.ProtocolFromRegistry(p.GetProtocol()), Port: p.GetPort()}
		if p.GetAppProtocol() != "" {
			sp.AppProtocol = new(p.GetAppProtocol())
		}
		ports = append(ports, sp)
	}
	return ports
}

// BuildServiceImport returns the apply request of the ServiceImport of imp:
// the merged spec, the exported labels and annotations, the managed-by and
// clusterset labels, the derived-service annotation and spec.ips.
func BuildServiceImport(imp *mcsv1.ServiceImport, clusterset string, ips []string) (*unstructured.Unstructured, error) {
	spec := mcsv1beta1.ServiceImportSpec{
		Ports:                 importPorts(imp),
		IPs:                   ips,
		Type:                  convert.ImportTypeFromRegistry(imp.GetType()),
		SessionAffinity:       convert.SessionAffinityFromRegistry(imp.GetSessionAffinity()),
		SessionAffinityConfig: convert.SessionAffinityConfigFromRegistry(imp.GetSessionAffinityConfig()),
		IPFamilies:            convert.IPFamiliesFromRegistry(imp.GetIpFamilies()),
	}
	specMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&spec)
	if err != nil {
		return nil, fmt.Errorf("convert ServiceImport spec: %w", err)
	}
	labels := validLabels(imp.GetLabels())
	maps.Copy(labels, managedLabels(clusterset))
	annotations := validAnnotations(imp.GetAnnotations())
	annotations[api.AnnotationMCSDerivedService] = api.DerivedServiceName(imp.GetNamespace(), imp.GetName())
	u := &unstructured.Unstructured{Object: map[string]any{"spec": specMap}}
	u.SetAPIVersion(importAPIVersion)
	u.SetKind(importKind)
	u.SetNamespace(imp.GetNamespace())
	u.SetName(imp.GetName())
	u.SetLabels(labels)
	u.SetAnnotations(annotations)
	return u, nil
}

// BuildServiceImportStatus returns the status of a ServiceImport: the
// participating clusters, Ready, and Conflict mirrored from the import.
// existing are the current conditions, so that transition times only move
// on a status change.
func BuildServiceImportStatus(imp *mcsv1.ServiceImport, existing []metav1.Condition, generation int64) mcsv1beta1.ServiceImportStatus {
	st := mcsv1beta1.ServiceImportStatus{Conditions: slices.Clone(existing), Clusters: []mcsv1beta1.ClusterStatus{}}
	participating := 0
	for _, c := range imp.GetClusters() {
		if c.GetParticipating() {
			st.Clusters = append(st.Clusters, mcsv1beta1.ClusterStatus{Cluster: c.GetCluster()})
			participating++
		}
	}
	ready := metav1.Condition{
		Type: string(mcsv1beta1.ServiceImportConditionReady), Status: metav1.ConditionTrue,
		Reason:             string(mcsv1beta1.ServiceImportReasonReady),
		Message:            fmt.Sprintf("Imported from the registry: %d of %d exporting clusters participate", participating, len(imp.GetClusters())),
		ObservedGeneration: generation,
	}
	if participating == 0 {
		ready.Status, ready.Reason = metav1.ConditionFalse, string(mcsv1beta1.ServiceImportReasonPending)
		ready.Message = fmt.Sprintf("None of the %d exporting clusters participates: their leases expired", len(imp.GetClusters()))
	}
	setCondition(&st.Conditions, ready)
	conflict := metav1.Condition{
		Type: api.ConditionConflict, Status: metav1.ConditionFalse, Reason: api.ConditionReasonNoConflicts,
		Message: "The exports agree", ObservedGeneration: generation,
	}
	if c := imp.GetConflict(); c.GetReason() != "" {
		conflict.Status, conflict.Reason, conflict.Message = metav1.ConditionTrue, c.GetReason(), c.GetMessage()
	}
	setCondition(&st.Conditions, conflict)
	return st
}

// setCondition is meta.SetStatusCondition without its dependency on the
// current time when the status does not change.
func setCondition(conds *[]metav1.Condition, c metav1.Condition) {
	for i := range *conds {
		old := &(*conds)[i]
		if old.Type != c.Type {
			continue
		}
		if old.Status != c.Status {
			old.Status = c.Status
			old.LastTransitionTime = metav1.Now()
		}
		old.Reason, old.Message, old.ObservedGeneration = c.Reason, c.Message, c.ObservedGeneration
		return
	}
	if c.LastTransitionTime.IsZero() {
		c.LastTransitionTime = metav1.Now()
	}
	*conds = append(*conds, c)
}

// ownerRef returns the owner reference to a ServiceImport.
func ownerRef(name string, uid types.UID) *metav1ac.OwnerReferenceApplyConfiguration {
	return metav1ac.OwnerReference().WithAPIVersion(importAPIVersion).WithKind(importKind).
		WithName(name).WithUID(uid).WithController(true)
}

// DerivedIPFamilies returns the ipFamilyPolicy and ipFamilies of the derived
// Service of imp in a cluster whose Service network has clusterFamilies.
// The import's families describe the exporters, not the importing cluster:
// the policy follows their count (SingleStack for one, PreferDualStack for
// more), and only the families the importing cluster has are listed, in
// the import's order, so that the API server never rejects the Service
// and picks the primary family itself when the import's is unavailable.
// Unknown clusterFamilies list no family at all.
func DerivedIPFamilies(imp *mcsv1.ServiceImport, clusterFamilies []corev1.IPFamily) (*corev1.IPFamilyPolicy, []corev1.IPFamily) {
	fams := convert.IPFamiliesFromRegistry(imp.GetIpFamilies())
	if len(fams) == 0 {
		return nil, nil
	}
	policy := corev1.IPFamilyPolicySingleStack
	if len(fams) > 1 {
		policy = corev1.IPFamilyPolicyPreferDualStack
	}
	var listed []corev1.IPFamily
	for _, f := range fams {
		if slices.Contains(clusterFamilies, f) {
			listed = append(listed, f)
		}
	}
	return &policy, listed
}

// BuildDerivedService returns the apply request of the derived Service of
// imp: no selector, ClusterIP for ClusterSetIP imports and clusterIP None for
// headless ones, the merged ports, session affinity and the IP family policy
// and families of DerivedIPFamilies, owned by the ServiceImport.
func BuildDerivedService(imp *mcsv1.ServiceImport, clusterset string, owner types.UID, clusterFamilies []corev1.IPFamily) *corev1ac.ServiceApplyConfiguration {
	labels := managedLabels(clusterset)
	labels[api.LabelMCSServiceName] = imp.GetName()
	spec := corev1ac.ServiceSpec().
		WithType(corev1.ServiceTypeClusterIP).
		WithSessionAffinity(convert.SessionAffinityFromRegistry(imp.GetSessionAffinity()))
	if imp.GetType() == mcsv1.ServiceType_SERVICE_TYPE_HEADLESS {
		spec.WithClusterIP(corev1.ClusterIPNone)
	}
	if cfg := convert.SessionAffinityConfigFromRegistry(imp.GetSessionAffinityConfig()); cfg != nil &&
		imp.GetSessionAffinity() == mcsv1.SessionAffinity_SESSION_AFFINITY_CLIENT_IP {
		spec.WithSessionAffinityConfig(corev1ac.SessionAffinityConfig().
			WithClientIP(corev1ac.ClientIPConfig().WithTimeoutSeconds(*cfg.ClientIP.TimeoutSeconds)))
	}
	if policy, fams := DerivedIPFamilies(imp, clusterFamilies); policy != nil {
		spec.WithIPFamilyPolicy(*policy)
		if len(fams) > 0 {
			spec.WithIPFamilies(fams...)
		}
	}
	for _, p := range importPorts(imp) {
		port := corev1ac.ServicePort().WithName(p.Name).WithProtocol(p.Protocol).WithPort(p.Port)
		if p.AppProtocol != nil {
			port.WithAppProtocol(*p.AppProtocol)
		}
		spec.WithPorts(port)
	}
	return corev1ac.Service(api.DerivedServiceName(imp.GetNamespace(), imp.GetName()), imp.GetNamespace()).
		WithLabels(labels).
		WithOwnerReferences(ownerRef(imp.GetName(), owner)).
		WithSpec(spec)
}

// SliceName returns the name of the i-th EndpointSlice of address type at
// of a source cluster: <derived>-<cluster>-<v4|v6|fqdn>-<i>. The address
// type is part of the name and the index counts per address type, since
// addressType is immutable: a slice name never has to change its type when
// the endpoints of another type grow or shrink.
func SliceName(derived, cluster string, at discoveryv1.AddressType, i int) string {
	return derived + "-" + cluster + "-" + addressTypeToken(at) + "-" + strconv.Itoa(i)
}

func addressTypeToken(at discoveryv1.AddressType) string {
	switch at {
	case discoveryv1.AddressTypeIPv4:
		return "v4"
	case discoveryv1.AddressTypeIPv6:
		return "v6"
	}
	return "fqdn"
}

// group is the endpoints of one source cluster that share an address type
// and a port set, which an EndpointSlice requires.
type group struct {
	addressType discoveryv1.AddressType
	portsKey    string
	ports       []discoveryv1.EndpointPort
	endpoints   []discoveryv1.Endpoint
}

// BuildEndpointSlices packs the endpoints of every source cluster of imp,
// the local cluster included, into EndpointSlices of at most perSlice
// endpoints named <derived>-<cluster>-<v4|v6|fqdn>-<i> (SliceName),
// labelled with the source cluster and, when the registry knows it and it
// is a valid label value, the source cluster's region
// (topology.kubernetes.io/region). Endpoint ports are filtered to the port
// names of the merged import; node names are kept for the local cluster
// only, since remote ones name nodes of another cluster; target references
// are set from the exported attributes unless omitTargetRefs. Packing is
// deterministic: groups by address type and port set, endpoints sorted by
// address.
func BuildEndpointSlices(imp *mcsv1.ServiceImport, clusterset, localCluster string, owner types.UID, perSlice int, omitTargetRefs bool) []*discoveryv1ac.EndpointSliceApplyConfiguration {
	if perSlice < 1 {
		perSlice = 100
	}
	derived := api.DerivedServiceName(imp.GetNamespace(), imp.GetName())
	names := map[string]bool{}
	for _, p := range imp.GetPorts() {
		names[p.GetName()] = true
	}
	var out []*discoveryv1ac.EndpointSliceApplyConfiguration
	for _, ce := range imp.GetEndpoints() {
		cluster := ce.GetCluster()
		groups := map[string]*group{}
		for _, ep := range ce.GetEndpoints() {
			if len(ep.GetAddresses()) == 0 {
				continue
			}
			at := convert.AddressTypeFromRegistry(ep.GetAddressType(), ep.GetAddresses())
			ports := slicePorts(ep.GetPorts(), names)
			pk := portsKey(ports)
			gk := string(at) + "|" + pk
			g := groups[gk]
			if g == nil {
				g = &group{addressType: at, portsKey: pk, ports: ports}
				groups[gk] = g
			}
			g.endpoints = append(g.endpoints, sliceEndpoint(ep, cluster == localCluster, omitTargetRefs))
		}
		sorted := slices.SortedFunc(maps.Values(groups), func(a, b *group) int {
			return cmp.Or(cmp.Compare(addressTypeOrder(a.addressType), addressTypeOrder(b.addressType)),
				strings.Compare(a.portsKey, b.portsKey))
		})
		index := map[discoveryv1.AddressType]int{}
		for _, g := range sorted {
			slices.SortFunc(g.endpoints, func(a, b discoveryv1.Endpoint) int {
				return cmp.Or(slices.Compare(a.Addresses, b.Addresses), strings.Compare(deref(a.Hostname), deref(b.Hostname)))
			})
			for start := 0; start < len(g.endpoints); start += perSlice {
				chunk := g.endpoints[start:min(start+perSlice, len(g.endpoints))]
				out = append(out, sliceApply(imp, clusterset, derived, cluster, ce.GetRegion(), index[g.addressType], owner, g, chunk))
				index[g.addressType]++
			}
		}
	}
	return out
}

func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func addressTypeOrder(t discoveryv1.AddressType) int {
	switch t {
	case discoveryv1.AddressTypeIPv4:
		return 0
	case discoveryv1.AddressTypeIPv6:
		return 1
	}
	return 2
}

// slicePorts maps endpoint ports, keeping those whose name is a port of
// the merged import, sorted.
func slicePorts(ports []*mcsv1.EndpointPort, names map[string]bool) []discoveryv1.EndpointPort {
	var out []discoveryv1.EndpointPort
	for _, p := range ports {
		if !names[p.GetName()] {
			continue
		}
		ep := discoveryv1.EndpointPort{
			Name:     new(p.GetName()),
			Protocol: new(convert.ProtocolFromRegistry(p.GetProtocol())),
			Port:     new(p.GetPort()),
		}
		if p.GetAppProtocol() != "" {
			ep.AppProtocol = new(p.GetAppProtocol())
		}
		out = append(out, ep)
	}
	slices.SortFunc(out, func(a, b discoveryv1.EndpointPort) int {
		return cmp.Or(strings.Compare(*a.Name, *b.Name), cmp.Compare(*a.Port, *b.Port), strings.Compare(string(*a.Protocol), string(*b.Protocol)))
	})
	return out
}

func portsKey(ports []discoveryv1.EndpointPort) string {
	var b strings.Builder
	for _, p := range ports {
		fmt.Fprintf(&b, "%s/%s/%d/%s;", *p.Name, *p.Protocol, *p.Port, deref(p.AppProtocol))
	}
	return b.String()
}

func sliceEndpoint(ep *mcsv1.Endpoint, local, omitTargetRef bool) discoveryv1.Endpoint {
	c := ep.GetConditions()
	out := discoveryv1.Endpoint{
		Addresses: slices.Clone(ep.GetAddresses()),
		Conditions: discoveryv1.EndpointConditions{
			Ready: new(c.GetReady()), Serving: new(c.GetServing()), Terminating: new(c.GetTerminating()),
		},
	}
	if ep.GetHostname() != "" {
		out.Hostname = new(ep.GetHostname())
	}
	if ep.GetZone() != "" {
		out.Zone = new(ep.GetZone())
	}
	if local && ep.GetNodeName() != "" {
		out.NodeName = new(ep.GetNodeName())
	}
	if !omitTargetRef {
		out.TargetRef = targetRef(ep.GetAttributes())
	}
	return out
}

// targetRef returns the target reference of an imported endpoint from the
// attributes of the exported one: the targetRef.name attribute with the
// targetRef.kind attribute (Pod when absent) and the targetRef.namespace
// attribute (empty when absent), or nil without a name.
func targetRef(attrs map[string]string) *corev1.ObjectReference {
	name := attrs[api.EndpointAttrTargetName]
	if name == "" {
		return nil
	}
	kind := attrs[api.EndpointAttrTargetKind]
	if kind == "" {
		kind = "Pod"
	}
	return &corev1.ObjectReference{Kind: kind, Name: name, Namespace: attrs[api.EndpointAttrTargetNamespace]}
}

func sliceApply(imp *mcsv1.ServiceImport, clusterset, derived, cluster, region string, i int, owner types.UID, g *group, eps []discoveryv1.Endpoint) *discoveryv1ac.EndpointSliceApplyConfiguration {
	labels := managedLabels(clusterset)
	labels[discoveryv1.LabelServiceName] = derived
	labels[api.LabelMCSServiceName] = imp.GetName()
	labels[api.LabelMCSSourceCluster] = cluster
	labels[api.LabelEndpointSliceManagedBy] = api.EndpointSliceManagedBy
	if region != "" && len(validation.IsValidLabelValue(region)) == 0 {
		// A region that is not a label value (an older controller did not
		// validate --region) is left off rather than failing the apply.
		labels[api.LabelTopologyRegion] = region
	}
	s := discoveryv1ac.EndpointSlice(SliceName(derived, cluster, g.addressType, i), imp.GetNamespace()).
		WithLabels(labels).
		WithOwnerReferences(ownerRef(imp.GetName(), owner)).
		WithAddressType(g.addressType)
	for _, e := range eps {
		ac := discoveryv1ac.Endpoint().WithAddresses(e.Addresses...).WithConditions(discoveryv1ac.EndpointConditions().
			WithReady(*e.Conditions.Ready).WithServing(*e.Conditions.Serving).WithTerminating(*e.Conditions.Terminating))
		if e.Hostname != nil {
			ac.WithHostname(*e.Hostname)
		}
		if e.Zone != nil {
			ac.WithZone(*e.Zone)
		}
		if e.NodeName != nil {
			ac.WithNodeName(*e.NodeName)
		}
		if ref := e.TargetRef; ref != nil {
			r := corev1ac.ObjectReference().WithKind(ref.Kind).WithName(ref.Name)
			if ref.Namespace != "" {
				r.WithNamespace(ref.Namespace)
			}
			ac.WithTargetRef(r)
		}
		s.WithEndpoints(ac)
	}
	for _, p := range g.ports {
		ac := discoveryv1ac.EndpointPort().WithName(*p.Name).WithProtocol(*p.Protocol).WithPort(*p.Port)
		if p.AppProtocol != nil {
			ac.WithAppProtocol(*p.AppProtocol)
		}
		s.WithPorts(ac)
	}
	return s
}
