// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"fmt"
	"net"
	"strings"
	"time"

	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/protobuf/types/known/durationpb"
	"k8s.io/apimachinery/pkg/util/validation"

	mcsapi "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/registry"
)

// Field limits not covered by api/errors.go.
const (
	maxNameBytes      = 253
	maxZones          = 64
	maxAddresses      = 100
	maxIPFamilies     = 2
	maxLabelKeyBytes  = 317
	maxLabelValBytes  = 1024
	maxClientIPTimout = 86400
)

// validator collects field violations.
type validator struct {
	violations []*errdetails.BadRequest_FieldViolation
}

func (v *validator) add(field, format string, args ...any) {
	v.violations = append(v.violations, &errdetails.BadRequest_FieldViolation{Field: field, Description: fmt.Sprintf(format, args...)})
}

func (v *validator) err() error {
	if len(v.violations) == 0 {
		return nil
	}
	return invalidArgument(v.violations)
}

func (v *validator) dnsLabel(field, value string) {
	switch {
	case value == "":
		v.add(field, "is required")
	case !registry.IsDNSLabel(value):
		v.add(field, "%q is not a DNS-1123 label", value)
	}
}

func (v *validator) optionalDNSLabel(field, value string) {
	if value != "" {
		v.dnsLabel(field, value)
	}
}

func (v *validator) maxLen(field, value string, n int) {
	if len(value) > n {
		v.add(field, "is longer than %d bytes", n)
	}
}

// labelValue requires value, when set, to be a valid Kubernetes label value:
// the controllers write it as a label of imported EndpointSlices.
func (v *validator) labelValue(field, value string) {
	if msgs := validation.IsValidLabelValue(value); len(msgs) > 0 {
		v.add(field, "is not a Kubernetes label value: %s", strings.Join(msgs, "; "))
	}
}

func (v *validator) labels(field string, labels map[string]string) {
	if len(labels) > mcsapi.MaxLabels {
		v.add(field, "has %d entries; at most %d are allowed", len(labels), mcsapi.MaxLabels)
	}
	for k, val := range labels {
		if k == "" || len(k) > maxLabelKeyBytes {
			v.add(field, "key %q must have 1 to %d bytes", k, maxLabelKeyBytes)
		}
		if len(val) > maxLabelValBytes {
			v.add(field+"["+k+"]", "is longer than %d bytes", maxLabelValBytes)
		}
	}
}

func (v *validator) annotations(field string, annotations map[string]string) {
	if len(annotations) > mcsapi.MaxAnnotations {
		v.add(field, "has %d entries; at most %d are allowed", len(annotations), mcsapi.MaxAnnotations)
	}
	size := 0
	for k, val := range annotations {
		if k == "" {
			v.add(field, "has an empty key")
		}
		size += len(k) + len(val)
	}
	if size > mcsapi.MaxAnnotationBytes {
		v.add(field, "has %d bytes; at most %d are allowed", size, mcsapi.MaxAnnotationBytes)
	}
}

func (v *validator) selector(field string, s *mcsv1.LabelSelector) {
	v.labels(field+".match_labels", s.GetMatchLabels())
}

func (v *validator) port(field string, port int32, allowZero bool) {
	if (port == 0 && !allowZero) || port < 0 || port > 65535 {
		v.add(field, "%d is not a port number", port)
	}
}

func (v *validator) protocol(field string, p mcsv1.Protocol) {
	if _, ok := mcsv1.Protocol_name[int32(p)]; !ok {
		v.add(field, "unknown protocol %d", p)
	}
}

// leaseTTL validates a requested lease TTL and returns the clamped value.
func (v *validator) leaseTTL(field string, d *durationpb.Duration, def time.Duration) time.Duration {
	if d == nil {
		return def
	}
	if err := d.CheckValid(); err != nil {
		v.add(field, "%v", err)
		return def
	}
	if d.AsDuration() < 0 {
		v.add(field, "must not be negative")
		return def
	}
	return registry.ClampLeaseTTL(d.AsDuration(), def)
}

func (v *validator) exportSpec(field string, s *mcsv1.ServiceExportSpec) {
	if s == nil {
		v.add(field, "is required")
		return
	}
	v.labels(field+".labels", s.GetLabels())
	v.annotations(field+".annotations", s.GetAnnotations())
	switch s.GetType() {
	case mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, mcsv1.ServiceType_SERVICE_TYPE_HEADLESS:
	default:
		v.add(field+".type", "must be SERVICE_TYPE_CLUSTER_SET_IP or SERVICE_TYPE_HEADLESS")
	}
	if len(s.GetPorts()) > mcsapi.MaxPorts {
		v.add(field+".ports", "has %d entries; at most %d are allowed", len(s.GetPorts()), mcsapi.MaxPorts)
	}
	for i, p := range s.GetPorts() {
		f := fmt.Sprintf("%s.ports[%d]", field, i)
		v.port(f+".port", p.GetPort(), false)
		v.protocol(f+".protocol", p.GetProtocol())
		v.maxLen(f+".name", p.GetName(), 63)
		v.maxLen(f+".app_protocol", p.GetAppProtocol(), maxNameBytes)
	}
	if _, ok := mcsv1.SessionAffinity_name[int32(s.GetSessionAffinity())]; !ok {
		v.add(field+".session_affinity", "unknown value %d", s.GetSessionAffinity())
	}
	if t := s.GetSessionAffinityConfig().GetClientIp().GetTimeoutSeconds(); t < 0 || t > maxClientIPTimout {
		v.add(field+".session_affinity_config.client_ip.timeout_seconds", "must be between 0 and %d", maxClientIPTimout)
	}
	if len(s.GetIpFamilies()) > maxIPFamilies {
		v.add(field+".ip_families", "has more than %d entries", maxIPFamilies)
	}
	for i, f := range s.GetIpFamilies() {
		if f != mcsv1.IPFamily_IP_FAMILY_IPV4 && f != mcsv1.IPFamily_IP_FAMILY_IPV6 {
			v.add(fmt.Sprintf("%s.ip_families[%d]", field, i), "must be IP_FAMILY_IPV4 or IP_FAMILY_IPV6")
		}
	}
	if _, ok := mcsv1.InternalTrafficPolicy_name[int32(s.GetInternalTrafficPolicy())]; !ok {
		v.add(field+".internal_traffic_policy", "unknown value %d", s.GetInternalTrafficPolicy())
	}
	v.maxLen(field+".traffic_distribution", s.GetTrafficDistribution(), 63)
	if ts := s.GetExportCreationTimestamp(); ts != nil {
		if err := ts.CheckValid(); err != nil {
			v.add(field+".export_creation_timestamp", "%v", err)
		}
	}
}

func (v *validator) endpoint(field string, e *mcsv1.Endpoint) {
	if n := len(e.GetAddresses()); n == 0 || n > maxAddresses {
		v.add(field+".addresses", "must have 1 to %d entries", maxAddresses)
	}
	for i, a := range e.GetAddresses() {
		if a == "" || len(a) > maxNameBytes {
			v.add(fmt.Sprintf("%s.addresses[%d]", field, i), "must have 1 to %d bytes", maxNameBytes)
		}
	}
	if _, ok := mcsv1.AddressType_name[int32(e.GetAddressType())]; !ok {
		v.add(field+".address_type", "unknown value %d", e.GetAddressType())
	}
	if len(e.GetAttributes()) > mcsapi.MaxEndpointAttributes {
		v.add(field+".attributes", "has %d entries; at most %d are allowed", len(e.GetAttributes()), mcsapi.MaxEndpointAttributes)
	}
	if len(e.GetPorts()) > mcsapi.MaxPorts {
		v.add(field+".ports", "has more than %d entries", mcsapi.MaxPorts)
	}
	for i, p := range e.GetPorts() {
		f := fmt.Sprintf("%s.ports[%d]", field, i)
		v.port(f+".port", p.GetPort(), true)
		v.protocol(f+".protocol", p.GetProtocol())
		v.maxLen(f+".name", p.GetName(), 63)
	}
	v.maxLen(field+".hostname", e.GetHostname(), maxNameBytes)
	v.maxLen(field+".node_name", e.GetNodeName(), maxNameBytes)
	v.maxLen(field+".zone", e.GetZone(), maxNameBytes)
}

// addressType infers the address type of an endpoint that names none.
func addressType(addresses []string) mcsv1.AddressType {
	if len(addresses) == 0 {
		return mcsv1.AddressType_ADDRESS_TYPE_UNSPECIFIED
	}
	ip := net.ParseIP(addresses[0])
	switch {
	case ip == nil:
		return mcsv1.AddressType_ADDRESS_TYPE_FQDN
	case ip.To4() != nil:
		return mcsv1.AddressType_ADDRESS_TYPE_IPV4
	}
	return mcsv1.AddressType_ADDRESS_TYPE_IPV6
}
