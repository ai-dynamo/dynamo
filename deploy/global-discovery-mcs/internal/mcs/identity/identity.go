// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package identity resolves the clusterset and cluster ids, the region and
// the zones of the cluster the controller runs in.
package identity

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
)

// Node labels read for the region and the zones.
const (
	LabelRegion = corev1.LabelTopologyRegion
	LabelZone   = corev1.LabelTopologyZone
)

// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=about.k8s.io,resources=clusterproperties,verbs=get;list;watch;create;update;patch

// Identity is what the controller registers with the registry.
type Identity struct {
	ClusterSet string
	Cluster    string
	Region     string
	// Zones are the distinct topology.kubernetes.io/zone values of the
	// nodes, sorted.
	Zones []string
	// Provider is the scheme of the nodes' spec.providerID, for example aws.
	Provider string
	Labels   map[string]string
}

// Options say where the identity comes from.
type Options struct {
	// Source is one of the config.IdentitySource* values; empty means auto.
	Source       string
	ClusterSetID string
	ClusterID    string
	Region       string
	Labels       map[string]string
}

// FromConfig returns the Options of a controller configuration.
func FromConfig(c *config.Config) Options {
	return Options{
		Source:       c.IdentitySource,
		ClusterSetID: c.ClusterSetID,
		ClusterID:    c.ClusterID,
		Region:       c.Region,
		Labels:       maps.Clone(c.ClusterLabels),
	}
}

// Resolve determines the identity. The id flags win in auto mode; otherwise
// the values of the ClusterProperty objects clusterset.k8s.io and
// cluster.clusterset.k8s.io are used. The region comes from Options.Region
// or the most common topology.kubernetes.io/region node label, the zones
// from the node zone labels. r must read from the API server (the manager's
// API reader), since caches are not running yet.
func Resolve(ctx context.Context, r client.Reader, opts Options) (Identity, error) {
	if opts.Source == "" {
		opts.Source = config.IdentitySourceAuto
	}
	id := Identity{ClusterSet: opts.ClusterSetID, Cluster: opts.ClusterID, Region: opts.Region, Labels: opts.Labels}
	if opts.Source != config.IdentitySourceFlags {
		for _, p := range []struct {
			name string
			flag string
			dst  *string
		}{
			{api.ClusterPropertyClusterSet, "--clusterset-id", &id.ClusterSet},
			{api.ClusterPropertyClusterID, "--cluster-id", &id.Cluster},
		} {
			if opts.Source == config.IdentitySourceAuto && *p.dst != "" {
				// The flag wins; the property is not needed.
				continue
			}
			value, err := GetClusterProperty(ctx, r, p.name)
			if err != nil {
				return Identity{}, err
			}
			switch {
			case opts.Source == config.IdentitySourceClusterProperty && value == "":
				return Identity{}, fmt.Errorf("ClusterProperty %s is missing or empty", p.name)
			case opts.Source == config.IdentitySourceClusterProperty && *p.dst != "" && *p.dst != value:
				return Identity{}, fmt.Errorf("%s %q disagrees with ClusterProperty %s %q", p.flag, *p.dst, p.name, value)
			case *p.dst == "":
				*p.dst = value
			}
		}
	}
	for what, v := range map[string]string{"clusterset id": id.ClusterSet, "cluster id": id.Cluster} {
		if v == "" {
			return Identity{}, fmt.Errorf("no %s: set the flag or the ClusterProperty", what)
		}
		if msgs := validation.IsDNS1123Label(v); len(msgs) > 0 {
			return Identity{}, fmt.Errorf("%s %q is not a DNS label: %s", what, v, strings.Join(msgs, "; "))
		}
	}
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes); err != nil {
		return Identity{}, fmt.Errorf("list nodes: %w", err)
	}
	regions := map[string]int{}
	zones := map[string]bool{}
	for _, n := range nodes.Items {
		if v := n.Labels[LabelRegion]; v != "" {
			regions[v]++
		}
		if v := n.Labels[LabelZone]; v != "" {
			zones[v] = true
		}
		if id.Provider == "" {
			if scheme, _, ok := strings.Cut(n.Spec.ProviderID, "://"); ok {
				id.Provider = scheme
			}
		}
	}
	if id.Region == "" {
		id.Region = mostCommon(regions)
	}
	id.Zones = slices.Sorted(maps.Keys(zones))
	return id, nil
}

// mostCommon returns the key with the highest count, the smallest key on a
// tie, or "".
func mostCommon(counts map[string]int) string {
	best := ""
	for _, k := range slices.Sorted(maps.Keys(counts)) {
		if best == "" || counts[k] > counts[best] {
			best = k
		}
	}
	return best
}

// NewClusterProperty returns an empty unstructured ClusterProperty.
func NewClusterProperty() *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(config.ClusterPropertyGVK)
	return u
}

// GetClusterProperty returns spec.value of the named ClusterProperty, or ""
// when it or its CRD does not exist.
func GetClusterProperty(ctx context.Context, r client.Reader, name string) (string, error) {
	u := NewClusterProperty()
	err := r.Get(ctx, client.ObjectKey{Name: name}, u)
	switch {
	case apierrors.IsNotFound(err), meta.IsNoMatchError(err):
		return "", nil
	case err != nil:
		return "", fmt.Errorf("get ClusterProperty %s: %w", name, err)
	}
	v, _, err := unstructured.NestedString(u.Object, "spec", "value")
	if err != nil {
		return "", fmt.Errorf("ClusterProperty %s: %w", name, err)
	}
	return strings.TrimSpace(v), nil
}

// WriteClusterProperties publishes the clusterset and cluster ids as
// ClusterProperty objects for other MCS tooling. An existing object with
// the same value is left alone; one with another value is an error, since
// a cluster must not change its identity silently.
func WriteClusterProperties(ctx context.Context, c client.Client, id Identity) error {
	for _, p := range []struct{ name, value string }{
		{api.ClusterPropertyClusterSet, id.ClusterSet},
		{api.ClusterPropertyClusterID, id.Cluster},
	} {
		current, err := GetClusterProperty(ctx, c, p.name)
		if err != nil {
			return err
		}
		switch current {
		case p.value:
			continue
		case "":
		default:
			return fmt.Errorf("ClusterProperty %s has value %q, not %q; refusing to overwrite it", p.name, current, p.value)
		}
		u := NewClusterProperty()
		u.SetName(p.name)
		u.SetLabels(map[string]string{api.LabelManagedBy: api.ManagedByController})
		if err := unstructured.SetNestedField(u.Object, p.value, "spec", "value"); err != nil {
			return err
		}
		if err := c.Create(ctx, u); err != nil {
			if !apierrors.IsAlreadyExists(err) {
				return fmt.Errorf("create ClusterProperty %s: %w", p.name, err)
			}
			// Created concurrently, for example by another replica: check
			// that it carries our value.
			if v, err := GetClusterProperty(ctx, c, p.name); err != nil || v != p.value {
				return cmp.Or(err, fmt.Errorf("ClusterProperty %s has value %q, not %q", p.name, v, p.value))
			}
		}
	}
	return nil
}
