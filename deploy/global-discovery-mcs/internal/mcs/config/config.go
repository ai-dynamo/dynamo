// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package config holds the mcs-controller command line and turns it into
// controller-runtime manager options.
package config

import (
	"errors"
	"flag"
	"fmt"
	"maps"
	"net/url"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
)

// Defaults of the flags.
const (
	DefaultLeaseTTL                = 30 * time.Second
	MinLeaseTTL                    = 10 * time.Second
	MaxLeaseTTL                    = 5 * time.Minute
	DefaultEndpointsPerSlice       = 100
	MaxEndpointsPerSlice           = 1000
	DefaultMaxConcurrentReconciles = 4
	DefaultHealthProbeBindAddress  = ":8081"
	// DefaultResyncPeriod is the default of --export-resync-period and
	// --import-resync-period.
	DefaultResyncPeriod = 10 * time.Minute
	// MinResyncPeriod is the shortest non-zero resync period accepted.
	MinResyncPeriod = time.Second
)

// Environment variables read by BindFlags.
const (
	// RegistryURLsEnv is a comma-separated list of registry URLs used when
	// --registry-url is not given.
	RegistryURLsEnv = "MCS_REGISTRY_URLS"
	// PodNamespaceEnv, set from the downward API, defaults
	// --operator-namespace.
	PodNamespaceEnv = "POD_NAMESPACE"
)

// Values of --cluster-identity-source.
const (
	// IdentitySourceAuto takes each id from its flag when set, else from
	// the ClusterProperty object.
	IdentitySourceAuto = "auto"
	// IdentitySourceFlags requires both id flags and never reads
	// ClusterProperty objects.
	IdentitySourceFlags = "flags"
	// IdentitySourceClusterProperty requires the ClusterProperty objects;
	// id flags, when set, must agree with them.
	IdentitySourceClusterProperty = "clusterproperty"
)

// ClusterPropertyGVK is the KEP-2149 ClusterProperty kind, used as
// unstructured: the controller registers no Go type for it.
var ClusterPropertyGVK = schema.GroupVersionKind{Group: "about.k8s.io", Version: "v1alpha1", Kind: "ClusterProperty"}

// Config is the parsed command line.
type Config struct {
	// RegistryURLs is the ordered list of registry endpoints.
	RegistryURLs []string
	// RegistryTLSCAFile is a PEM bundle of the CAs that sign the registry
	// certificates. Empty uses the system roots.
	RegistryTLSCAFile string
	// RegistryAllowInsecureHTTP permits http:// URLs.
	RegistryAllowInsecureHTTP bool

	// ClusterSetID and ClusterID are the KEP-1645 identifiers; see
	// IdentitySource.
	ClusterSetID string
	ClusterID    string
	// IdentitySource is one of the IdentitySource* values.
	IdentitySource string
	// Region of the cluster. Empty derives it from the
	// topology.kubernetes.io/region node label.
	Region string
	// ClusterLabels are the labels of the cluster registration.
	ClusterLabels map[string]string
	// LeaseTTL is the cluster lease requested from the registry; heartbeats run
	// every LeaseTTL/3.
	LeaseTTL time.Duration

	// WatchNamespaces restricts the namespaced caches, the exports and the
	// imports. Empty means all namespaces.
	WatchNamespaces []string
	// CreateNamespaces creates the namespace of an import that has none,
	// instead of holding the import pending.
	CreateNamespaces bool
	// AdoptUnmanagedImports takes over ServiceImports the controller did not
	// create.
	AdoptUnmanagedImports bool
	// ImportTargetRef sets targetRef on imported endpoints from the exported
	// target attributes. Off for installations whose service mesh resolves
	// endpoint target references locally.
	ImportTargetRef bool
	// ExportResyncPeriod is the interval of the periodic export resync; 0
	// disables it.
	ExportResyncPeriod time.Duration
	// ImportResyncPeriod is the interval at which every import is applied
	// again; 0 disables it.
	ImportResyncPeriod time.Duration
	// WriteClusterProperties publishes the resolved identity as
	// ClusterProperty objects.
	WriteClusterProperties bool
	// DeregisterOnShutdown deregisters the cluster, which deletes its exports
	// in the registry, when the controller stops.
	DeregisterOnShutdown bool
	// EndpointsPerSlice is the maximum number of endpoints per imported
	// EndpointSlice.
	EndpointsPerSlice int
	// MaxConcurrentReconciles bounds each controller's workers.
	MaxConcurrentReconciles int

	// MetricsBindAddress serves /metrics; "0" disables it.
	MetricsBindAddress string
	// HealthProbeBindAddress serves /healthz and /readyz.
	HealthProbeBindAddress string
	// LeaderElect enables leader election; only the leader talks to the
	// registry as a writer and applies imports.
	LeaderElect bool
	// OperatorNamespace holds the leader election Lease.
	OperatorNamespace string
	// OTelEnabled turns on OpenTelemetry tracing, configured through the
	// standard OTEL_* environment variables.
	OTelEnabled bool
}

// BindFlags registers every flag on fs with its default. MCS_REGISTRY_URLS and
// POD_NAMESPACE provide the defaults of --registry-url and
// --operator-namespace.
func (c *Config) BindFlags(fs *flag.FlagSet) {
	c.bindFlags(fs, os.LookupEnv)
}

func (c *Config) bindFlags(fs *flag.FlagSet, lookupEnv func(string) (string, bool)) {
	urls := &urlList{target: &c.RegistryURLs}
	if v, ok := lookupEnv(RegistryURLsEnv); ok {
		urls.setDefault(v)
	}
	fs.Var(urls, "registry-url", "Registry URL, repeatable or comma-separated, in failover order. Defaults to $"+RegistryURLsEnv+". Required.")
	fs.StringVar(&c.RegistryTLSCAFile, "registry-tls-ca-file", "", "PEM bundle of the CAs that sign the registry certificates. Empty uses the system roots.")
	fs.BoolVar(&c.RegistryAllowInsecureHTTP, "registry-allow-insecure-http", false, "Permit plaintext http:// registry URLs.")
	fs.StringVar(&c.ClusterSetID, "clusterset-id", "", "Clusterset name, a DNS label. Falls back to the ClusterProperty "+api.ClusterPropertyClusterSet+".")
	fs.StringVar(&c.ClusterID, "cluster-id", "", "Cluster id, a DNS label unique in the clusterset. Falls back to the ClusterProperty "+api.ClusterPropertyClusterID+".")
	c.IdentitySource = IdentitySourceAuto
	fs.StringVar(&c.IdentitySource, "cluster-identity-source", IdentitySourceAuto, "Where the clusterset and cluster ids come from: auto (flags win, else ClusterProperty), flags or clusterproperty.")
	fs.StringVar(&c.Region, "region", "", "Region of this cluster. Empty uses the topology.kubernetes.io/region node label.")
	fs.Var((*labelMap)(&c.ClusterLabels), "cluster-labels", "Comma-separated key=value labels of the cluster registration.")
	fs.DurationVar(&c.LeaseTTL, "lease-ttl", DefaultLeaseTTL, "Cluster lease TTL requested from the registry, between 10s and 5m. Heartbeats run every TTL/3.")
	fs.Var((*namespaceList)(&c.WatchNamespaces), "watch-namespaces", "Comma-separated namespaces to export from and import into. Empty means all namespaces.")
	fs.BoolVar(&c.CreateNamespaces, "create-namespaces", false, "Create a missing namespace for an import instead of holding the import pending.")
	fs.BoolVar(&c.AdoptUnmanagedImports, "adopt-unmanaged-imports", false, "Take over ServiceImports this controller did not create.")
	fs.BoolVar(&c.ImportTargetRef, "import-target-ref", true, "Set targetRef on imported endpoints from the exported target attributes. The Global Discovery Service needs it on; turn it off when a service mesh resolves endpoint target references locally.")
	fs.DurationVar(&c.ExportResyncPeriod, "export-resync-period", DefaultResyncPeriod, "Interval of the periodic export resync that garbage-collects and resends every export. 0 disables it; otherwise at least 1s.")
	fs.DurationVar(&c.ImportResyncPeriod, "import-resync-period", DefaultResyncPeriod, "Interval at which every import is applied again, repairing drift the watches missed. 0 disables it; otherwise at least 1s.")
	fs.BoolVar(&c.WriteClusterProperties, "write-cluster-properties", false, "Publish the clusterset and cluster ids as ClusterProperty objects.")
	fs.BoolVar(&c.DeregisterOnShutdown, "deregister-on-shutdown", false, "Deregister the cluster from the registry, deleting its exports, when the controller stops.")
	fs.IntVar(&c.EndpointsPerSlice, "endpoints-per-slice", DefaultEndpointsPerSlice, "Maximum endpoints per imported EndpointSlice, at most 1000.")
	fs.IntVar(&c.MaxConcurrentReconciles, "max-concurrent-reconciles", DefaultMaxConcurrentReconciles, "Workers of each controller.")
	fs.StringVar(&c.MetricsBindAddress, "metrics-bind-address", metricsserver.DefaultBindAddress, "Address the metrics endpoint binds to. Use 0 to disable it.")
	fs.StringVar(&c.HealthProbeBindAddress, "health-probe-bind-address", DefaultHealthProbeBindAddress, "Address the /healthz and /readyz endpoints bind to.")
	fs.BoolVar(&c.LeaderElect, "leader-elect", true, "Enable leader election so that one replica writes to the registry and applies imports.")
	ns, _ := lookupEnv(PodNamespaceEnv)
	fs.StringVar(&c.OperatorNamespace, "operator-namespace", strings.TrimSpace(ns), "Namespace of the leader election Lease. Defaults to $"+PodNamespaceEnv+"; required with --leader-elect.")
	fs.BoolVar(&c.OTelEnabled, "otel-enabled", false, "Enable OpenTelemetry tracing, configured by the OTEL_* environment variables.")
}

// Validate reports every invalid field at once.
func (c *Config) Validate() error {
	return errors.Join(slices.Concat(c.validateRegistry(), c.validateIdentity(), c.validateLimits())...)
}

// validateRegistry checks the registry URLs and their TLS settings.
func (c *Config) validateRegistry() []error {
	var errs []error
	if len(c.RegistryURLs) == 0 {
		errs = append(errs, fmt.Errorf("--registry-url (or $%s) is required", RegistryURLsEnv))
	}
	scheme := ""
	for _, raw := range c.RegistryURLs {
		u, err := url.Parse(raw)
		switch {
		case err != nil:
			errs = append(errs, fmt.Errorf("--registry-url %q: %w", raw, err))
			continue
		case u.Scheme != "https" && u.Scheme != "http":
			errs = append(errs, fmt.Errorf("--registry-url %q: scheme must be https or http", raw))
			continue
		case u.Hostname() == "":
			errs = append(errs, fmt.Errorf("--registry-url %q has no host", raw))
		case u.Scheme == "http" && !c.RegistryAllowInsecureHTTP:
			errs = append(errs, fmt.Errorf("--registry-url %q is plaintext; set --registry-allow-insecure-http", raw))
		}
		if scheme == "" {
			scheme = u.Scheme
		} else if u.Scheme != scheme {
			errs = append(errs, fmt.Errorf("--registry-url %q: every URL must use the scheme %s", raw, scheme))
		}
	}
	if c.RegistryTLSCAFile != "" {
		if _, err := os.Stat(c.RegistryTLSCAFile); err != nil {
			errs = append(errs, fmt.Errorf("--registry-tls-ca-file: %w", err))
		}
	}
	return errs
}

// validateIdentity checks the identity source, the ids and the cluster
// labels.
func (c *Config) validateIdentity() []error {
	var errs []error
	switch c.IdentitySource {
	case IdentitySourceAuto, IdentitySourceClusterProperty:
	case IdentitySourceFlags:
		if c.ClusterSetID == "" || c.ClusterID == "" {
			errs = append(errs, errors.New("--cluster-identity-source=flags requires --clusterset-id and --cluster-id"))
		}
	default:
		errs = append(errs, fmt.Errorf("--cluster-identity-source must be auto, flags or clusterproperty, got %q", c.IdentitySource))
	}
	for _, id := range []struct{ flag, value string }{{"--clusterset-id", c.ClusterSetID}, {"--cluster-id", c.ClusterID}} {
		if id.value == "" {
			continue
		}
		if msgs := validation.IsDNS1123Label(id.value); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("%s %q is not a DNS label: %s", id.flag, id.value, strings.Join(msgs, "; ")))
		}
	}
	if c.Region != "" {
		// The importer writes the region as the topology.kubernetes.io/region
		// label of imported EndpointSlices.
		if msgs := validation.IsValidLabelValue(c.Region); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("--region %q is not a label value: %s", c.Region, strings.Join(msgs, "; ")))
		}
	}
	for _, k := range slices.Sorted(maps.Keys(c.ClusterLabels)) {
		if msgs := validation.IsQualifiedName(k); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("--cluster-labels key %q: %s", k, strings.Join(msgs, "; ")))
		}
		if msgs := validation.IsValidLabelValue(c.ClusterLabels[k]); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("--cluster-labels value of %q: %s", k, strings.Join(msgs, "; ")))
		}
	}
	if len(c.ClusterLabels) > api.MaxLabels {
		errs = append(errs, fmt.Errorf("--cluster-labels has %d entries; at most %d are allowed", len(c.ClusterLabels), api.MaxLabels))
	}
	return errs
}

// validateLimits checks the lease, namespaces, sizes, workers, probes and
// leader election.
func (c *Config) validateLimits() []error {
	var errs []error
	if c.LeaseTTL < MinLeaseTTL || c.LeaseTTL > MaxLeaseTTL {
		errs = append(errs, fmt.Errorf("--lease-ttl must be between %s and %s, got %s", MinLeaseTTL, MaxLeaseTTL, c.LeaseTTL))
	}
	for _, ns := range c.WatchNamespaces {
		if msgs := validation.IsDNS1123Label(ns); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("--watch-namespaces entry %q is not a namespace name: %s", ns, strings.Join(msgs, "; ")))
		}
	}
	if c.EndpointsPerSlice < 1 || c.EndpointsPerSlice > MaxEndpointsPerSlice {
		errs = append(errs, fmt.Errorf("--endpoints-per-slice must be between 1 and %d, got %d", MaxEndpointsPerSlice, c.EndpointsPerSlice))
	}
	if c.MaxConcurrentReconciles < 1 {
		errs = append(errs, fmt.Errorf("--max-concurrent-reconciles must be at least 1, got %d", c.MaxConcurrentReconciles))
	}
	for _, p := range []struct {
		flag  string
		value time.Duration
	}{{"--export-resync-period", c.ExportResyncPeriod}, {"--import-resync-period", c.ImportResyncPeriod}} {
		if p.value != 0 && p.value < MinResyncPeriod {
			errs = append(errs, fmt.Errorf("%s must be 0 or at least %s, got %s", p.flag, MinResyncPeriod, p.value))
		}
	}
	if c.HealthProbeBindAddress == "" {
		errs = append(errs, errors.New("--health-probe-bind-address is required"))
	}
	if c.LeaderElect && c.OperatorNamespace == "" {
		errs = append(errs, fmt.Errorf("--operator-namespace (or $%s) is required with --leader-elect", PodNamespaceEnv))
	}
	if c.OperatorNamespace != "" {
		if msgs := validation.IsDNS1123Label(c.OperatorNamespace); len(msgs) > 0 {
			errs = append(errs, fmt.Errorf("--operator-namespace %q is not a namespace name: %s", c.OperatorNamespace, strings.Join(msgs, "; ")))
		}
	}
	return errs
}

// WatchesNamespace reports whether the controller handles objects in ns.
func (c *Config) WatchesNamespace(ns string) bool {
	return len(c.WatchNamespaces) == 0 || slices.Contains(c.WatchNamespaces, ns)
}

// NewScheme returns the controller's scheme: the client-go types (core and
// discovery/v1 among them) and multicluster.x-k8s.io/v1beta1. ClusterProperty
// objects are handled as unstructured.
func NewScheme() (*runtime.Scheme, error) {
	s := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(s); err != nil {
		return nil, err
	}
	if err := mcsv1beta1.Install(s); err != nil {
		return nil, err
	}
	return s, nil
}

// ManagerOptions builds the controller-runtime manager options. With
// WatchNamespaces set, the caches of namespaced objects (Services,
// EndpointSlices, ServiceExports, ServiceImports) are restricted to those
// namespaces; Namespaces themselves stay cluster-wide. Managed fields are
// stripped from cached objects.
func (c *Config) ManagerOptions(scheme *runtime.Scheme) ctrl.Options {
	opts := ctrl.Options{
		Scheme:                        scheme,
		Metrics:                       metricsserver.Options{BindAddress: c.MetricsBindAddress},
		HealthProbeBindAddress:        c.HealthProbeBindAddress,
		LeaderElection:                c.LeaderElect,
		LeaderElectionID:              api.LeaderElectionID,
		LeaderElectionNamespace:       c.OperatorNamespace,
		LeaderElectionReleaseOnCancel: true,
		Cache:                         cache.Options{DefaultTransform: cache.TransformStripManagedFields()},
	}
	if len(c.WatchNamespaces) > 0 {
		namespaces := make(map[string]cache.Config, len(c.WatchNamespaces))
		for _, ns := range c.WatchNamespaces {
			namespaces[ns] = cache.Config{}
		}
		opts.Cache.DefaultNamespaces = namespaces
	}
	return opts
}

// urlList is a flag.Value for repeatable, comma-separated URLs. A default
// from the environment is replaced, not extended, by the first flag.
type urlList struct {
	target     *[]string
	fromEnv    bool
	setOnFlags bool
}

func (l *urlList) setDefault(v string) {
	*l.target = splitList(v)
	l.fromEnv = true
}

func (l *urlList) String() string {
	if l == nil || l.target == nil {
		return ""
	}
	return strings.Join(*l.target, ",")
}

func (l *urlList) Set(v string) error {
	if !l.setOnFlags {
		*l.target = nil
		l.setOnFlags = true
	}
	*l.target = append(*l.target, splitList(v)...)
	return nil
}

func splitList(v string) []string {
	var out []string
	for s := range strings.SplitSeq(v, ",") {
		if s = strings.TrimSpace(s); s != "" {
			out = append(out, s)
		}
	}
	return out
}

// namespaceList is a flag.Value for a comma-separated, de-duplicated and
// sorted list of namespaces.
type namespaceList []string

func (l *namespaceList) String() string {
	if l == nil {
		return ""
	}
	return strings.Join(*l, ",")
}

func (l *namespaceList) Set(value string) error {
	seen := map[string]bool{}
	out := []string{}
	for _, ns := range splitList(value) {
		if !seen[ns] {
			seen[ns] = true
			out = append(out, ns)
		}
	}
	sort.Strings(out)
	*l = out
	return nil
}

// labelMap is a flag.Value for comma-separated key=value pairs.
type labelMap map[string]string

func (m *labelMap) String() string {
	if m == nil || len(*m) == 0 {
		return ""
	}
	parts := make([]string, 0, len(*m))
	for _, k := range slices.Sorted(maps.Keys(*m)) {
		parts = append(parts, k+"="+(*m)[k])
	}
	return strings.Join(parts, ",")
}

func (m *labelMap) Set(value string) error {
	out := map[string]string{}
	for _, kv := range splitList(value) {
		k, v, ok := strings.Cut(kv, "=")
		if !ok || strings.TrimSpace(k) == "" {
			return fmt.Errorf("%q is not key=value", kv)
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	*m = out
	return nil
}
