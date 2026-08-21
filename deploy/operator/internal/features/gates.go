/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package features defines operator feature gates.
package features

import (
	"context"
	"errors"
	"fmt"
	"os"
	"reflect"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Name identifies an operator feature gate.
type Name string

const (
	// Checkpoint enables checkpoint creation and restore.
	//
	// Owner: @galletas1712
	// Experimental since: v1.0.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: checkpoint.enabled
	// Auto-detection: nvidia.com/v1alpha1 PodSnapshot resource
	// Requires: Snapshot operator serving nvidia.com/v1alpha1 PodSnapshot resources
	// Default: false
	Checkpoint Name = "checkpoint"

	// Grove enables Grove-backed workload orchestration.
	//
	// Owner: @julienmancuso
	// Experimental since: v0.4.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: orchestrators.grove.enabled
	// Auto-detection: grove.io API group
	// Requires: Grove serving grove.io/v1alpha1
	// Default: true when the grove.io API group is detected; false otherwise
	Grove Name = "grove"

	// LWS enables LeaderWorkerSet-backed workload orchestration.
	//
	// Owner: @julienmancuso
	// Experimental since: v0.3.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: orchestrators.lws.enabled
	// Auto-detection: leaderworkerset.x-k8s.io and scheduling.volcano.sh API groups
	// Requires: LWS serving leaderworkerset.x-k8s.io/v1 and Volcano serving
	// scheduling.volcano.sh/v1beta1
	// Default: true when both API groups are detected; false otherwise
	LWS Name = "lws"

	// KaiScheduler enables Kai Scheduler integration.
	//
	// Owner: @julienmancuso
	// Experimental since: v0.5.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: orchestrators.kaiScheduler.enabled
	// Auto-detection: scheduling.run.ai API group
	// Requires: Kai Scheduler serving scheduling.run.ai/v2 Queue resources
	// Default: true when the scheduling.run.ai API group is detected; false otherwise
	KaiScheduler Name = "kaiScheduler"

	// VolcanoScheduler injects Volcano scheduler settings into Grove PodCliqueSets.
	//
	// Owner: @xianlubird
	// Experimental since: v1.4.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: orchestrators.volcanoScheduler.enabled
	// Auto-detection: N/A; API availability is verified when explicitly enabled
	// Requires: Volcano serving scheduling.volcano.sh/v1beta1
	// Default: false
	VolcanoScheduler Name = "volcanoScheduler"

	// DRA enables Dynamic Resource Allocation using resource.k8s.io/v1.
	//
	// Owner: @julienmancuso
	// Experimental since: v1.2.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: dra.enabled
	// Auto-detection: resource.k8s.io/v1 API
	// Requires: Kubernetes 1.34 or later
	// Default: true when resource.k8s.io/v1 is detected; false otherwise
	DRA Name = "dra"

	// Istio enables Istio DestinationRule reconciliation.
	//
	// Owner: @atchernych
	// Experimental since: N/A
	// Beta since: N/A
	// GA since: v1.2.0
	// Configuration: serviceMesh.provider=istio and serviceMesh.enabled
	// Auto-detection: networking.istio.io/v1beta1 DestinationRule resource
	// Requires: Istio serving networking.istio.io/v1beta1 DestinationRule resources
	// Default: true when provider is istio and DestinationRule is detected; false otherwise
	Istio Name = "istio"

	// GPUDiscovery enables automatic GPU hardware discovery.
	//
	// Owner: @hhzhang16
	// Experimental since: N/A
	// Beta since: N/A
	// GA since: v1.0.0
	// Configuration: gpu.discoveryEnabled in namespace-restricted mode;
	// cluster-wide mode always enables the gate
	// Auto-detection: N/A
	// Requires: N/A
	// Default: true, since v1.0.0
	GPUDiscovery Name = "gpuDiscovery"

	// ElasticEPAutoscale enables the operator to fire elastic-EP scale calls on its own
	// initiative (Phase 6). It is OFF by default and gated deliberately: until
	// https://github.com/ai-dynamo/dynamo/pull/12991 gives scale_elastic_ep a clean
	// rollback, a failed grow can leave the engine holding a reserved-but-dead rank and
	// serving nothing while the pod still looks healthy -- an operator that can wedge a
	// serving engine unattended is worse than a human choosing to take the risk. Flip the
	// default to true once that rollback lands.
	//
	// Owner: @tzulingk
	// Experimental since: v1.3.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: DYN_OPERATOR_ENABLE_ELASTIC_EP_AUTOSCALE=1
	// Auto-detection: N/A
	// Requires: elastic-EP leader Service (Phase 3) and get_ep_capacity (Phase 5)
	// Default: false
	ElasticEPAutoscale Name = "elasticEPAutoscale"
)

var allNames = [...]Name{
	Checkpoint,
	Grove,
	LWS,
	KaiScheduler,
	VolcanoScheduler,
	DRA,
	Istio,
	GPUDiscovery,
	ElasticEPAutoscale,
}

// Gate reports whether operator features are enabled.
type Gate interface {
	Enabled(Name) bool
}

// Gates is the complete set of operator feature gates.
type Gates struct {
	Checkpoint         bool `json:"checkpoint"`
	Grove              bool `json:"grove"`
	LWS                bool `json:"lws"`
	KaiScheduler       bool `json:"kaiScheduler"`
	VolcanoScheduler   bool `json:"volcanoScheduler"`
	DRA                bool `json:"dra"`
	Istio              bool `json:"istio"`
	GPUDiscovery       bool `json:"gpuDiscovery"`
	ElasticEPAutoscale bool `json:"elasticEPAutoscale"`
}

// Defaults returns the default feature gates.
func Defaults() Gates {
	return Gates{
		GPUDiscovery: true,
	}
}

// New detects cluster capabilities and resolves them with operator configuration.
func New(ctx context.Context, mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration) (Gates, error) {
	gates := Defaults()
	// Off unless explicitly opted in; see the ElasticEPAutoscale gate doc for why it stays
	// gated until the scale_elastic_ep rollback (#12991) lands.
	gates.ElasticEPAutoscale = os.Getenv("DYN_OPERATOR_ENABLE_ELASTIC_EP_AUTOSCALE") == "1"
	gates.GPUDiscovery = config.Namespace.Restricted == "" || ptr.Deref(config.GPU.DiscoveryEnabled, true)

	var err error
	// Enable Checkpoint only when explicitly configured and its external API dependency is available.
	if config.Checkpoint.Enabled {
		podSnapshotResource := snapshotv1alpha1.GroupVersion.WithResource("podsnapshots")
		podSnapshotAvailable, detectErr := detectAPIAvailability(
			ctx,
			mgr.GetConfig(),
			podSnapshotResource.Group,
			podSnapshotResource.Version,
			podSnapshotResource.Resource,
		)
		if detectErr != nil {
			return Gates{}, detectErr
		}
		if gates.Checkpoint, err = resolve(ptr.To(true), podSnapshotAvailable,
			"checkpoint is explicitly enabled in config but the nvidia.com/v1alpha1 PodSnapshot API was not detected in the cluster"); err != nil {
			return Gates{}, err
		}
	}
	groveAvailable, err := detectAPIAvailability(ctx, mgr.GetConfig(), "grove.io", "", "")
	if err != nil {
		return Gates{}, err
	}
	if gates.Grove, err = resolve(config.Orchestrators.Grove.Enabled, groveAvailable,
		"Grove is explicitly enabled in config but the Grove API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}

	lwsAvailable, err := detectAPIAvailability(ctx, mgr.GetConfig(), "leaderworkerset.x-k8s.io", "", "")
	if err != nil {
		return Gates{}, err
	}
	volcanoAvailable, err := detectAPIAvailability(ctx, mgr.GetConfig(), "scheduling.volcano.sh", "", "")
	if err != nil {
		return Gates{}, err
	}
	if ptr.Deref(config.Orchestrators.LWS.Enabled, lwsAvailable && volcanoAvailable) {
		if !lwsAvailable {
			return Gates{}, fmt.Errorf("LWS is explicitly enabled in config but the LWS API group was not detected in the cluster")
		}
		if !volcanoAvailable {
			return Gates{}, fmt.Errorf("LWS is explicitly enabled in config but the Volcano API group was not detected in the cluster")
		}
		gates.LWS = true
	}

	if ptr.Deref(config.Orchestrators.VolcanoScheduler.Enabled, false) {
		if !volcanoAvailable {
			return Gates{}, fmt.Errorf("Volcano scheduler integration is explicitly enabled in config but the Volcano API group was not detected in the cluster")
		}
		gates.VolcanoScheduler = true
	}

	kaiSchedulerAvailable, err := detectAPIAvailability(ctx, mgr.GetConfig(), "scheduling.run.ai", "", "")
	if err != nil {
		return Gates{}, err
	}
	if gates.KaiScheduler, err = resolve(config.Orchestrators.KaiScheduler.Enabled, kaiSchedulerAvailable,
		"Kai-scheduler is explicitly enabled in config but the scheduling.run.ai API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}
	draAvailable, err := detectAPIAvailability(
		ctx,
		mgr.GetConfig(),
		resourcev1.SchemeGroupVersion.Group,
		resourcev1.SchemeGroupVersion.Version,
		"",
	)
	if err != nil {
		return Gates{}, err
	}
	if gates.DRA, err = resolve(config.DRA.Enabled, draAvailable,
		"DRA is explicitly enabled in config but the resource.k8s.io/v1 API was not detected in the cluster (requires Kubernetes 1.34+)"); err != nil {
		return Gates{}, err
	}
	if config.ServiceMesh.IsEnabled() {
		istioAvailable, detectErr := DetectIstioDestinationRuleAvailability(ctx, mgr.GetConfig())
		if detectErr != nil {
			return Gates{}, detectErr
		}
		if gates.Istio, err = resolve(config.ServiceMesh.Enabled, istioAvailable,
			"service mesh is explicitly enabled in config but the networking.istio.io DestinationRule API was not detected in the cluster"); err != nil {
			return Gates{}, err
		}
	}

	logValues := make([]any, 0, 2*len(allNames))
	for _, name := range allNames {
		logValues = append(logValues, string(name), gates.Enabled(name))
	}
	log.FromContext(ctx).Info("Resolved operator feature gates", logValues...)
	return gates, nil
}

// DetectInferencePoolAvailability checks whether the Gateway API Inference Extension is registered.
func DetectInferencePoolAvailability(ctx context.Context, mgr ctrl.Manager) (bool, error) {
	return detectAPIAvailability(ctx, mgr.GetConfig(), "inference.networking.k8s.io", "", "")
}

// detectAPIAvailability reports whether an API group, version, or resource is discoverable.
// An empty version checks only the group. An empty resource checks the group and optional version.
// A nil cfg is supported and returns an error because discovery is unavailable.
func detectAPIAvailability(ctx context.Context, cfg *rest.Config, group, version, resource string) (bool, error) {
	logger := log.FromContext(ctx)
	logValues := []any{"group", group}
	if version != "" {
		logValues = append(logValues, "version", version)
	}
	if resource != "" {
		logValues = append(logValues, "resource", resource)
	}
	if cfg == nil {
		return false, errors.New("API detection failed, no discovery client available")
	}
	if resource != "" && version == "" {
		return false, errors.New("API resource detection requires a version")
	}

	// Create a client for direct API resource discovery.
	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		return false, fmt.Errorf("create API discovery client: %w", err)
	}

	// Group-only detection uses the server group index and optionally matches a served version.
	if resource == "" {
		apiGroups, err := discoveryClient.ServerGroups()
		if err != nil {
			return false, fmt.Errorf("list server API groups: %w", err)
		}
		available := apiGroupServesVersion(apiGroups, group, version)
		logger.Info("API availability detected", append(logValues, "available", available)...)
		return available, nil
	}
	// Query the exact group version and distinguish absence from discovery failures.
	groupVersion := group + "/" + version
	apiResourceList, err := discoveryClient.ServerResourcesForGroupVersion(groupVersion)
	if apierrors.IsNotFound(err) {
		logger.Info("API availability detected", append(logValues, "available", false)...)
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("discover %s API resources: %w", groupVersion, err)
	}

	// Match the exact resource within the discovered group version.
	for _, candidate := range apiResourceList.APIResources {
		if candidate.Name == resource {
			logger.Info("API availability detected", append(logValues, "available", true)...)
			return true, nil
		}
	}
	logger.Info("API availability detected", append(logValues, "available", false)...)
	return false, nil
}

// resolve uses auto-detection when unset, disables on false, and requires availability on true.
func resolve(configured *bool, available bool, unavailableMessage string) (bool, error) {
	if configured == nil {
		return available, nil
	}
	if !*configured {
		return false, nil
	}
	if !available {
		return false, errors.New(unavailableMessage)
	}
	return true, nil
}

// DetectIstioDestinationRuleAvailability checks whether DestinationRule is registered.
func DetectIstioDestinationRuleAvailability(ctx context.Context, cfg *rest.Config) (bool, error) {
	return detectAPIAvailability(ctx, cfg, "networking.istio.io", "v1beta1", "destinationrules")
}

func apiGroupServesVersion(apiGroups *metav1.APIGroupList, groupName, version string) bool {
	if apiGroups == nil {
		return false
	}
	for _, group := range apiGroups.Groups {
		if group.Name != groupName {
			continue
		}
		if version == "" {
			return true
		}
		for _, served := range group.Versions {
			if served.Version == version {
				return true
			}
		}
		return false
	}
	return false
}

// Enabled reports whether name is enabled.
func (g Gates) Enabled(name Name) bool {
	value := reflect.ValueOf(g)
	typeOfGates := value.Type()
	for i := 0; i < value.NumField(); i++ {
		if typeOfGates.Field(i).Tag.Get("json") == string(name) {
			return value.Field(i).Bool()
		}
	}
	panic(fmt.Sprintf("unknown feature gate %q", name))
}

type gateContextKey struct{}

// WithGate attaches the effective feature gates to a request context.
func WithGate(ctx context.Context, gate Gate) context.Context {
	if gate == nil {
		panic("feature gate must not be nil")
	}
	return context.WithValue(ctx, gateContextKey{}, gate)
}

// GateFrom returns the effective feature gates from a request context.
func GateFrom(ctx context.Context) (Gate, bool) {
	gate, ok := ctx.Value(gateContextKey{}).(Gate)
	return gate, ok
}

// MustGateFrom returns the effective feature gates or panics when none are attached.
// Prefer it when a gate is required so broken context plumbing fails immediately.
func MustGateFrom(ctx context.Context) Gate {
	gate, ok := GateFrom(ctx)
	if !ok {
		panic("feature gate missing from context")
	}
	return gate
}
