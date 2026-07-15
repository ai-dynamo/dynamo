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
	resourcev1 "k8s.io/api/resource/v1"
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
	// GMSSnapshot enables the temporary GMS + Snapshot integration.
	//
	// Owner: @galletas1712
	// Experimental since: v1.2.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: DYN_OPERATOR_ALLOW_GMS_SNAPSHOT=1
	// Auto-detection: N/A
	// Requires: N/A
	// Default: false
	GMSSnapshot Name = "gmsSnapshot"

	// Checkpoint enables checkpoint creation and restore.
	//
	// Owner: @galletas1712
	// Experimental since: v1.0.0
	// Beta since: N/A
	// GA since: N/A
	// Configuration: checkpoint.enabled
	// Auto-detection: N/A
	// Requires: N/A
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
)

var allNames = [...]Name{
	GMSSnapshot,
	Checkpoint,
	Grove,
	LWS,
	KaiScheduler,
	VolcanoScheduler,
	DRA,
	Istio,
	GPUDiscovery,
}

// GMSSnapshotEnvVar enables the temporary internal GMS + Snapshot feature gate when set to "1".
const GMSSnapshotEnvVar = "DYN_OPERATOR_ALLOW_GMS_SNAPSHOT"

// Gate reports whether operator features are enabled.
type Gate interface {
	Enabled(Name) bool
}

// Gates is the complete set of operator feature gates.
type Gates struct {
	GMSSnapshot      bool `json:"gmsSnapshot"`
	Checkpoint       bool `json:"checkpoint"`
	Grove            bool `json:"grove"`
	LWS              bool `json:"lws"`
	KaiScheduler     bool `json:"kaiScheduler"`
	VolcanoScheduler bool `json:"volcanoScheduler"`
	DRA              bool `json:"dra"`
	Istio            bool `json:"istio"`
	GPUDiscovery     bool `json:"gpuDiscovery"`
}

// Defaults returns the default feature gates.
func Defaults() Gates {
	return Gates{
		GPUDiscovery: true,
	}
}

func fromEnvironment() Gates {
	gates := Defaults()
	gates.GMSSnapshot = os.Getenv(GMSSnapshotEnvVar) == "1"
	return gates
}

// New detects cluster capabilities and resolves them with operator configuration.
func New(ctx context.Context, mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration) (Gates, error) {
	gates := fromEnvironment()
	gates.Checkpoint = config.Checkpoint.Enabled
	gates.GPUDiscovery = config.Namespace.Restricted == "" || ptr.Deref(config.GPU.DiscoveryEnabled, true)

	var err error
	if gates.Grove, err = resolve(config.Orchestrators.Grove.Enabled, detectAPIGroup(ctx, mgr, "grove.io", ""),
		"Grove is explicitly enabled in config but the Grove API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}

	lwsAvailable := detectAPIGroup(ctx, mgr, "leaderworkerset.x-k8s.io", "")
	volcanoAvailable := detectAPIGroup(ctx, mgr, "scheduling.volcano.sh", "")
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

	if gates.KaiScheduler, err = resolve(config.Orchestrators.KaiScheduler.Enabled, detectAPIGroup(ctx, mgr, "scheduling.run.ai", ""),
		"Kai-scheduler is explicitly enabled in config but the scheduling.run.ai API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}
	if gates.DRA, err = resolve(config.DRA.Enabled,
		detectAPIGroup(ctx, mgr, resourcev1.SchemeGroupVersion.Group, resourcev1.SchemeGroupVersion.Version),
		"DRA is explicitly enabled in config but the resource.k8s.io/v1 API was not detected in the cluster (requires Kubernetes 1.34+)"); err != nil {
		return Gates{}, err
	}
	if config.ServiceMesh.IsEnabled() {
		if gates.Istio, err = resolve(config.ServiceMesh.Enabled, DetectIstioDestinationRuleAvailability(ctx, mgr.GetConfig()),
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
func DetectInferencePoolAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroup(ctx, mgr, "inference.networking.k8s.io", "")
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

func detectAPIGroup(ctx context.Context, mgr ctrl.Manager, groupName, version string) bool {
	logger := log.FromContext(ctx)
	logValues := []any{"group", groupName}
	if version != "" {
		logValues = append(logValues, "version", version)
	}

	cfg := mgr.GetConfig()
	if cfg == nil {
		logger.Info("detection failed, no discovery client available", logValues...)
		return false
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		logger.Error(err, "detection failed, could not create discovery client", logValues...)
		return false
	}
	apiGroups, err := discoveryClient.ServerGroups()
	if err != nil {
		logger.Error(err, "detection failed, could not list server groups", logValues...)
		return false
	}
	if apiGroupServesVersion(apiGroups, groupName, version) {
		logger.Info("API group is available", logValues...)
		return true
	}
	logger.Info("API group not available", logValues...)
	return false
}

// DetectIstioDestinationRuleAvailability checks whether DestinationRule is registered.
func DetectIstioDestinationRuleAvailability(ctx context.Context, cfg *rest.Config) bool {
	logger := log.FromContext(ctx)
	logValues := []any{"groupVersion", "networking.istio.io/v1beta1", "resource", "destinationrules"}
	if cfg == nil {
		logger.Info("detection failed, no discovery client available", logValues...)
		return false
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		logger.Error(err, "detection failed, could not create discovery client", logValues...)
		return false
	}
	apiResourceList, err := discoveryClient.ServerResourcesForGroupVersion("networking.istio.io/v1beta1")
	if err != nil {
		logger.Info("API resource not available", append(logValues, "error", err.Error())...)
		return false
	}
	for _, resource := range apiResourceList.APIResources {
		if resource.Name == "destinationrules" {
			logger.Info("API resource is available", logValues...)
			return true
		}
	}
	logger.Info("API resource not available", logValues...)
	return false
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

// GateFromContext returns the effective feature gates from a request context.
func GateFromContext(ctx context.Context) Gate {
	gate, ok := ctx.Value(gateContextKey{}).(Gate)
	if !ok {
		panic("feature gate missing from context")
	}
	return gate
}
