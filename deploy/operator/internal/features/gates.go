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
	GMSSnapshot Name = "gmsSnapshot"

	// Checkpoint enables checkpoint creation and restore.
	Checkpoint Name = "checkpoint"

	// Grove enables Grove-backed workload orchestration.
	Grove Name = "grove"

	// LWS enables LeaderWorkerSet-backed workload orchestration.
	LWS Name = "lws"

	// KaiScheduler enables Kai Scheduler integration.
	KaiScheduler Name = "kaiScheduler"

	// VolcanoScheduler enables Volcano scheduling for Grove workloads.
	VolcanoScheduler Name = "volcanoScheduler"

	// DRA enables Dynamic Resource Allocation using resource.k8s.io/v1.
	DRA Name = "dra"

	// Istio enables Istio DestinationRule reconciliation.
	Istio Name = "istio"

	// GPUDiscovery enables automatic GPU hardware discovery.
	GPUDiscovery Name = "gpuDiscovery"
)

// GMSSnapshotEnvVar enables GMSSnapshot when set to "1".
const GMSSnapshotEnvVar = "DYN_OPERATOR_ALLOW_GMS_SNAPSHOT"

// Gate reports whether operator features are enabled.
type Gate interface {
	Enabled(Name) bool
}

// Stage is the lifecycle stage of a feature gate.
type Stage string

const (
	StageExperimental Stage = "experimental"
	StageBeta         Stage = "beta"
	StageGA           Stage = "ga"
)

// Definition describes the ownership, lifecycle, and default of a feature gate.
type Definition struct {
	Owner        string
	Stage        Stage
	Since        string
	Default      bool
	DefaultSince string
}

var definitions = map[Name]Definition{
	GMSSnapshot: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	Checkpoint: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	Grove: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	LWS: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	KaiScheduler: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	VolcanoScheduler: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	DRA: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	Istio: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      false,
		DefaultSince: "",
	},
	GPUDiscovery: {
		Owner:        "",
		Stage:        "",
		Since:        "",
		Default:      true,
		DefaultSince: "",
	},
}

// DefinitionFor returns the metadata for name.
func DefinitionFor(name Name) Definition {
	definition, ok := definitions[name]
	if !ok {
		panic(fmt.Sprintf("unknown feature gate %q", name))
	}
	return definition
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
		GMSSnapshot:      DefinitionFor(GMSSnapshot).Default,
		Checkpoint:       DefinitionFor(Checkpoint).Default,
		Grove:            DefinitionFor(Grove).Default,
		LWS:              DefinitionFor(LWS).Default,
		KaiScheduler:     DefinitionFor(KaiScheduler).Default,
		VolcanoScheduler: DefinitionFor(VolcanoScheduler).Default,
		DRA:              DefinitionFor(DRA).Default,
		Istio:            DefinitionFor(Istio).Default,
		GPUDiscovery:     DefinitionFor(GPUDiscovery).Default,
	}
}

func fromEnvironment() Gates {
	gates := Defaults()
	gates.GMSSnapshot = os.Getenv(GMSSnapshotEnvVar) == "1"
	return gates
}

// New detects cluster capabilities and resolves them with operator configuration.
func New(ctx context.Context, mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration) (Gates, error) {
	available := detect(ctx, mgr, config)
	gates := fromEnvironment()
	gates.Checkpoint = config.Checkpoint.Enabled
	gates.GPUDiscovery = config.Namespace.Restricted == "" || ptr.Deref(config.GPU.DiscoveryEnabled, true)

	var err error
	if gates.Grove, err = resolve(config.Orchestrators.Grove.Enabled, available.grove,
		"Grove is explicitly enabled in config but the Grove API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}

	if ptr.Deref(config.Orchestrators.LWS.Enabled, available.lws && available.volcano) {
		if !available.lws {
			return Gates{}, fmt.Errorf("LWS is explicitly enabled in config but the LWS API group was not detected in the cluster")
		}
		if !available.volcano {
			return Gates{}, fmt.Errorf("LWS is explicitly enabled in config but the Volcano API group was not detected in the cluster")
		}
		gates.LWS = true
	}

	if ptr.Deref(config.Orchestrators.VolcanoScheduler.Enabled, false) {
		if !available.volcano {
			return Gates{}, fmt.Errorf("Volcano scheduler integration is explicitly enabled in config but the Volcano API group was not detected in the cluster")
		}
		gates.VolcanoScheduler = true
	}

	if gates.KaiScheduler, err = resolve(config.Orchestrators.KaiScheduler.Enabled, available.kaiScheduler,
		"Kai-scheduler is explicitly enabled in config but the scheduling.run.ai API group was not detected in the cluster"); err != nil {
		return Gates{}, err
	}
	if gates.DRA, err = resolve(config.DRA.Enabled, available.dra,
		"DRA is explicitly enabled in config but the resource.k8s.io/v1 API was not detected in the cluster (requires Kubernetes 1.34+)"); err != nil {
		return Gates{}, err
	}
	if config.ServiceMesh.IsEnabled() {
		if gates.Istio, err = resolve(config.ServiceMesh.Enabled, available.istio,
			"service mesh is explicitly enabled in config but the networking.istio.io DestinationRule API was not detected in the cluster"); err != nil {
			return Gates{}, err
		}
	}

	log.FromContext(ctx).Info("Resolved operator feature gates",
		"grove", gates.Grove,
		"lws", gates.LWS,
		"volcano-scheduler", gates.VolcanoScheduler,
		"kai-scheduler", gates.KaiScheduler,
		"dra", gates.DRA,
		"istio", gates.Istio,
	)
	return gates, nil
}

type availability struct {
	grove        bool
	lws          bool
	volcano      bool
	kaiScheduler bool
	dra          bool
	istio        bool
}

func detect(ctx context.Context, mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration) availability {
	resourceVersion := resourcev1.SchemeGroupVersion.Version
	available := availability{
		grove:        detectAPIGroup(ctx, mgr, "grove.io", ""),
		lws:          detectAPIGroup(ctx, mgr, "leaderworkerset.x-k8s.io", ""),
		volcano:      detectAPIGroup(ctx, mgr, "scheduling.volcano.sh", ""),
		kaiScheduler: detectAPIGroup(ctx, mgr, "scheduling.run.ai", ""),
		dra:          detectAPIGroup(ctx, mgr, resourcev1.SchemeGroupVersion.Group, resourceVersion),
	}
	if ptr.Deref(config.ServiceMesh.Enabled, true) {
		available.istio = DetectIstioDestinationRuleAvailability(ctx, mgr.GetConfig())
	}
	return available
}

// DetectInferencePoolAvailability checks whether the Gateway API Inference Extension is registered.
func DetectInferencePoolAvailability(ctx context.Context, mgr ctrl.Manager) bool {
	return detectAPIGroup(ctx, mgr, "inference.networking.k8s.io", "")
}

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
	switch name {
	case GMSSnapshot:
		return g.GMSSnapshot
	case Checkpoint:
		return g.Checkpoint
	case Grove:
		return g.Grove
	case LWS:
		return g.LWS
	case KaiScheduler:
		return g.KaiScheduler
	case VolcanoScheduler:
		return g.VolcanoScheduler
	case DRA:
		return g.DRA
	case Istio:
		return g.Istio
	case GPUDiscovery:
		return g.GPUDiscovery
	default:
		panic(fmt.Sprintf("unknown feature gate %q", name))
	}
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
