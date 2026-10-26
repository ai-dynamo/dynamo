/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
)

const (
	relayPort       = int32(5561)
	systemPort      = int32(8081)
	sourcesFilePath = "/var/run/dynamo-relay/sources/sources.json"
	mainContainer   = "main"
)

type reconciler struct {
	client.Client
	scheme         *runtime.Scheme
	defaultBackend configv1alpha1.DiscoveryBackend
}

// Setup registers the one-DGD Relay controller and its DGD dependency watch.
func Setup(mgr ctrl.Manager, config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig) error {
	r := &reconciler{Client: mgr.GetClient(), scheme: mgr.GetScheme(), defaultBackend: config.Discovery.Backend}
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.DynamoPoolRelay{}).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(config, runtimeConfig)).
		Watches(&v1beta1.DynamoGraphDeployment{}, handler.EnqueueRequestsFromMapFunc(func(_ context.Context, object client.Object) []ctrl.Request {
			key := types.NamespacedName{Namespace: object.GetNamespace(), Name: object.GetName()}
			return []ctrl.Request{{NamespacedName: key}}
		})).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		Complete(r)
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamopoolrelays/finalizers,verbs=update
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamoworkermetadatas,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;delete
// +kubebuilder:rbac:groups=core,resources=configmaps;serviceaccounts,verbs=get;list;watch;create;update
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update

// Reconcile keeps one Relay Deployment and Service in step with its named DGD.
func (r *reconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	var relay v1alpha1.DynamoPoolRelay
	if err := r.Get(ctx, request.NamespacedName, &relay); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !relay.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// A missing or unsupported DGD must withdraw its discoverable listener.
	var dgd v1beta1.DynamoGraphDeployment
	if err := r.Get(ctx, request.NamespacedName, &dgd); err != nil {
		if !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		return r.withdraw(ctx, &relay, "DGDNotFound", "The named DGD does not exist")
	}
	if !dgd.DeletionTimestamp.IsZero() {
		return r.withdraw(ctx, &relay, "DGDTerminating", "The named DGD is terminating")
	}
	if !supportsPodDiscovery(&dgd, r.defaultBackend) {
		return r.withdraw(ctx, &relay, "UnsupportedDiscovery", "The named DGD must use Kubernetes pod-mode discovery")
	}
	for _, component := range dgd.Spec.Components {
		if !component.GlobalDynamoNamespace {
			continue
		}
		if dynamo.IsWorkerComponent(string(component.ComponentType)) {
			return r.withdraw(ctx, &relay, "UnsupportedSourceScope", "The Relay source file cannot isolate a DGD in the shared global Dynamo namespace")
		}
		if component.ComponentType == v1beta1.ComponentTypeFrontend {
			return r.withdraw(ctx, &relay, "UnsupportedSourceScope", "The Relay cannot isolate frontend load in the shared global Dynamo namespace")
		}
	}
	namespaces := sourceNamespaces(&dgd)
	if len(namespaces) == 0 {
		return r.withdraw(ctx, &relay, "NoWorkers", "The named DGD has no worker components")
	}
	if slices.Contains(namespaces, v1beta1.ComputeDynamoNamespace(true, dgd.Namespace, dgd.Name)) {
		return r.withdraw(ctx, &relay, "UnsupportedSourceScope", "The Relay source file cannot isolate a DGD in the shared global Dynamo namespace")
	}

	resources, err := render(&relay, namespaces)
	if err != nil {
		return r.withdraw(ctx, &relay, "InvalidConfiguration", err.Error())
	}
	for _, resource := range resources {
		if err := r.apply(ctx, &relay, resource); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Deployment status is listener availability, never model or KV readiness.
	deployment := resources[len(resources)-1].(*appsv1.Deployment)
	available := deployment.Status.AvailableReplicas
	desired := ptr.Deref(relay.Spec.Replicas, 2)
	ready := deployment.Status.ObservedGeneration >= deployment.Generation &&
		deployment.Status.UpdatedReplicas == desired && available == desired
	reason, message := "DeploymentProgressing", "Waiting for Relay listeners"
	if ready {
		reason, message = "ListenerAvailable", "All requested Relay listeners are available"
	}
	if err := r.setStatus(ctx, &relay, ready, reason, message, serviceName(relay.Name), available); err != nil {
		return ctrl.Result{}, err
	}
	if !ready {
		return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
	}
	return ctrl.Result{}, nil
}

func supportsPodDiscovery(dgd *v1beta1.DynamoGraphDeployment, defaultBackend configv1alpha1.DiscoveryBackend) bool {
	// Grove takes the DGD metadata backend while ordinary component workloads
	// can use a more specific setting. Publish only if every authored setting
	// agrees on Kubernetes pod discovery.
	if commoncontroller.GetDiscoveryBackend(defaultBackend, dgd.Annotations) != configv1alpha1.DiscoveryBackendKubernetes {
		return false
	}
	compatible := func(values map[string]string) bool {
		return commoncontroller.GetDiscoveryBackend(configv1alpha1.DiscoveryBackendKubernetes, values) == configv1alpha1.DiscoveryBackendKubernetes &&
			commoncontroller.GetKubeDiscoveryMode(values) == configv1alpha1.KubeDiscoveryModePod
	}
	compatibleEnv := func(env []corev1.EnvVar) bool {
		for _, item := range env {
			switch item.Name {
			case commonconsts.DynamoDiscoveryBackendEnvVar:
				if item.ValueFrom != nil || item.Value != string(configv1alpha1.DiscoveryBackendKubernetes) {
					return false
				}
			case "DYN_KUBE_DISCOVERY_MODE":
				if item.ValueFrom != nil || item.Value != string(configv1alpha1.KubeDiscoveryModePod) {
					return false
				}
			}
		}
		return true
	}
	if !compatible(dgd.Annotations) || !compatible(dgd.Spec.Annotations) || !compatibleEnv(dgd.Spec.Env) {
		return false
	}
	var alpha v1alpha1.DynamoGraphDeployment
	if err := alpha.ConvertFrom(dgd); err != nil {
		return false
	}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) && component.ComponentType != v1beta1.ComponentTypeFrontend {
			continue
		}
		if alphaComponent := alpha.Spec.Services[component.ComponentName]; alphaComponent != nil {
			if !compatible(alphaComponent.Annotations) || !compatibleEnv(alphaComponent.Envs) {
				return false
			}
		}
		for _, template := range dynamo.ComponentPodTemplates(component) {
			if !compatible(template.Annotations) {
				return false
			}
			for _, container := range template.Spec.Containers {
				if !compatibleEnv(container.Env) {
					return false
				}
			}
			for _, container := range template.Spec.InitContainers {
				if !compatibleEnv(container.Env) {
					return false
				}
			}
		}
	}
	return true
}

// withdraw removes the publication endpoint before reporting an unresolved source.
func (r *reconciler) withdraw(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, reason, message string) (ctrl.Result, error) {
	name := serviceName(relay.Name)
	metadata := metav1.ObjectMeta{Name: name, Namespace: relay.Namespace}
	var cleanupErrors []error
	for _, object := range []client.Object{
		&corev1.Service{ObjectMeta: metadata},
		&appsv1.Deployment{ObjectMeta: metadata},
	} {
		if err := r.Get(ctx, client.ObjectKeyFromObject(object), object); err != nil {
			if apierrors.IsNotFound(err) {
				continue
			}
			cleanupErrors = append(cleanupErrors, err)
			continue
		}
		if !metav1.IsControlledBy(object, relay) {
			cleanupErrors = append(cleanupErrors, fmt.Errorf("%T %s is not owned by DynamoPoolRelay %s", object, name, relay.Name))
			continue
		}
		if err := r.Delete(ctx, object); err != nil && !apierrors.IsNotFound(err) {
			cleanupErrors = append(cleanupErrors, err)
		}
	}
	if err := r.setStatus(ctx, relay, false, reason, message, "", 0); err != nil {
		cleanupErrors = append(cleanupErrors, err)
	}
	return ctrl.Result{}, errors.Join(cleanupErrors...)
}

// apply changes only resources controlled by this Relay; foreign objects are never adopted.
func (r *reconciler) apply(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, desired client.Object) error {
	current := desired.DeepCopyObject().(client.Object)
	key := client.ObjectKeyFromObject(desired)
	if err := r.Get(ctx, key, current); err != nil {
		if !apierrors.IsNotFound(err) {
			return err
		}
		if err := controllerutil.SetControllerReference(relay, desired, r.scheme); err != nil {
			return err
		}
		return r.Create(ctx, desired)
	}
	if !metav1.IsControlledBy(current, relay) {
		return fmt.Errorf("%T %s is not owned by DynamoPoolRelay %s", current, key, relay.Name)
	}
	if !current.GetDeletionTimestamp().IsZero() {
		return fmt.Errorf("%T %s is being deleted", current, key)
	}
	before := current.DeepCopyObject().(client.Object)

	// Restore controller-owned fields, including values removed from the Relay spec.
	switch object := current.(type) {
	case *corev1.ServiceAccount:
	case *rbacv1.Role:
		object.Rules = desired.(*rbacv1.Role).Rules
	case *rbacv1.RoleBinding:
		object.RoleRef = desired.(*rbacv1.RoleBinding).RoleRef
		object.Subjects = desired.(*rbacv1.RoleBinding).Subjects
	case *corev1.Service:
		object.Spec.Type = corev1.ServiceTypeClusterIP
		object.Spec.Selector = desired.(*corev1.Service).Spec.Selector
		object.Spec.Ports = desired.(*corev1.Service).Spec.Ports
		object.Spec.PublishNotReadyAddresses = false
	case *corev1.ConfigMap:
		object.Data = desired.(*corev1.ConfigMap).Data
	case *appsv1.Deployment:
		mergeDeploymentSpec(object, desired.(*appsv1.Deployment))
	default:
		return fmt.Errorf("unsupported Relay resource %T", current)
	}
	current.SetLabels(desired.GetLabels())
	if equality.Semantic.DeepEqual(before, current) {
		if deployment, ok := desired.(*appsv1.Deployment); ok {
			*deployment = *current.(*appsv1.Deployment)
		}
		return nil
	}
	if err := r.Update(ctx, current); err != nil {
		return fmt.Errorf("update %T %s: %w", current, key, err)
	}
	if deployment, ok := desired.(*appsv1.Deployment); ok {
		*deployment = *current.(*appsv1.Deployment)
	}
	return nil
}

// Keep fields defaulted by the API server when updating an existing Deployment.
// Replacing the whole PodSpec would erase those defaults on each reconcile and
// create an endless rollout loop.
func mergeDeploymentSpec(current, desired *appsv1.Deployment) {
	current.Spec.Replicas = desired.Spec.Replicas
	current.Spec.Selector = desired.Spec.Selector
	current.Spec.Strategy = desired.Spec.Strategy
	current.Spec.Template.Labels = desired.Spec.Template.Labels
	pod, wanted := &current.Spec.Template.Spec, &desired.Spec.Template.Spec
	pod.ServiceAccountName = wanted.ServiceAccountName
	pod.HostNetwork = wanted.HostNetwork
	pod.ImagePullSecrets = wanted.ImagePullSecrets
	pod.TerminationGracePeriodSeconds = wanted.TerminationGracePeriodSeconds
	var sourceVolume corev1.Volume
	if len(pod.Volumes) == 1 {
		sourceVolume = *pod.Volumes[0].DeepCopy()
		if sourceVolume.ConfigMap != nil {
			if ptr.Deref(sourceVolume.ConfigMap.DefaultMode, int32(0644)) == 0644 {
				sourceVolume.ConfigMap.DefaultMode = nil
			}
			if !ptr.Deref(sourceVolume.ConfigMap.Optional, false) {
				sourceVolume.ConfigMap.Optional = nil
			}
		}
	}
	if !equality.Semantic.DeepEqual(sourceVolume, wanted.Volumes[0]) {
		pod.Volumes = wanted.Volumes
	}
	if len(pod.Containers) != 1 || pod.Containers[0].Name != mainContainer {
		pod.Containers = wanted.Containers
		return
	}
	container, target := &pod.Containers[0], &wanted.Containers[0]
	container.Image = target.Image
	container.Command = target.Command
	container.Args = target.Args
	container.Env = target.Env
	container.Ports = target.Ports
	container.VolumeMounts = target.VolumeMounts
	container.StartupProbe = target.StartupProbe
	container.ReadinessProbe = target.ReadinessProbe
	container.Resources = target.Resources
}

func (r *reconciler) setStatus(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, ready bool, reason, message, endpoint string, available int32) error {
	before := relay.DeepCopy()
	condition := metav1.Condition{Type: "Ready", Status: metav1.ConditionFalse, Reason: reason, Message: message, ObservedGeneration: relay.Generation}
	if ready {
		condition.Status = metav1.ConditionTrue
	}
	meta.SetStatusCondition(&relay.Status.Conditions, condition)
	relay.Status.ObservedGeneration = relay.Generation
	relay.Status.AvailableReplicas = available
	relay.Status.Endpoint = ""
	if endpoint != "" {
		relay.Status.Endpoint = fmt.Sprintf("%s.%s.svc:%d", endpoint, relay.Namespace, relayPort)
	}
	if !equality.Semantic.DeepEqual(before.Status, relay.Status) {
		return r.Status().Update(ctx, relay)
	}
	return nil
}

func sourceNamespaces(dgd *v1beta1.DynamoGraphDeployment) []string {
	unique := map[string]struct{}{}
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		namespace := dgd.Status.Components[component.ComponentName].RuntimeNamespace
		if namespace == "" {
			namespace = dgd.GetDynamoNamespaceForComponent(component)
		}
		unique[namespace] = struct{}{}
	}
	namespaces := make([]string, 0, len(unique))
	for namespace := range unique {
		namespaces = append(namespaces, namespace)
	}
	slices.Sort(namespaces)
	return namespaces
}

type sourcesDocument struct {
	Version            int    `json:"version"`
	ConnectionRevision string `json:"connectionRevision"`
	Sources            []struct {
		Namespace string `json:"namespace"`
	} `json:"sources"`
}

func sourceJSON(namespaces []string, connectionRevision string) string {
	document := sourcesDocument{Version: 1, ConnectionRevision: connectionRevision}
	for _, namespace := range namespaces {
		document.Sources = append(document.Sources, struct {
			Namespace string `json:"namespace"`
		}{Namespace: namespace})
	}
	encoded, _ := json.Marshal(document)
	return string(encoded)
}

func serviceName(dgdName string) string { return dgdName + "-pool-relay" }

func render(relay *v1alpha1.DynamoPoolRelay, namespaces []string) ([]client.Object, error) {
	name := serviceName(relay.Name)
	if problems := validation.IsDNS1035Label(name); len(problems) != 0 {
		return nil, fmt.Errorf("DGD name %q cannot produce a valid <dgd>-pool-relay Service name %q: %s", relay.Name, name, strings.Join(problems, "; "))
	}
	clusterID := relay.Spec.ClusterID
	if clusterID == "" || strings.TrimFunc(clusterID, func(r rune) bool {
		return unicode.IsSpace(r) || r >= '\x1c' && r <= '\x1f'
	}) != clusterID {
		return nil, fmt.Errorf("clusterId must be non-empty and have no surrounding whitespace")
	}
	if relay.Spec.Image == "" {
		return nil, fmt.Errorf("image is required")
	}
	replicas := ptr.Deref(relay.Spec.Replicas, 2)
	if replicas < 1 {
		return nil, fmt.Errorf("replicas must be at least one")
	}
	labels := map[string]string{"app.kubernetes.io/name": "pool-relay", "nvidia.com/relay-uid": string(relay.UID)}
	metadata := metav1.ObjectMeta{Name: name, Namespace: relay.Namespace, Labels: labels}

	// Listen for frontend load in the DGD's base runtime namespace even when Grove workers use suffixed namespaces.
	runtimeNamespace := v1beta1.ComputeDynamoNamespace(false, relay.Namespace, relay.Name)

	// Runtime-managed values cannot be replaced by transport configuration.
	env := []corev1.EnvVar{
		{Name: "DYN_NAMESPACE", Value: runtimeNamespace},
		{Name: "DYN_DISCOVERY_BACKEND", Value: "kubernetes"},
		{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "pod"},
		{Name: "DYN_SYSTEM_PORT", Value: fmt.Sprint(systemPort)},
		{Name: "DYN_TCP_RESPONSE_STREAM_PORT", Value: "5562"},
		{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.name"}}},
		{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.namespace"}}},
		{Name: "POD_UID", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.uid"}}},
		{Name: "DYN_TCP_RESPONSE_STREAM_HOST", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "status.podIP"}}},
	}
	reserved := map[string]bool{}
	for _, item := range env {
		reserved[item.Name] = true
	}
	for _, name := range []string{
		"DYN_CLUSTER_ID", "DYN_RELAY_BIND", "DYN_RELAY_CONNECTION_REVISION",
		"DYN_RELAY_NAMESPACES", "DYN_RELAY_WATCH_ALL", "DYN_RELAY_ENDPOINT_PREFIXES",
	} {
		reserved[name] = true
	}
	for _, item := range relay.Spec.Env {
		if reserved[item.Name] {
			return nil, fmt.Errorf("env %s is reserved for the Relay controller", item.Name)
		}
		copy := item.DeepCopy()
		if copy.ValueFrom != nil && copy.ValueFrom.FieldRef != nil && copy.ValueFrom.FieldRef.APIVersion == "" {
			copy.ValueFrom.FieldRef.APIVersion = "v1"
		}
		env = append(env, *copy)
	}
	connection, err := json.Marshal(struct {
		ClusterID string
		Image     string
		Env       []corev1.EnvVar
	}{relay.Spec.ClusterID, relay.Spec.Image, env})
	if err != nil {
		return nil, fmt.Errorf("encode Relay connection settings: %w", err)
	}
	connectionRevision := fmt.Sprintf("%x", sha256.Sum256(connection))
	env = append(env, corev1.EnvVar{Name: "DYN_RELAY_CONNECTION_REVISION", Value: connectionRevision})

	// Every replica receives the same file; the runtime watches projected changes.
	serviceAccount := &corev1.ServiceAccount{ObjectMeta: *metadata.DeepCopy()}
	role := &rbacv1.Role{ObjectMeta: *metadata.DeepCopy(), Rules: []rbacv1.PolicyRule{
		{APIGroups: []string{"nvidia.com"}, Resources: []string{"dynamoworkermetadatas"}, Verbs: []string{"get", "list", "watch", "create", "patch", "delete"}},
		{APIGroups: []string{"discovery.k8s.io"}, Resources: []string{"endpointslices"}, Verbs: []string{"get", "list", "watch"}},
	}}
	binding := &rbacv1.RoleBinding{ObjectMeta: *metadata.DeepCopy(), RoleRef: rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: name}, Subjects: []rbacv1.Subject{{Kind: "ServiceAccount", Name: name, Namespace: relay.Namespace}}}
	configMap := &corev1.ConfigMap{ObjectMeta: *metadata.DeepCopy(), Data: map[string]string{"sources.json": sourceJSON(namespaces, connectionRevision)}}
	service := &corev1.Service{ObjectMeta: *metadata.DeepCopy(), Spec: corev1.ServiceSpec{Type: corev1.ServiceTypeClusterIP, Selector: labels, Ports: []corev1.ServicePort{{Name: "grpc", Port: relayPort, TargetPort: intstr.FromString("grpc"), Protocol: corev1.ProtocolTCP}}}}
	service.Labels["nvidia.com/dynamo-discovery-backend"] = "kubernetes"
	service.Labels["nvidia.com/dynamo-discovery-enabled"] = "true"
	podLabels := metadata.DeepCopy().Labels
	podLabels["nvidia.com/dynamo-discovery-backend"] = "kubernetes"
	podLabels["nvidia.com/dynamo-discovery-enabled"] = "true"
	probeHandler := corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: relayPort, Service: new("dynamo.kvrelay.v1.KvEventRelay")}}
	container := corev1.Container{
		Name: mainContainer, Image: relay.Spec.Image,
		Command:      []string{"python3", "-m", "dynamo.pool_relay"},
		Args:         []string{"--cluster-id", relay.Spec.ClusterID, "--sources-file", sourcesFilePath, "--bind", "0.0.0.0:5561"},
		Env:          env,
		Ports:        []corev1.ContainerPort{{Name: "grpc", ContainerPort: relayPort, Protocol: corev1.ProtocolTCP}, {Name: "system", ContainerPort: systemPort, Protocol: corev1.ProtocolTCP}, {Name: "recovery", ContainerPort: 5562, Protocol: corev1.ProtocolTCP}},
		VolumeMounts: []corev1.VolumeMount{{Name: "relay-sources", MountPath: "/var/run/dynamo-relay/sources", ReadOnly: true}},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("1"), corev1.ResourceMemory: resource.MustParse("1Gi")},
			Limits:   corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("4Gi")},
		},
		StartupProbe:   &corev1.Probe{ProbeHandler: probeHandler, PeriodSeconds: 5, TimeoutSeconds: 1, SuccessThreshold: 1, FailureThreshold: 60},
		ReadinessProbe: &corev1.Probe{ProbeHandler: probeHandler, PeriodSeconds: 10, TimeoutSeconds: 1, SuccessThreshold: 1, FailureThreshold: 3},
	}
	sourceReference := corev1.LocalObjectReference{Name: name}
	sourceConfigMap := &corev1.ConfigMapVolumeSource{LocalObjectReference: sourceReference}
	sourceVolume := corev1.VolumeSource{ConfigMap: sourceConfigMap}
	sourcesVolume := corev1.Volume{Name: "relay-sources", VolumeSource: sourceVolume}
	podMeta := metav1.ObjectMeta{Labels: podLabels}
	deployment := &appsv1.Deployment{ObjectMeta: *metadata.DeepCopy(), Spec: appsv1.DeploymentSpec{
		Replicas: new(replicas), Selector: &metav1.LabelSelector{MatchLabels: labels},
		Strategy: appsv1.DeploymentStrategy{Type: appsv1.RollingUpdateDeploymentStrategyType, RollingUpdate: &appsv1.RollingUpdateDeployment{MaxUnavailable: new(intstr.FromInt32(0)), MaxSurge: new(intstr.FromInt32(1))}},
		Template: corev1.PodTemplateSpec{ObjectMeta: podMeta, Spec: corev1.PodSpec{
			ServiceAccountName:            name,
			ImagePullSecrets:              append([]corev1.LocalObjectReference(nil), relay.Spec.ImagePullSecrets...),
			TerminationGracePeriodSeconds: new(int64(60)),
			Containers:                    []corev1.Container{container},
			Volumes:                       []corev1.Volume{sourcesVolume},
		}},
	}}
	return []client.Object{serviceAccount, role, binding, configMap, service, deployment}, nil
}
