package dynamo

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	ctrl "sigs.k8s.io/controller-runtime"
)

type GroveMultinodeDeployer struct {
	MultinodeDeployer
}

func (d *GroveMultinodeDeployer) GetLeaderHostname(serviceName string) string {
	return fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-0.$(GROVE_HEADLESS_SERVICE)", strings.ToLower(serviceName), commonconsts.GroveRoleSuffixLeader)
}

func (d *GroveMultinodeDeployer) GetNodeRank() (string, bool) {
	// This requires shell expansion for arithmetic expression
	return "$((GROVE_PCLQ_POD_INDEX + 1))", true
}

func (d *GroveMultinodeDeployer) NeedsDNSWait() bool {
	// Grove doesn't need DNS wait - it handles startup coordination differently
	return false
}

func (d *GroveMultinodeDeployer) GetHostNames(serviceName string, numberOfNodes int32) []string {
	hostnames := make([]string, 0, numberOfNodes)
	leaderHostname := d.GetLeaderHostname(serviceName)
	hostnames = append(hostnames, leaderHostname)
	// Add worker hostnames
	for i := int32(0); i < numberOfNodes-1; i++ {
		workerHostname := fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-%s-%s-%d.$(GROVE_HEADLESS_SERVICE)",
			strings.ToLower(serviceName), commonconsts.GroveRoleSuffixWorker, i)
		hostnames = append(hostnames, workerHostname)
	}
	return hostnames
}

// GetComponentReadinessAndServiceReplicaStatuses determines if all Grove components are ready
// and returns the service replica statuses for each component.
// - PodCliques: spec.replicas == status.readyReplicas
// - PodCliqueScalingGroups: spec.replicas == status.availableReplicas
func GetComponentReadinessAndServiceReplicaStatuses(ctx context.Context, client client.Client, dgd *nvidiacomv1alpha1.DynamoGraphDeployment) (bool, string, map[string]v1alpha1.ServiceReplicaStatus) {
	logger := log.FromContext(ctx)
	var notReadyComponents []string

	serviceStatuses := make(map[string]v1alpha1.ServiceReplicaStatus, len(dgd.Spec.Services))

	for serviceName, component := range dgd.Spec.Services {
		isMultinode := component.GetNumberOfNodes() > 1
		resourceName := fmt.Sprintf("%s-0-%s", dgd.Name, strings.ToLower(serviceName))

		if isMultinode {
			// Check PodCliqueScalingGroup: spec.replicas == status.availableReplicas
			ok, reason, serviceStatus := CheckPCSGReady(ctx, client, resourceName, dgd.Namespace, logger)
			serviceStatuses[serviceName] = serviceStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("pcsg/%s: %s", resourceName, reason))
			}
		} else {
			// Check PodClique: spec.replicas == status.readyReplicas
			ok, reason, serviceStatus := CheckPodCliqueReady(ctx, client, resourceName, dgd.Namespace, logger)
			serviceStatuses[serviceName] = serviceStatus
			if !ok {
				notReadyComponents = append(notReadyComponents, fmt.Sprintf("podclique/%s: %s", resourceName, reason))
			}
		}
	}

	if len(notReadyComponents) > 0 {
		return false, strings.Join(notReadyComponents, "; "), serviceStatuses
	}

	return true, "", serviceStatuses
}

// CheckPodCliqueReady determines if a Grove PodClique is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodClique
// configuration has been fully applied. This is the PodClique equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPodCliqueReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1alpha1.ServiceReplicaStatus) {
	podClique := &grovev1alpha1.PodClique{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, podClique)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodClique not found", "resourceName", resourceName)
			return false, "resource not found", v1alpha1.ServiceReplicaStatus{}
		}
		logger.V(1).Info("Failed to get PodClique", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err), v1alpha1.ServiceReplicaStatus{}
	}

	desiredReplicas := podClique.Spec.Replicas
	readyReplicas := podClique.Status.ReadyReplicas
	updatedReplicas := podClique.Status.UpdatedReplicas
	replicas := podClique.Status.Replicas
	observedGeneration := podClique.Status.ObservedGeneration
	generation := podClique.Generation

	logger.V(1).Info("CheckPodCliqueFullyUpdated",
		"resourceName", resourceName,
		"generation", podClique.Generation,
		"observedGeneration", podClique.Status.ObservedGeneration,
		"desiredReplicas", desiredReplicas,
		"readyReplicas", readyReplicas,
		"updatedReplicas", updatedReplicas,
		"replicas", replicas,
	)

	serviceStatus := v1alpha1.ServiceReplicaStatus{
		ComponentKind:   v1alpha1.ComponentKindPodClique,
		ComponentName:   resourceName,
		ComponentNames:  []string{resourceName},
		Replicas:        podClique.Status.Replicas,
		UpdatedReplicas: podClique.Status.UpdatedReplicas,
		ReadyReplicas:   &readyReplicas,
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodClique observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodClique spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus
	}

	if desiredReplicas == 0 {
		return true, "", serviceStatus
	}

	if desiredReplicas != readyReplicas {
		logger.V(1).Info("PodClique not ready", "resourceName", resourceName, "desired", desiredReplicas, "ready", readyReplicas)
		return false, fmt.Sprintf("desired=%d, ready=%d", desiredReplicas, readyReplicas), serviceStatus
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodClique not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodClique performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus
	}

	return true, "", serviceStatus
}

// CheckPCSGReady determines if a Grove PodCliqueScalingGroup is fully ready and available.
// It checks various status fields to ensure all replicas are available and the PodClique
// configuration has been fully applied. This is the PodCliqueScalingGroup equivalent of IsDeploymentReady
// for standard Kubernetes Deployments.
func CheckPCSGReady(ctx context.Context, client client.Client, resourceName, namespace string, logger logr.Logger) (bool, string, v1alpha1.ServiceReplicaStatus) {
	pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, pcsg)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("PodCliqueScalingGroup not found", "resourceName", resourceName)
			return false, "resource not found", v1alpha1.ServiceReplicaStatus{}
		}
		logger.V(1).Info("Failed to get PodCliqueScalingGroup", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err), v1alpha1.ServiceReplicaStatus{}
	}

	desiredReplicas := pcsg.Spec.Replicas
	availableReplicas := pcsg.Status.AvailableReplicas
	updatedReplicas := pcsg.Status.UpdatedReplicas
	replicas := pcsg.Status.Replicas
	observedGeneration := pcsg.Status.ObservedGeneration
	generation := pcsg.Generation

	logger.V(1).Info("CheckPCSGFullyUpdated",
		"resourceName", resourceName,
		"generation", pcsg.Generation,
		"observedGeneration", pcsg.Status.ObservedGeneration,
		"desiredReplicas", desiredReplicas,
		"availableReplicas", availableReplicas,
		"updatedReplicas", updatedReplicas,
		"replicas", replicas,
	)

	serviceStatus := v1alpha1.ServiceReplicaStatus{
		ComponentKind:     v1alpha1.ComponentKindPodCliqueScalingGroup,
		ComponentName:     resourceName,
		ComponentNames:    []string{resourceName},
		Replicas:          pcsg.Status.Replicas,
		UpdatedReplicas:   pcsg.Status.UpdatedReplicas,
		AvailableReplicas: &availableReplicas,
	}

	if observedGeneration == nil {
		logger.V(1).Info("PodCliqueScalingGroup observedGeneration is nil", "resourceName", resourceName)
		return false, "observedGeneration is nil", serviceStatus
	}

	if observedGeneration != nil && *observedGeneration < generation {
		logger.V(1).Info("PodCliqueScalingGroup spec not yet processed", "resourceName", resourceName, "generation", generation, "observedGeneration", observedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", generation, *observedGeneration), serviceStatus
	}

	if desiredReplicas == 0 {
		// No replicas desired, so it's ready
		return true, "", serviceStatus
	}

	if desiredReplicas != availableReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not ready", "resourceName", resourceName, "desired", desiredReplicas, "available", availableReplicas)
		return false, fmt.Sprintf("desired=%d, available=%d", desiredReplicas, availableReplicas), serviceStatus
	}

	if desiredReplicas != updatedReplicas {
		logger.V(1).Info("PodCliqueScalingGroup not fully updated", "resourceName", resourceName, "desired", desiredReplicas, "updated", updatedReplicas)
		return false, fmt.Sprintf("desired=%d, updated=%d", desiredReplicas, updatedReplicas), serviceStatus
	}

	if replicas != desiredReplicas {
		logger.V(1).Info("PodCliqueScalingGroup performing rolling update", "resourceName", resourceName, "desired", desiredReplicas, "replicas", replicas)
		return false, fmt.Sprintf("performing rolling update: desired=%d, replicas=%d", desiredReplicas, replicas), serviceStatus
	}

	return true, "", serviceStatus
}

// specToGroveTopologyConstraint converts a deployment-level SpecTopologyConstraint
// to a Grove TopologyConstraint, preserving the topology profile when the
// Grove API exposes it.
func specToGroveTopologyConstraint(tc *v1alpha1.SpecTopologyConstraint) *grovev1alpha1.TopologyConstraint {
	if tc == nil || (tc.PackDomain == "" && strings.TrimSpace(tc.TopologyProfile) == "") {
		return nil
	}
	constraint := &grovev1alpha1.TopologyConstraint{}
	if tc.PackDomain != "" {
		constraint.PackDomain = grovev1alpha1.TopologyDomain(tc.PackDomain)
	}
	setOptionalStringField(constraint, "TopologyProfile", tc.TopologyProfile)
	return constraint
}

// toGroveTopologyConstraint converts a service-level TopologyConstraint to a
// Grove TopologyConstraint and carries the inherited deployment topology
// profile when the Grove API supports it.
func toGroveTopologyConstraint(tc *v1alpha1.TopologyConstraint, inherited *v1alpha1.SpecTopologyConstraint) *grovev1alpha1.TopologyConstraint {
	if tc == nil {
		return nil
	}
	inheritedProfile := ""
	if inherited != nil {
		inheritedProfile = inherited.TopologyProfile
	}
	if tc.PackDomain == "" && strings.TrimSpace(inheritedProfile) == "" {
		return nil
	}
	constraint := &grovev1alpha1.TopologyConstraint{}
	if tc.PackDomain != "" {
		constraint.PackDomain = grovev1alpha1.TopologyDomain(tc.PackDomain)
	}
	setOptionalStringField(constraint, "TopologyProfile", inheritedProfile)
	return constraint
}

func setOptionalStringField(target any, fieldName, value string) {
	if value == "" {
		return
	}
	v := reflect.ValueOf(target)
	if !v.IsValid() || v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	field := v.Elem().FieldByName(fieldName)
	if !field.IsValid() || !field.CanSet() || field.Kind() != reflect.String {
		return
	}
	field.SetString(value)
}

func getOptionalStringField(target any, fieldName string) string {
	return extractNamedStringField(reflect.ValueOf(target), fieldName)
}

func setReuseReservationRef(target any, refName string) {
	if refName == "" {
		return
	}
	assignReuseReservationRef(reflect.ValueOf(target), refName)
}

func assignReuseReservationRef(v reflect.Value, refName string) bool {
	if !v.IsValid() {
		return false
	}
	if v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return false
		}
		return assignReuseReservationRef(v.Elem(), refName)
	}
	if v.Kind() != reflect.Struct {
		return false
	}

	field := v.FieldByName("ReuseReservationRef")
	if field.IsValid() && field.CanSet() {
		if setReferenceField(field, refName) {
			return true
		}
	}

	if spec := v.FieldByName("Spec"); spec.IsValid() && spec.CanAddr() {
		if assignReuseReservationRef(spec.Addr(), refName) {
			return true
		}
	}

	return false
}

func setReferenceField(field reflect.Value, refName string) bool {
	switch field.Kind() {
	case reflect.String:
		field.SetString(refName)
		return true
	case reflect.Pointer:
		if field.Type().Elem().Kind() == reflect.String {
			value := reflect.New(field.Type().Elem())
			value.Elem().SetString(refName)
			field.Set(value)
			return true
		}
		value := reflect.New(field.Type().Elem())
		if setReferenceField(value.Elem(), refName) {
			field.Set(value)
			return true
		}
	case reflect.Struct:
		nameField := field.FieldByName("Name")
		if nameField.IsValid() && nameField.CanSet() && nameField.Kind() == reflect.String {
			nameField.SetString(refName)
			return true
		}
	}
	return false
}

// ExtractReservationRef returns the first reservation reference found on a
// Grove object, preferring status fields over spec/template fields.
func ExtractReservationRef(target any) string {
	v := reflect.ValueOf(target)
	for _, container := range []string{"Status", "Spec", "Template"} {
		if ref := extractNamedStringField(v, container, "ReservationRef"); ref != "" {
			return ref
		}
	}
	for _, fieldName := range []string{"ReservationRef", "ReuseReservationRef"} {
		if ref := extractNamedStringField(v, fieldName); ref != "" {
			return ref
		}
	}
	return ""
}

func extractNamedStringField(v reflect.Value, path ...string) string {
	if !v.IsValid() || len(path) == 0 {
		return ""
	}
	for v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return ""
		}
		v = v.Elem()
	}
	if !v.IsValid() || v.Kind() != reflect.Struct {
		return ""
	}

	field := v.FieldByName(path[0])
	if !field.IsValid() {
		return ""
	}
	if len(path) == 1 {
		return referenceFieldValue(field)
	}
	return extractNamedStringField(field, path[1:]...)
}

func referenceFieldValue(field reflect.Value) string {
	if !field.IsValid() {
		return ""
	}
	for field.Kind() == reflect.Pointer {
		if field.IsNil() {
			return ""
		}
		field = field.Elem()
	}
	if !field.IsValid() {
		return ""
	}
	switch field.Kind() {
	case reflect.String:
		return strings.TrimSpace(field.String())
	case reflect.Struct:
		if nameField := field.FieldByName("Name"); nameField.IsValid() {
			return referenceFieldValue(nameField)
		}
	}
	return ""
}

// resolveKaiSchedulerQueueName extracts the queue name from annotations or returns default
// This is the shared logic between DetermineKaiSchedulerQueue and ResolveKaiSchedulerQueue
func resolveKaiSchedulerQueueName(annotations map[string]string) string {
	queueName := commonconsts.DefaultKaiSchedulerQueue
	if annotations != nil {
		if annotationQueue, exists := annotations[commonconsts.KubeAnnotationKaiSchedulerQueue]; exists && strings.TrimSpace(annotationQueue) != "" {
			queueName = strings.TrimSpace(annotationQueue)
		}
	}
	return queueName
}

// ensureQueueExists validates that a Queue resource with the given name exists in the cluster
// Returns an error if the queue doesn't exist or if validation fails
func ensureQueueExists(ctx context.Context, dynamicClient dynamic.Interface, queueName string) error {
	logger := log.FromContext(ctx)

	// Try to get the queue resource using the predefined GVR
	_, err := dynamicClient.Resource(commonconsts.QueueGVR).Get(ctx, queueName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Error(err, "Queue not found", "queueName", queueName)
			return fmt.Errorf("queue '%s' not found in cluster. Ensure the queue exists before using kai-scheduler", queueName)
		}
		logger.Error(err, "Failed to validate queue", "queueName", queueName)
		return fmt.Errorf("failed to validate queue '%s': %w", queueName, err)
	}

	logger.Info("Queue validation successful", "queueName", queueName)
	return nil
}

// DetermineKaiSchedulerQueue determines the queue name for kai-scheduler from deployment annotations or returns default
// Also validates that the queue exists in the cluster
func DetermineKaiSchedulerQueue(ctx context.Context, annotations map[string]string) (string, error) {
	// Get the queue name from annotation or use default
	queueName := resolveKaiSchedulerQueueName(annotations)

	// Create a dynamic client for CRD validation (Queue CRD might not be in the standard client scheme)
	cfg, err := ctrl.GetConfig()
	if err != nil {
		return "", fmt.Errorf("failed to get kubernetes config for queue validation: %w", err)
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to create dynamic client for queue validation: %w", err)
	}

	// Validate that the queue exists
	if err := ensureQueueExists(ctx, dynamicClient, queueName); err != nil {
		return "", fmt.Errorf("kai-scheduler queue validation failed: %w", err)
	}

	return queueName, nil
}

// ResolveKaiSchedulerQueue determines the queue name for kai-scheduler from deployment annotations or returns default
// Does NOT validate - use DetermineKaiSchedulerQueue for validation
func ResolveKaiSchedulerQueue(annotations map[string]string) string {
	return resolveKaiSchedulerQueueName(annotations)
}

// injectKaiSchedulerIfEnabled injects kai-scheduler settings into a clique if kai-scheduler is enabled and grove is enabled
func injectKaiSchedulerIfEnabled(
	clique *grovev1alpha1.PodCliqueTemplateSpec,
	runtimeConfig *controller_common.RuntimeConfig,
	validatedQueueName string,
) {
	// Only proceed if grove is enabled, kai-scheduler is enabled, and no manual schedulerName is set
	if !runtimeConfig.GroveEnabled || !runtimeConfig.KaiSchedulerEnabled {
		return
	}

	// Check if user has manually set schedulerName - if so, respect their choice
	if clique.Spec.PodSpec.SchedulerName != "" && clique.Spec.PodSpec.SchedulerName != commonconsts.KaiSchedulerName {
		return
	}

	// Use the pre-validated queue name
	queueName := validatedQueueName

	// Inject schedulerName
	clique.Spec.PodSpec.SchedulerName = commonconsts.KaiSchedulerName

	// Inject queue label
	if clique.Labels == nil {
		clique.Labels = make(map[string]string)
	}
	clique.Labels[commonconsts.KubeLabelKaiSchedulerQueue] = queueName
}
