/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path"
	"strconv"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/aisimulate"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/yaml"
)

const (
	dgdrRunJobControllerUIDIndex = ".metadata.dgdrRunControllerUID"
	dgdrRunMaxJobNameLength      = 63

	dgdrRunInputKey                = "input.yaml"
	dgdrRunInputMountPath          = "/etc/aisimulate"
	dgdrRunInputPath               = dgdrRunInputMountPath + "/" + dgdrRunInputKey
	dgdrRunSnapshotPath            = "/var/run/aisimulate/output"
	dgdrRunServiceAccountMountPath = "/var/run/secrets/kubernetes.io/serviceaccount"
	dgdrRunPublisherBinary         = "/dgdr-run-publisher"
	dgdrRunSweeperContainer        = "sweeper"
	dgdrRunPublisherContainer      = "publisher"
	dgdrRunInputVolume             = "input"
	dgdrRunSnapshotVolume          = "snapshot"
	dgdrRunServiceAccountVolume    = "publisher-kube-api"
	dgdrRunGPUsPerNodeAnnotation   = "nvidia.com/dgdr-gpus-per-node"
	dgdrRunSnapshotPathAnnotation  = "nvidia.com/dgdr-snapshot-path"
	dgdrRunResourceComponent       = "aisimulate"
	dgdrRunServiceAccountTokenTTL  = int64(3600)
	dgdrRunReasonPending           = "JobPending"
	dgdrRunReasonRunning           = "JobRunning"
	dgdrRunReasonSucceeded         = "SearchSucceeded"
	dgdrRunReasonSearchFailed      = "SearchFailed"
	dgdrRunReasonProtocolViolation = "ProtocolViolation"
	dgdrRunReasonPublicationFailed = "PublicationFailed"
	dgdrRunReasonJobFailed         = "JobFailed"
	dgdrRunMessagePending          = "AISimulate Job is waiting to start"
	dgdrRunMessageRunning          = "AISimulate Job is running"
	dgdrRunMessageSucceeded        = "AISimulate search completed successfully"
)

type dgdrRunHardwareDiscoverer interface {
	Discover(context.Context, nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error)
}

// DynamoGraphDeploymentRunV1beta2Reconciler executes one immutable Run as an
// AISimulate Job. The publisher sidecar owns progress and candidate publication;
// this controller owns the Job reference, execution phase, timestamps and Completed.
type DynamoGraphDeploymentRunV1beta2Reconciler struct {
	client.Client
	Config                  *configv1alpha1.OperatorConfiguration
	RuntimeConfig           *commoncontroller.RuntimeConfig
	Hardware                dgdrRunHardwareDiscoverer
	PlannerImage            string
	OperatorImage           string
	OperatorImagePullPolicy corev1.PullPolicy
}

// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns,verbs=get;list;watch
// +kubebuilder:rbac:groups=nvidia.com,resources=dynamographdeploymentruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps;serviceaccounts,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update;patch

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	run := &nvidiacomv1beta2.DynamoGraphDeploymentRun{}
	if err := r.Get(ctx, req.NamespacedName, run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if run.DeletionTimestamp != nil || meta.IsStatusConditionTrue(run.Status.Conditions, nvidiacomv1beta2.DGDRRunConditionCompleted) {
		return ctrl.Result{}, nil
	}
	jobs, err := r.ownedJobs(ctx, run)
	if err != nil {
		return ctrl.Result{}, err
	}
	if len(jobs) > 1 {
		return ctrl.Result{}, fmt.Errorf("Run %q controls %d Jobs; expected at most one", run.Name, len(jobs))
	}

	var job *batchv1.Job
	if len(jobs) == 1 {
		job = &jobs[0]
	}
	resourceName := dgdrRunResourceName(run.Name)
	if err := r.reconcilePublisherAccess(ctx, run, resourceName); err != nil {
		return ctrl.Result{}, err
	}
	execution, err := r.reconcileInput(ctx, run, job, resourceName)
	if err != nil {
		return ctrl.Result{}, err
	}

	if job == nil {
		job = r.buildJob(run, resourceName, execution)
		if err := controllerutil.SetControllerReference(run, job, r.Scheme()); err != nil {
			return ctrl.Result{}, fmt.Errorf("set Job owner: %w", err)
		}
		if err := r.Create(ctx, job); err != nil {
			// A cache miss followed by AlreadyExists is expected to heal on the next
			// reconciliation once the informer observes the Job. Never bypass the cache.
			return ctrl.Result{}, fmt.Errorf("create Job %q: %w", job.Name, err)
		}
	}

	return ctrl.Result{}, r.reconcileRunStatus(ctx, run, job, execution)
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) ownedJobs(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) ([]batchv1.Job, error) {
	list := &batchv1.JobList{}
	if err := r.List(ctx, list, client.InNamespace(run.Namespace),
		client.MatchingFields{dgdrRunJobControllerUIDIndex: string(run.UID)}); err != nil {
		return nil, fmt.Errorf("list Jobs of Run %q: %w", run.Name, err)
	}
	owned := make([]batchv1.Job, 0, len(list.Items))
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], run) {
			owned = append(owned, list.Items[i])
		}
	}
	return owned, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcilePublisherAccess(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	name string,
) error {
	sa := &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: run.Namespace}}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, sa, func() error {
		return controllerutil.SetControllerReference(run, sa, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile ServiceAccount %q: %w", name, err)
	}

	role := &rbacv1.Role{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: run.Namespace}}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, role, func() error {
		role.Rules = dgdrRunPublisherRules(run.Name)
		return controllerutil.SetControllerReference(run, role, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile Role %q: %w", name, err)
	}

	binding := &rbacv1.RoleBinding{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: run.Namespace}}
	if _, err := controllerutil.CreateOrUpdate(ctx, r.Client, binding, func() error {
		binding.RoleRef = rbacv1.RoleRef{APIGroup: rbacv1.GroupName, Kind: "Role", Name: name}
		binding.Subjects = []rbacv1.Subject{{Kind: "ServiceAccount", Name: name, Namespace: run.Namespace}}
		return controllerutil.SetControllerReference(run, binding, r.Scheme())
	}); err != nil {
		return fmt.Errorf("reconcile RoleBinding %q: %w", name, err)
	}
	return nil
}

func dgdrRunPublisherRules(runName string) []rbacv1.PolicyRule {
	return []rbacv1.PolicyRule{
		{
			APIGroups:     []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources:     []string{"dynamographdeploymentruns"},
			ResourceNames: []string{runName},
			Verbs:         []string{"get", "list", "watch"},
		},
		{
			APIGroups:     []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources:     []string{"dynamographdeploymentruns/status"},
			ResourceNames: []string{runName},
			Verbs:         []string{"get", "update", "patch"},
		},
		{
			APIGroups: []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources: []string{"dynamographdeploymentcandidates"},
			Verbs:     []string{"get", "list", "watch", "create", "patch", "delete"},
		},
		{
			APIGroups: []string{nvidiacomv1beta2.GroupVersion.Group},
			Resources: []string{"dynamographdeploymentcandidates/status"},
			Verbs:     []string{"get", "update", "patch"},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list", "watch"},
		},
	}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcileInput(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
	name string,
) (dgdrRunExecution, error) {
	configMap := &corev1.ConfigMap{}
	key := types.NamespacedName{Namespace: run.Namespace, Name: name}
	err := r.Get(ctx, key, configMap)
	if err == nil {
		if !metav1.IsControlledBy(configMap, run) {
			return dgdrRunExecution{}, fmt.Errorf("ConfigMap %q is not controlled by Run %q", name, run.Name)
		}
		gpusPerNode, parseErr := positiveAnnotation(configMap.Annotations, dgdrRunGPUsPerNodeAnnotation)
		if parseErr != nil {
			return dgdrRunExecution{}, fmt.Errorf("read ConfigMap %q: %w", name, parseErr)
		}
		input, inputErr := dgdrRunStatusInput(configMap.Data[dgdrRunInputKey])
		if inputErr != nil {
			return dgdrRunExecution{}, fmt.Errorf("read ConfigMap %q input: %w", name, inputErr)
		}
		execution := dgdrRunExecution{
			GPUsPerNode:  gpusPerNode,
			SnapshotPath: annotationOrDefault(configMap.Annotations, dgdrRunSnapshotPathAnnotation, dgdrRunSnapshotPath),
			Input:        input,
		}
		if job != nil {
			jobGPUs, jobErr := positiveAnnotation(job.Annotations, dgdrRunGPUsPerNodeAnnotation)
			if jobErr != nil {
				return dgdrRunExecution{}, fmt.Errorf("read Job %q: %w", job.Name, jobErr)
			}
			if jobGPUs != gpusPerNode {
				return dgdrRunExecution{}, fmt.Errorf("Job %q and ConfigMap %q disagree about GPUs per node", job.Name, name)
			}
			jobSnapshotPath := annotationOrDefault(job.Annotations, dgdrRunSnapshotPathAnnotation, dgdrRunSnapshotPath)
			if jobSnapshotPath != execution.SnapshotPath {
				return dgdrRunExecution{}, fmt.Errorf("Job %q and ConfigMap %q disagree about the snapshot path", job.Name, name)
			}
		}
		return execution, nil
	}
	if !apierrors.IsNotFound(err) {
		return dgdrRunExecution{}, fmt.Errorf("get ConfigMap %q: %w", name, err)
	}

	execution := dgdrRunExecution{SnapshotPath: dgdrRunSnapshotPath}
	if job != nil {
		execution.GPUsPerNode, err = positiveAnnotation(job.Annotations, dgdrRunGPUsPerNodeAnnotation)
		if err != nil {
			return dgdrRunExecution{}, fmt.Errorf("read Job %q: %w", job.Name, err)
		}
		execution.SnapshotPath = annotationOrDefault(job.Annotations, dgdrRunSnapshotPathAnnotation, dgdrRunSnapshotPath)
	}
	if execution.GPUsPerNode == 0 {
		execution.GPUsPerNode, err = r.discoverGPUsPerNode(ctx, run)
		if err != nil {
			return dgdrRunExecution{}, err
		}
	}
	input, snapshotPath, err := dgdrRunInput(run, r.PlannerImage, execution.GPUsPerNode)
	if err != nil {
		return dgdrRunExecution{}, err
	}
	contents, err := yaml.Marshal(input)
	if err != nil {
		return dgdrRunExecution{}, fmt.Errorf("encode AISimulate input: %w", err)
	}
	execution.Input, err = dgdrRunStatusInput(string(contents))
	if err != nil {
		return dgdrRunExecution{}, err
	}
	if job == nil {
		execution.SnapshotPath = snapshotPath
	}

	configMap = &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: run.Namespace,
			Annotations: map[string]string{
				dgdrRunGPUsPerNodeAnnotation:  strconv.Itoa(execution.GPUsPerNode),
				dgdrRunSnapshotPathAnnotation: execution.SnapshotPath,
			},
		},
		Immutable: ptr.To(true),
		Data:      map[string]string{dgdrRunInputKey: string(contents)},
	}
	if err := controllerutil.SetControllerReference(run, configMap, r.Scheme()); err != nil {
		return dgdrRunExecution{}, fmt.Errorf("set ConfigMap owner: %w", err)
	}
	if err := r.Create(ctx, configMap); err != nil {
		return dgdrRunExecution{}, fmt.Errorf("create ConfigMap %q: %w", name, err)
	}
	return execution, nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) discoverGPUsPerNode(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
) (int, error) {
	if r.Hardware == nil {
		return 0, errors.New("discover GPUs per node: hardware discoverer is not configured")
	}
	if run.Spec.Hardware == nil || run.Spec.Hardware.GPU == nil || run.Spec.Hardware.GPU.SKU == nil {
		return 0, errors.New("discover GPUs per node: spec.hardware.gpu.sku was not resolved by Request admission")
	}
	info, err := r.Hardware.Discover(ctx, nvidiacomv1beta1.GPUSKUType(*run.Spec.Hardware.GPU.SKU))
	if err != nil {
		return 0, fmt.Errorf("discover GPUs per node: %w", err)
	}
	if info == nil || info.GPUsPerNode < 1 {
		return 0, errors.New("discover GPUs per node: discovery returned no positive GPU count")
	}
	return info.GPUsPerNode, nil
}

func dgdrRunInput(
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	plannerImage string,
	gpusPerNode int,
) (map[string]any, string, error) {
	if run.Spec.Recommendation == nil || run.Spec.Recommendation.MaxCandidates < 1 {
		return nil, "", errors.New("spec.recommendation.maxCandidates must be defaulted before the Run is created")
	}
	underlay := map[string]any{"dgdr_run": map[string]any{
		"name":                  run.Name,
		"snapshot_dir":          dgdrRunSnapshotPath,
		"namespace":             run.Namespace,
		"runtime_image":         plannerImage,
		"derive_runtime_images": true,
		"num_gpus_per_node":     int64(gpusPerNode),
		"max_candidates":        int64(run.Spec.Recommendation.MaxCandidates),
	}}
	input, err := aisimulate.LowerWithUnderlay(&run.Spec.NativeSearchInput, underlay)
	if err != nil {
		return nil, "", err
	}
	snapshotPath := dgdrRunSnapshotPath
	if configured, found, err := unstructured.NestedString(input, "dgdr_run", "snapshot_dir"); err == nil && found && path.IsAbs(configured) {
		snapshotPath = path.Clean(configured)
	}
	return input, snapshotPath, nil
}

type dgdrRunExecution struct {
	GPUsPerNode  int
	SnapshotPath string
	Input        *runtime.RawExtension
}

func dgdrRunStatusInput(contents string) (*runtime.RawExtension, error) {
	var input map[string]any
	if err := yaml.Unmarshal([]byte(contents), &input); err != nil {
		return nil, fmt.Errorf("decode AISimulate input: %w", err)
	}
	if input == nil {
		return nil, errors.New("decode AISimulate input: expected an object")
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode AISimulate status input: %w", err)
	}
	return &runtime.RawExtension{Raw: raw}, nil
}

func positiveAnnotation(annotations map[string]string, key string) (int, error) {
	value := annotations[key]
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < 1 {
		return 0, fmt.Errorf("annotation %s=%q is not a positive integer", key, value)
	}
	return parsed, nil
}

func annotationOrDefault(annotations map[string]string, key, defaultValue string) string {
	if value := annotations[key]; value != "" {
		return value
	}
	return defaultValue
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) buildJob(
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	name string,
	execution dgdrRunExecution,
) *batchv1.Job {
	pullPolicy := r.OperatorImagePullPolicy
	if pullPolicy == "" {
		pullPolicy = corev1.PullIfNotPresent
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: run.Namespace,
			Annotations: map[string]string{
				dgdrRunGPUsPerNodeAnnotation:  strconv.Itoa(execution.GPUsPerNode),
				dgdrRunSnapshotPathAnnotation: execution.SnapshotPath,
			},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: ptr.To[int32](0),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					ServiceAccountName:           name,
					AutomountServiceAccountToken: ptr.To(false),
					RestartPolicy:                corev1.RestartPolicyNever,
					SecurityContext:              &corev1.PodSecurityContext{FSGroup: ptr.To[int64](65532)},
					Containers: []corev1.Container{
						{
							Name:            dgdrRunSweeperContainer,
							Image:           r.PlannerImage,
							ImagePullPolicy: corev1.PullIfNotPresent,
							Command:         []string{"aisimulate"},
							Args: []string{
								"recommend", "--stack", "dynamo", "--config", dgdrRunInputPath,
								"--output", "dgdr_run", "--output-dir", execution.SnapshotPath,
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: dgdrRunInputVolume, MountPath: dgdrRunInputMountPath, ReadOnly: true},
								{Name: dgdrRunSnapshotVolume, MountPath: execution.SnapshotPath},
							},
						},
						{
							Name:            dgdrRunPublisherContainer,
							Image:           r.OperatorImage,
							ImagePullPolicy: pullPolicy,
							Command:         []string{dgdrRunPublisherBinary},
							Args: []string{
								"--snapshot-dir", execution.SnapshotPath,
								"--run-name", run.Name,
								"--run-uid", string(run.UID),
								"--sweeper-container", dgdrRunSweeperContainer,
							},
							Env: []corev1.EnvVar{
								{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
								{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
							},
							VolumeMounts: []corev1.VolumeMount{
								{Name: dgdrRunSnapshotVolume, MountPath: execution.SnapshotPath},
								{Name: dgdrRunServiceAccountVolume, MountPath: dgdrRunServiceAccountMountPath, ReadOnly: true},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: dgdrRunInputVolume,
							VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: name},
							}},
						},
						{Name: dgdrRunSnapshotVolume, VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
						dgdrRunPublisherKubeAPIVolume(),
					},
				},
			},
		},
	}
	return job
}

func dgdrRunPublisherKubeAPIVolume() corev1.Volume {
	return corev1.Volume{
		Name: dgdrRunServiceAccountVolume,
		VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{
			Sources: []corev1.VolumeProjection{
				{ServiceAccountToken: &corev1.ServiceAccountTokenProjection{
					Path: "token", ExpirationSeconds: ptr.To(dgdrRunServiceAccountTokenTTL),
				}},
				{ConfigMap: &corev1.ConfigMapProjection{
					LocalObjectReference: corev1.LocalObjectReference{Name: ConfigMapNameKubeRootCA},
					Items:                []corev1.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}},
				}},
				{DownwardAPI: &corev1.DownwardAPIProjection{Items: []corev1.DownwardAPIVolumeFile{{
					Path: "namespace", FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.namespace"},
				}}}},
			},
		}},
	}
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) reconcileRunStatus(
	ctx context.Context,
	run *nvidiacomv1beta2.DynamoGraphDeploymentRun,
	job *batchv1.Job,
	execution dgdrRunExecution,
) error {
	if execution.Input == nil {
		return errors.New("update Run status: effective AISimulate input is missing")
	}
	original := run.DeepCopy()
	run.Status.AISimulate = &nvidiacomv1beta2.AISimulateRunStatus{Input: execution.Input.DeepCopy()}
	run.Status.JobRef = &nvidiacomv1beta2.ResourceReference{Name: job.Name, UID: job.UID}
	run.Status.Phase = nvidiacomv1beta2.DGDRRunPhasePending
	reason, message := dgdrRunReasonPending, dgdrRunMessagePending
	completed := metav1.ConditionFalse

	if job.Status.StartTime != nil {
		run.Status.StartTime = job.Status.StartTime.DeepCopy()
	}
	if condition := dgdrRunJobCondition(job, batchv1.JobComplete); condition != nil && condition.Status == corev1.ConditionTrue {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseSucceeded
		reason, message = dgdrRunReasonSucceeded, dgdrRunMessageSucceeded
		completed = metav1.ConditionTrue
		run.Status.CompletionTime = dgdrRunCompletionTime(job)
	} else if condition := dgdrRunJobCondition(job, batchv1.JobFailed); condition != nil && condition.Status == corev1.ConditionTrue {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseFailed
		reason, message = r.dgdrRunFailure(ctx, job, condition)
		completed = metav1.ConditionTrue
		run.Status.CompletionTime = dgdrRunCompletionTime(job)
	} else if job.Status.Active > 0 || job.Status.StartTime != nil {
		run.Status.Phase = nvidiacomv1beta2.DGDRRunPhaseRunning
		reason, message = dgdrRunReasonRunning, dgdrRunMessageRunning
	}

	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               nvidiacomv1beta2.DGDRRunConditionCompleted,
		Status:             completed,
		Reason:             reason,
		Message:            message,
		ObservedGeneration: run.Generation,
	})
	if apiequality.Semantic.DeepEqual(original.Status, run.Status) {
		return nil
	}
	if err := r.Status().Patch(ctx, run, client.MergeFrom(original)); err != nil {
		return fmt.Errorf("update Run %q status: %w", run.Name, err)
	}
	return nil
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) dgdrRunFailure(
	ctx context.Context,
	job *batchv1.Job,
	condition *batchv1.JobCondition,
) (string, string) {
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(job.Namespace), client.MatchingLabels{batchv1.JobNameLabel: job.Name}); err != nil {
		log.FromContext(ctx).Error(err, "Could not inspect failed AISimulate Job pods", "job", job.Name)
		return dgdrRunReasonJobFailed, condition.Message
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !metav1.IsControlledBy(pod, job) {
			continue
		}
		for _, status := range pod.Status.ContainerStatuses {
			terminated := status.State.Terminated
			if terminated == nil || terminated.ExitCode == 0 {
				continue
			}
			message := terminated.Message
			if message == "" {
				message = terminated.Reason
			}
			if message == "" {
				message = fmt.Sprintf("container %s exited with code %d", status.Name, terminated.ExitCode)
			}
			if status.Name == dgdrRunPublisherContainer {
				switch int(terminated.ExitCode) {
				case dgdrrunpublisher.ExitRunFailed:
					return dgdrRunReasonSearchFailed, message
				case dgdrrunpublisher.ExitProtocolViolation:
					return dgdrRunReasonProtocolViolation, message
				case dgdrrunpublisher.ExitReconcileFailed:
					return dgdrRunReasonPublicationFailed, message
				}
			}
			if status.Name == dgdrRunSweeperContainer {
				return dgdrRunReasonSearchFailed, message
			}
		}
	}
	message := condition.Message
	if message == "" {
		message = "AISimulate Job failed"
	}
	return dgdrRunReasonJobFailed, message
}

func dgdrRunCompletionTime(job *batchv1.Job) *metav1.Time {
	if job.Status.CompletionTime != nil {
		return job.Status.CompletionTime.DeepCopy()
	}
	for _, condition := range job.Status.Conditions {
		if (condition.Type == batchv1.JobComplete || condition.Type == batchv1.JobFailed) && condition.Status == corev1.ConditionTrue {
			return condition.LastTransitionTime.DeepCopy()
		}
	}
	return nil
}

func dgdrRunJobCondition(job *batchv1.Job, conditionType batchv1.JobConditionType) *batchv1.JobCondition {
	for i := range job.Status.Conditions {
		if job.Status.Conditions[i].Type == conditionType {
			return &job.Status.Conditions[i]
		}
	}
	return nil
}

func dgdrRunResourceName(runName string) string {
	suffix := "-" + dgdrRunResourceComponent
	if len(runName)+len(suffix) <= dgdrRunMaxJobNameLength {
		return runName + suffix
	}
	sum := sha256.Sum256([]byte(runName))
	hash := hex.EncodeToString(sum[:6])
	room := dgdrRunMaxJobNameLength - len(suffix) - len(hash) - 1
	prefix := strings.TrimRight(runName[:room], "-.")
	return prefix + "-" + hash + suffix
}

func (r *DynamoGraphDeploymentRunV1beta2Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.PlannerImage == "" {
		return errors.New("planner image is required for the DGDR Run sweeper")
	}
	if r.OperatorImage == "" {
		return errors.New("operator image is required for the DGDR Run publisher")
	}
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(), &batchv1.Job{}, dgdrRunJobControllerUIDIndex, dgdrRunJobControllerUIDs,
	); err != nil {
		return fmt.Errorf("index Jobs by Run controller UID: %w", err)
	}
	return ctrl.NewControllerManagedBy(mgr).
		Named("dynamographdeploymentrun-v1beta2").
		For(&nvidiacomv1beta2.DynamoGraphDeploymentRun{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&rbacv1.Role{}).
		Owns(&rbacv1.RoleBinding{}).
		WithEventFilter(commoncontroller.EphemeralDeploymentEventFilter(r.Config, r.RuntimeConfig)).
		Complete(r)
}

func dgdrRunJobControllerUIDs(obj client.Object) []string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil || owner.Kind != "DynamoGraphDeploymentRun" || owner.APIVersion != nvidiacomv1beta2.GroupVersion.String() {
		return nil
	}
	return []string{string(owner.UID)}
}
