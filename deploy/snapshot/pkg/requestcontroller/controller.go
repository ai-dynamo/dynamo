package requestcontroller

import (
	"context"
	"fmt"
	"strings"
	"time"

	snapshotv1alpha1 "github.com/ai-dynamo/dynamo/deploy/snapshot/api/v1alpha1"
	snapshotkube "github.com/ai-dynamo/dynamo/deploy/snapshot/pkg/kube"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

const requeueDelay = 2 * time.Second

type Reconciler struct {
	client.Client
	Scheme                       *runtime.Scheme
	DisableCudaCheckpointJobFile bool
}

// +kubebuilder:rbac:groups=snapshot.nvidia.com,resources=snapshotrequests,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=snapshot.nvidia.com,resources=snapshotrequests/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=snapshot.nvidia.com,resources=snapshotrequests/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;update;patch;delete

func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	request := &snapshotv1alpha1.SnapshotRequest{}
	if err := r.Get(ctx, req.NamespacedName, request); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if request.Status.ObservedGeneration == request.Generation && isTerminalState(request.Status.State) {
		return ctrl.Result{}, nil
	}

	storage, err := snapshotkube.DiscoverSnapshotStorage(ctx, r.Client, request.Namespace)
	if err != nil {
		return ctrl.Result{}, r.setFailed(ctx, request, err.Error())
	}

	switch request.Spec.Phase {
	case snapshotv1alpha1.SnapshotRequestPhaseCheckpoint:
		return r.reconcileCheckpoint(ctx, request, storage)
	case snapshotv1alpha1.SnapshotRequestPhaseRestore:
		return r.reconcileRestore(ctx, request, storage)
	default:
		logger.Info("Rejecting SnapshotRequest with unknown phase", "name", request.Name, "phase", request.Spec.Phase)
		return ctrl.Result{}, r.setFailed(ctx, request, fmt.Sprintf("unknown phase %q", request.Spec.Phase))
	}
}

func (r *Reconciler) reconcileCheckpoint(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage) (ctrl.Result, error) {
	if request.Spec.PodTemplate == nil {
		return ctrl.Result{}, r.setFailed(ctx, request, "checkpoint requests require spec.podTemplate")
	}

	jobName := childName(request.Name, "checkpoint")
	location := storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
	job := &batchv1.Job{}
	err := r.Get(ctx, client.ObjectKey{Namespace: request.Namespace, Name: jobName}, job)
	if apierrors.IsNotFound(err) {
		job, err = r.buildCheckpointJob(request, storage, jobName)
		if err != nil {
			return ctrl.Result{}, r.setFailed(ctx, request, err.Error())
		}
		if err := controllerutil.SetControllerReference(request, job, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Create(ctx, job); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: requeueDelay}, r.setRunning(ctx, request, func(status *snapshotv1alpha1.SnapshotRequestStatus) {
			status.JobName = jobName
			status.Location = location
			status.StorageType = snapshotkube.StorageTypePVC
			status.Message = "checkpoint job created"
		})
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	status := strings.TrimSpace(job.Annotations[snapshotkube.CheckpointStatusAnnotation])
	if status == "failed" {
		return ctrl.Result{}, r.setFailed(ctx, request, "snapshot-agent reported checkpoint failure")
	}
	if jobCompleted(job) {
		if status == "completed" {
			return ctrl.Result{}, r.setSucceeded(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
				reqStatus.JobName = jobName
				reqStatus.Location = location
				reqStatus.StorageType = snapshotkube.StorageTypePVC
				reqStatus.Message = ""
			})
		}
		return ctrl.Result{RequeueAfter: requeueDelay}, r.setRunning(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
			reqStatus.JobName = jobName
			reqStatus.Location = location
			reqStatus.StorageType = snapshotkube.StorageTypePVC
			reqStatus.Message = "waiting for snapshot-agent checkpoint completion"
		})
	}
	if jobFailed(job) {
		return ctrl.Result{}, r.setFailed(ctx, request, "checkpoint job failed")
	}

	return ctrl.Result{RequeueAfter: requeueDelay}, r.setRunning(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
		reqStatus.JobName = jobName
		reqStatus.Location = location
		reqStatus.StorageType = snapshotkube.StorageTypePVC
		reqStatus.Message = "checkpoint job running"
	})
}

func (r *Reconciler) reconcileRestore(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage) (ctrl.Result, error) {
	if request.Spec.TargetPodRef != nil {
		return r.reconcileTargetRestore(ctx, request, storage)
	}
	if request.Spec.PodTemplate != nil {
		return r.reconcileStandaloneRestore(ctx, request, storage)
	}
	return ctrl.Result{}, r.setFailed(ctx, request, "restore requests require spec.targetPodRef or spec.podTemplate")
}

func (r *Reconciler) reconcileStandaloneRestore(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage) (ctrl.Result, error) {
	podName := childName(request.Name, "restore")
	pod := &corev1.Pod{}
	err := r.Get(ctx, client.ObjectKey{Namespace: request.Namespace, Name: podName}, pod)
	if apierrors.IsNotFound(err) {
		pod, err = r.buildRestorePod(request, storage, podName)
		if err != nil {
			return ctrl.Result{}, r.setFailed(ctx, request, err.Error())
		}
		if err := controllerutil.SetControllerReference(request, pod, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.Create(ctx, pod); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: requeueDelay}, r.setWaiting(ctx, request, func(status *snapshotv1alpha1.SnapshotRequestStatus) {
			status.PodName = podName
			status.Location = storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
			status.StorageType = snapshotkube.StorageTypePVC
			status.Message = "restore pod created"
		})
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	return r.observeRestorePod(ctx, request, pod, storage)
}

func (r *Reconciler) reconcileTargetRestore(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage) (ctrl.Result, error) {
	podName := request.Spec.TargetPodRef.Name
	pod := &corev1.Pod{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: request.Namespace, Name: podName}, pod); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{RequeueAfter: requeueDelay}, r.setWaiting(ctx, request, func(status *snapshotv1alpha1.SnapshotRequestStatus) {
				status.PodName = podName
				status.Location = storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
				status.StorageType = snapshotkube.StorageTypePVC
				status.Message = "waiting for target pod"
			})
		}
		return ctrl.Result{}, err
	}
	if err := snapshotkube.ValidateRestoreTargetPod(pod, storage, snapshotkube.DefaultSeccompLocalhostProfile); err != nil {
		return ctrl.Result{}, r.setFailed(ctx, request, fmt.Sprintf("target pod %s is not prepared for snapshot restore: %v", podName, err))
	}

	desiredLabels := mapsClone(pod.Labels)
	desiredAnnotations := mapsClone(pod.Annotations)
	snapshotkube.ApplyRestoreTargetMetadata(
		desiredLabels,
		desiredAnnotations,
		true,
		request.Spec.SnapshotID,
		storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion),
		snapshotkube.StorageTypePVC,
	)
	if !mapsEqual(pod.Labels, desiredLabels) || !mapsEqual(pod.Annotations, desiredAnnotations) {
		pod = pod.DeepCopy()
		pod.Labels = desiredLabels
		pod.Annotations = desiredAnnotations
		if err := r.Update(ctx, pod); err != nil {
			return ctrl.Result{}, err
		}
	}

	return r.observeRestorePod(ctx, request, pod, storage)
}

func (r *Reconciler) observeRestorePod(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, pod *corev1.Pod, storage snapshotkube.SnapshotStorage) (ctrl.Result, error) {
	status := strings.TrimSpace(pod.Annotations[snapshotkube.RestoreStatusAnnotation])
	if status == "completed" {
		return ctrl.Result{}, r.setSucceeded(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
			reqStatus.PodName = pod.Name
			reqStatus.Location = storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
			reqStatus.StorageType = snapshotkube.StorageTypePVC
			reqStatus.Message = ""
		})
	}
	if status == "failed" || pod.Status.Phase == corev1.PodFailed {
		return ctrl.Result{}, r.setFailed(ctx, request, "restore failed")
	}
	if pod.Status.Phase == corev1.PodPending {
		return ctrl.Result{RequeueAfter: requeueDelay}, r.setWaiting(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
			reqStatus.PodName = pod.Name
			reqStatus.Location = storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
			reqStatus.StorageType = snapshotkube.StorageTypePVC
			reqStatus.Message = "waiting for restore pod"
		})
	}
	return ctrl.Result{RequeueAfter: requeueDelay}, r.setRunning(ctx, request, func(reqStatus *snapshotv1alpha1.SnapshotRequestStatus) {
		reqStatus.PodName = pod.Name
		reqStatus.Location = storage.CheckpointLocation(request.Spec.SnapshotID, request.Spec.ArtifactVersion)
		reqStatus.StorageType = snapshotkube.StorageTypePVC
		reqStatus.Message = "restore running"
	})
}

func (r *Reconciler) buildCheckpointJob(request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage, jobName string) (*batchv1.Job, error) {
	podTemplate := request.Spec.PodTemplate.DeepCopy()
	if err := snapshotkube.PrepareCheckpointPodTemplate(
		podTemplate,
		request.Spec.SnapshotID,
		request.Spec.ArtifactVersion,
		storage,
		r.DisableCudaCheckpointJobFile,
		snapshotkube.DefaultSeccompLocalhostProfile,
	); err != nil {
		return nil, err
	}

	activeDeadlineSeconds := request.Spec.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		activeDeadlineSeconds = ptr.To[int64](3600)
	}
	ttlSeconds := request.Spec.TTLSecondsAfterFinished
	if ttlSeconds == nil {
		ttlSeconds = ptr.To[int32](300)
	}

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: request.Namespace,
		},
		Spec: batchv1.JobSpec{
			ActiveDeadlineSeconds:   activeDeadlineSeconds,
			BackoffLimit:            ptr.To[int32](0),
			TTLSecondsAfterFinished: ttlSeconds,
			Template:                *podTemplate,
		},
	}, nil
}

func (r *Reconciler) buildRestorePod(request *snapshotv1alpha1.SnapshotRequest, storage snapshotkube.SnapshotStorage, podName string) (*corev1.Pod, error) {
	podTemplate := request.Spec.PodTemplate.DeepCopy()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   request.Namespace,
			Labels:      mapsClone(podTemplate.Labels),
			Annotations: mapsClone(podTemplate.Annotations),
		},
		Spec: *podTemplate.Spec.DeepCopy(),
	}
	if err := snapshotkube.PrepareStandaloneRestorePod(
		pod,
		request.Spec.SnapshotID,
		request.Spec.ArtifactVersion,
		storage,
		snapshotkube.DefaultSeccompLocalhostProfile,
	); err != nil {
		return nil, err
	}
	return pod, nil
}

func (r *Reconciler) setWaiting(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, mutate func(*snapshotv1alpha1.SnapshotRequestStatus)) error {
	return r.updateStatus(ctx, request, snapshotv1alpha1.SnapshotRequestStateWaitingForTarget, mutate)
}

func (r *Reconciler) setRunning(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, mutate func(*snapshotv1alpha1.SnapshotRequestStatus)) error {
	return r.updateStatus(ctx, request, snapshotv1alpha1.SnapshotRequestStateRunning, mutate)
}

func (r *Reconciler) setSucceeded(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, mutate func(*snapshotv1alpha1.SnapshotRequestStatus)) error {
	return r.updateStatus(ctx, request, snapshotv1alpha1.SnapshotRequestStateSucceeded, func(status *snapshotv1alpha1.SnapshotRequestStatus) {
		if status.CompletedAt == nil {
			status.CompletedAt = snapshotv1alpha1.Now()
		}
		mutate(status)
	})
}

func (r *Reconciler) setFailed(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, message string) error {
	return r.updateStatus(ctx, request, snapshotv1alpha1.SnapshotRequestStateFailed, func(status *snapshotv1alpha1.SnapshotRequestStatus) {
		status.Message = message
		if status.CompletedAt == nil {
			status.CompletedAt = snapshotv1alpha1.Now()
		}
	})
}

func (r *Reconciler) updateStatus(ctx context.Context, request *snapshotv1alpha1.SnapshotRequest, state snapshotv1alpha1.SnapshotRequestState, mutate func(*snapshotv1alpha1.SnapshotRequestStatus)) error {
	current := request.DeepCopy()
	if current.Status.State == state && current.Status.ObservedGeneration == current.Generation {
		before := current.Status.DeepCopy()
		mutate(&current.Status)
		if before != nil && *before == current.Status {
			return nil
		}
	} else {
		current.Status.State = state
		current.Status.ObservedGeneration = current.Generation
		if current.Status.StartedAt == nil && state != snapshotv1alpha1.SnapshotRequestStatePending {
			current.Status.StartedAt = snapshotv1alpha1.Now()
		}
		mutate(&current.Status)
	}
	return r.Status().Patch(ctx, current, client.MergeFrom(request))
}

func childName(base string, suffix string) string {
	maxBaseLen := 63 - len(suffix) - 1
	if maxBaseLen < 1 {
		maxBaseLen = 1
	}
	if len(base) > maxBaseLen {
		base = base[:maxBaseLen]
	}
	return base + "-" + suffix
}

func jobCompleted(job *batchv1.Job) bool {
	for _, condition := range job.Status.Conditions {
		if condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func jobFailed(job *batchv1.Job) bool {
	for _, condition := range job.Status.Conditions {
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func mapsClone(in map[string]string) map[string]string {
	if in == nil {
		return map[string]string{}
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func mapsEqual(a map[string]string, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func isTerminalState(state snapshotv1alpha1.SnapshotRequestState) bool {
	return state == snapshotv1alpha1.SnapshotRequestStateSucceeded || state == snapshotv1alpha1.SnapshotRequestStateFailed
}

func (r *Reconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&snapshotv1alpha1.SnapshotRequest{}).
		Owns(&batchv1.Job{}, builder.WithPredicates(predicate.ResourceVersionChangedPredicate{})).
		Owns(&corev1.Pod{}, builder.WithPredicates(predicate.ResourceVersionChangedPredicate{})).
		Complete(r)
}
