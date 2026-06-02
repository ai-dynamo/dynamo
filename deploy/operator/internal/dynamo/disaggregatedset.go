package dynamo

import (
	"context"
	"fmt"
	"strings"

	disaggv1alpha1 "sigs.k8s.io/disaggregatedset/api/v1alpha1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// DSMultinodeDeployer implements the multinode pod-spec helpers for a
// DisaggregatedSet (DS) managed workload. A DS owns N LeaderWorkerSets, one
// per role, and the kubelet-injected LWS env vars (LWS_LEADER_ADDRESS,
// LWS_GROUP_SIZE, LWS_WORKER_INDEX) are still set on every pod, so the
// pod-spec generation logic is identical to the standalone-LWS case. DS
// differs from LWS only at the orchestrator layer (one DS per DGD with
// N role-shaped LWSes, plus an N-dimensional rolling update), not in how
// a single pod renders its hostname or rank.
//
// For that reason DSMultinodeDeployer embeds LWSMultinodeDeployer and
// inherits all of its methods. The type exists so that the rest of the
// operator can branch on the orchestrator without changing the pod-spec
// rendering pipeline.
type DSMultinodeDeployer struct {
	LWSMultinodeDeployer
}

// DSNameForDGD returns the name used for the DisaggregatedSet that backs
// a given DynamoGraphDeployment. We deliberately reuse the same naming
// scheme as the standalone-LWS path (DCD-name based) so a DGD that flips
// from LWS to DS (or back) does not leave orphan resources.
func DSNameForDGD(dgdName string) string {
	return fmt.Sprintf("%s-disagg", dgdName)
}

// DSRolesForDGD returns the role-name list for a DisaggregatedSet built
// from a DGD. The role name is what shows up in DS status.roleStatuses[].
// We use the service name (matching the existing DCD naming) so that
// observability tools that already group by service name keep working.
func DSRolesForDGD(components []v1beta1.DynamoComponentDeploymentSharedSpec) []string {
	roles := make([]string, 0, len(components))
	for i := range components {
		roles = append(roles, components[i].ComponentName)
	}
	return roles
}

// GenerateDisaggregatedSet builds the DisaggregatedSet resource for a
// given DynamoGraphDeployment. One role per spec.services entry; each
// role embeds the same LeaderWorkerSetSpec that the DCD controller would
// have generated for that service if it were using the standalone-LWS
// pathway. This is the DGD-level entry point used by the DGD controller's
// reconcileDisaggregatedSet method.
//
// The caller is responsible for the actual create/update via the
// controller-runtime client and for watching the resulting status.
func GenerateDisaggregatedSet(
	ctx context.Context,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
) (*disaggv1alpha1.DisaggregatedSet, error) {
	logger := log.FromContext(ctx)
	logger.Info("Generating DisaggregatedSet", "dgd", dynamoDeployment.Name)

	ds := &disaggv1alpha1.DisaggregatedSet{}
	ds.Name = DSNameForDGD(dynamoDeployment.Name)
	ds.Namespace = dynamoDeployment.Namespace
	ds.Labels = map[string]string{
		commonconsts.KubeLabelDynamoGraphDeploymentName: dynamoDeployment.Name,
	}

	roles := make([]disaggv1alpha1.DisaggregatedRoleSpec, 0, len(dynamoDeployment.Spec.Components))
	// Per-role LeaderWorkerSetSpec is built by the caller (DCD-level
	// generateLeaderWorkerSet) and spliced in by the controller. We leave
	// the Spec empty here; the controller fills it in after invoking the
	// DCD-level pod-spec generator for each service.
	for _, component := range dynamoDeployment.Spec.Components {
		role := disaggv1alpha1.DisaggregatedRoleSpec{
			Name: component.ComponentName,
		}
		// Replicas is taken from the service's DCD spec. If the DGD
		// does not pin replicas, default to 1 (matching the LWS path).
		replicas := int32(1)
		if component.Replicas != nil {
			replicas = *component.Replicas
		}
		role.Replicas = &replicas
		roles = append(roles, role)
	}
	ds.Spec.Roles = roles
	return ds, nil
}

// DisaggregatedSetRoleSpecForService builds the per-role LeaderWorkerSetSpec
// for one service inside a DGD. It mirrors the DCD-level
// generateLeaderWorkerSet, but returns just the spec (so the caller can
// embed it in a DS role) instead of a full LWS object. The leading
// LeaderTemplate and WorkerTemplate are produced by GeneratePodSpecForComponent
// using a DSMultinodeDeployer as the multinode deployer override.
//
// numberOfNodes is taken from the service's multinode.nodeCount; it
// becomes the LWS group size.
func DisaggregatedSetRoleSpecForService(
	ctx context.Context,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
	secretsRetriever SecretsRetriever,
	dynamoDeployment *v1beta1.DynamoGraphDeployment,
	operatorConfig *v1alpha1.OperatorConfiguration,
	checkpointInfo *checkpoint.CheckpointInfo,
) (leaderworkersetv1.LeaderWorkerSetSpec, error) {
	numberOfNodes := component.GetNumberOfNodes()
	deployer := &DSMultinodeDeployer{}

	leaderPodSpec, err := GeneratePodSpecForComponent(
		component, backendFramework, secretsRetriever,
		dynamoDeployment, RoleLeader, numberOfNodes,
		operatorConfig, commonconsts.MultinodeDeploymentTypeDisaggregatedSet,
		component.ComponentName, checkpointInfo, deployer,
	)
	if err != nil {
		return leaderworkersetv1.LeaderWorkerSetSpec{}, fmt.Errorf(
			"failed to build leader pod spec for role %q: %w", component.ComponentName, err)
	}
	workerPodSpec, err := GeneratePodSpecForComponent(
		component, backendFramework, secretsRetriever,
		dynamoDeployment, RoleWorker, numberOfNodes,
		operatorConfig, commonconsts.MultinodeDeploymentTypeDisaggregatedSet,
		component.ComponentName, checkpointInfo, deployer,
	)
	if err != nil {
		return leaderworkersetv1.LeaderWorkerSetSpec{}, fmt.Errorf(
			"failed to build worker pod spec for role %q: %w", component.ComponentName, err)
	}

	desiredReplicas := int32(1)
	if component.Replicas != nil {
		desiredReplicas = *component.Replicas
	}

	spec := leaderworkersetv1.LeaderWorkerSetSpec{
		Replicas:      &desiredReplicas,
		StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
		LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
			LeaderTemplate: &corev1.PodTemplateSpec{Spec: *leaderPodSpec},
			WorkerTemplate: corev1.PodTemplateSpec{Spec: *workerPodSpec},
			Size:           &numberOfNodes,
		},
	}
	return spec, nil
}

// CheckDisaggregatedSetReady determines if a DisaggregatedSet is fully ready
// and available. It mirrors the role-by-role readiness check that the DCD
// controller performs for the standalone-LWS pathway, but consumes the
// aggregate DisaggregatedSetStatus.roleStatuses[] instead of the LWS status.
//
// A DS is ready when every role has:
//   - observedGeneration >= generation
//   - replicas == readyReplicas == updatedReplicas (when desiredReplicas > 0)
func CheckDisaggregatedSetReady(ctx context.Context, client client.Client, resourceName, namespace string) (bool, string) {
	logger := log.FromContext(ctx)

	ds := &disaggv1alpha1.DisaggregatedSet{}
	err := client.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, ds)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.V(2).Info("DisaggregatedSet not found", "resourceName", resourceName)
			return false, "resource not found"
		}
		logger.V(1).Info("Failed to get DisaggregatedSet", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err)
	}

	if ds.Status.RoleStatuses == nil {
		return false, "roleStatuses not yet populated"
	}

	// Index statuses by role name so we can cross-check against the spec
	// and detect missing roles.
	statusByRole := make(map[string]disaggv1alpha1.RoleStatus, len(ds.Status.RoleStatuses))
	for i := range ds.Status.RoleStatuses {
		statusByRole[ds.Status.RoleStatuses[i].Name] = ds.Status.RoleStatuses[i]
	}

	var notReady []string
	for i := range ds.Spec.Roles {
		role := &ds.Spec.Roles[i]
		desiredReplicas := int32(0)
		if role.Replicas != nil {
			desiredReplicas = *role.Replicas
		}
		// Empty (zero) desiredReplicas roles are valid: skip them.
		if desiredReplicas == 0 {
			continue
		}
		st, ok := statusByRole[role.Name]
		if !ok {
			notReady = append(notReady, fmt.Sprintf("role %q: status missing", role.Name))
			continue
		}
		if st.Replicas != desiredReplicas {
			notReady = append(notReady, fmt.Sprintf("role %q: desired=%d, replicas=%d", role.Name, desiredReplicas, st.Replicas))
		}
		if st.ReadyReplicas != desiredReplicas {
			notReady = append(notReady, fmt.Sprintf("role %q: desired=%d, ready=%d", role.Name, desiredReplicas, st.ReadyReplicas))
		}
		if st.UpdatedReplicas != desiredReplicas {
			notReady = append(notReady, fmt.Sprintf("role %q: desired=%d, updated=%d", role.Name, desiredReplicas, st.UpdatedReplicas))
		}
	}

	if len(notReady) > 0 {
		return false, strings.Join(notReady, "; ")
	}
	return true, ""
}

// IsDisaggregatedSetPathwaySelected reports whether a DGD has opted into
// the DS pathway. Mirrors isGrovePathway semantics but for DS: it returns
// true when the nvidia.com/enable-disaggregatedset annotation is "true"
// AND the operator has DS enabled at the cluster level. The annotation
// is the only opt-in mechanism in Phase 1 - per-ComponentKind will come
// in a follow-up.
func IsDisaggregatedSetPathwaySelected(
	annotations map[string]string,
	runtimeConfig *controller_common.RuntimeConfig,
) bool {
	if !runtimeConfig.DisaggregatedSetEnabled {
		return false
	}
	if annotations == nil {
		return false
	}
	return strings.ToLower(annotations[commonconsts.KubeAnnotationEnableDisaggregatedSet]) == "true"
}
