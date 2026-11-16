// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"maps"
	"math/big"
	"path"
	"slices"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	consts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

const (
	deploymentUIDLabel = "lpx.nvidia.com/deployment-uid"
	// clientCertAnnotation publishes an agent Pod's client certificate, base64 DER.
	clientCertAnnotation = "lpx.nvidia.com/client-cert"
	// clientIdentityAnnotation names the identity Secret and its certificate's SHA-256 on every
	// clique and scaling group. New keys change it, so Grove rolls every Pod onto them.
	clientIdentityAnnotation = "lpx.nvidia.com/client-identity"
	// identityVolume and identityMountPath belong to the operator in agent templates.
	identityVolume    = "lpx-identity"
	identityMountPath = "/etc/lpx/identity"
)

// resolveWorkloads resolves every component group and finalizes its resource names.
// deployment and dgd must be non-nil; dgd must have passed admission.
// Returned maps share conductor component keys. Inputs are not mutated.
func (r *graphReconciler) resolveWorkloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
) (map[string]*lpx.Workload, map[string]*lpx.MaterializationPlan, error) {
	var (
		workloads = make(map[string]*lpx.Workload)
		plans     = make(map[string]*lpx.MaterializationPlan)
		groups    = lpx.ComponentGroups(dgd)
	)

	registry := r.modelRegistry
	if !adoptBuildCheckpoints(r.config.Infrastructure.ModelExpressURL, deployment) {
		registry = lpx.WithoutBuildCheckpoints(registry)
	}

	for _, groupName := range slices.Sorted(maps.Keys(groups)) {
		workload, err := lpx.ResolveWorkload(ctx, r.Client, dgd, groups[groupName], registry)
		if err != nil {
			return nil, nil, err
		}
		workloads[groupName] = workload

		plan, err := workload.PlanNodeLocalMaterialization(dynamo.PCSNameForLPX(deployment), dgd.GetComponentByName(workload.ServingComponentName()))
		if err != nil {
			return nil, nil, err
		}

		// Only independent workloads need additional names within the shared PCS.
		if len(groups) > 1 {
			plan, err = plan.WithGroup(groupName)
			if err != nil {
				return nil, nil, err
			}
		}

		// Compare finalized identities only against previously resolved groups.
		for previousGroup, previous := range plans {
			if previous.ScalingGroupTemplate == plan.ScalingGroupTemplate || previous.ResourcePrefix == plan.ResourcePrefix {
				return nil, nil, fmt.Errorf("LPX components %q and %q resolve to conflicting resource names; rename one component", previousGroup, groupName)
			}
		}
		plans[groupName] = plan
	}

	return workloads, plans, nil
}

// renderPodCliqueSet composes resolved workloads into one Grove envelope with
// runtime ConfigMaps and, for Kubernetes discovery, serving Services.
// Inputs must be non-nil except existingPCS, which is nil during creation.
// Workloads must contain every component group.
// Plans must have finalized names and use the same keys as workloads.
// Inputs remain read-only.
func (r *graphReconciler) renderPodCliqueSet(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workloads map[string]*lpx.Workload,
	plans map[string]*lpx.MaterializationPlan,
	existingPCS *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Shared defaults and queue resolution belong to the single PCS envelope.
	pcs, err := dynamo.RenderLPXPodCliqueSet(ctx, dgd, r.config, r.runtimeConfig, dynamo.PCSNameForLPX(deployment), existingPCS)
	if err != nil {
		return nil, nil, err
	}

	var resources []client.Object

	for _, groupName := range slices.Sorted(maps.Keys(plans)) {
		workload, plan := workloads[groupName], plans[groupName]

		// Render this workload's roles using the full graph for shared defaults.
		rendered, err := dynamo.RenderLPXWorkloadTemplates(dgd, r.config, r.runtimeConfig, r.dockerSecretRetriever, workload, plan)
		if err != nil {
			return nil, nil, err
		}

		// Discovery selects only this workload's serving role; all auxiliary roles stay private.
		serving := plan.ConductorTemplate
		if serving == "" {
			serving = plan.CyborgTemplate
		}
		if serving == "" {
			serving = plan.Agents[len(plan.Agents)-1].TemplateName
		}
		for _, clique := range rendered.Cliques {
			if token := deployment.Annotations[dynamo.LPXRestartAnnotation]; token != "" {
				clique.Annotations[consts.RestartAnnotation] = token
			}
			delete(clique.Labels, dynamo.LPXServingLabel)
			if clique.Name == serving {
				clique.Labels[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			} else {
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryBackend)
				delete(clique.Labels, consts.KubeLabelDynamoBaseModelHash)
			}
		}

		// Keep workload templates in canonical component order.
		pcs.Spec.Template.Cliques = append(pcs.Spec.Template.Cliques, rendered.Cliques...)
		pcs.Spec.Template.PodCliqueScalingGroupConfigs = append(pcs.Spec.Template.PodCliqueScalingGroupConfigs, rendered.ScalingGroup)
		resources = append(resources, rendered.Resources...)

		// Kubernetes discovery exposes only this workload's serving role within its PCS.
		if commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dgd.Annotations) {
			component := dgd.GetComponentByName(workload.ServingComponentName())
			service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
				ServiceName: plan.ResourcePrefix + "-serve", Namespace: deployment.Namespace,
				ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
				DynamoNamespace: dgd.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
				Labels:      dynamo.GetDGDComponentResourceLabels(dgd, component.ComponentName, component),
				Annotations: dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component),
			})
			if err != nil {
				return nil, nil, err
			}
			service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			service.Spec.Selector[grovecommon.LabelPartOfKey] = pcs.Name
			resources = append(resources, service)
		}

	}

	stampDeploymentIdentity(deployment, pcs, resources)

	// Enforce the aggregate size budget after identity and discovery metadata are final;
	// stampClientIdentity checks it again after adding the client identity.
	if err := checkPodCliqueSetSize(pcs); err != nil {
		return nil, nil, err
	}
	return pcs, resources, nil
}

func checkPodCliqueSetSize(pcs *grovev1alpha1.PodCliqueSet) error {
	serialized, err := json.Marshal(pcs)
	if err != nil {
		return fmt.Errorf("serializing selected LPX PodCliqueSet: %w", err)
	}
	if len(serialized) > lpx.MaxRenderedPodCliqueSetBytes {
		return fmt.Errorf("rendered LPX PodCliqueSet is %d bytes; maximum is %d", len(serialized), lpx.MaxRenderedPodCliqueSetBytes)
	}
	return nil
}

// stampDeploymentIdentity propagates stable ownership labels and annotations, never DGD revision.
func stampDeploymentIdentity(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, resources []client.Object) {
	stamp := func(annotations *map[string]string) {
		if *annotations == nil {
			*annotations = make(map[string]string)
		}
		(*annotations)[lpx.DeploymentNameAnnotation] = deployment.Name
	}
	stampOwnerLabel := func(object client.Object) {
		labels := object.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		labels[deploymentUIDLabel] = string(deployment.UID)
		object.SetLabels(labels)
	}
	stamp(&pcs.Annotations)
	stampOwnerLabel(pcs)
	for _, clique := range pcs.Spec.Template.Cliques {
		stamp(&clique.Annotations)
	}
	for i := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		stamp(&pcs.Spec.Template.PodCliqueScalingGroupConfigs[i].Annotations)
	}
	for _, resource := range resources {
		annotations := resource.GetAnnotations()
		stamp(&annotations)
		resource.SetAnnotations(annotations)
		stampOwnerLabel(resource)
	}
}

// renderDesiredPodCliqueSet renders the PCS and stamps the DGD's client identity on it.
func (r *graphReconciler) renderDesiredPodCliqueSet(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workloads map[string]*lpx.Workload,
	plans map[string]*lpx.MaterializationPlan,
	existingPCS *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	pcs, resources, err := r.renderPodCliqueSet(ctx, deployment, dgd, workloads, plans, existingPCS)
	if err != nil {
		return nil, nil, err
	}
	return pcs, resources, r.stampClientIdentity(ctx, dgd, pcs)
}

// stampClientIdentity gives every LPX Pod its DGD's UID label, and mounts into each agent Pod's main
// container the DGD's client key and certificate from Secret lpx-id-<DGD UID> plus the optional
// lpx-server-ca bundle. The Secret is created once, never rewritten and never adopted. Agent Pods
// also publish the certificate in an annotation, so node-side software can authenticate them.
// Secret references to the cluster-wide MPI SSH Secret in LPX Pods get the DGD's own SSH key pair
// instead, under the same key names, so one DGD's key no longer opens another DGD's Pods.
func (r *graphReconciler) stampClientIdentity(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, pcs *grovev1alpha1.PodCliqueSet) error {
	name := "lpx-id-" + string(dgd.UID)
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: name}
	identity := &corev1.Secret{}
	err := r.Get(ctx, key, identity)
	if apierrors.IsNotFound(err) {
		if identity, err = newClientIdentitySecret(dgd.Namespace, name); err == nil {
			if err = controllerutil.SetControllerReference(dgd, identity, r.Scheme()); err == nil {
				// An earlier reconcile may have created it before the cache observed it.
				if err = r.Create(ctx, identity); apierrors.IsAlreadyExists(err) {
					// Decode into a fresh object, so nothing generated above survives into the ownership check.
					identity = &corev1.Secret{}
					err = r.apiReader.Get(ctx, key, identity)
				}
			}
		}
	}
	if err != nil {
		return fmt.Errorf("LPX client identity Secret %s/%s: %w", dgd.Namespace, name, err)
	}
	block, _ := pem.Decode(identity.Data["client.pem"])
	if !metav1.IsControlledBy(identity, dgd) || block == nil {
		return fmt.Errorf("LPX client identity Secret %s/%s is not controlled by this DGD or holds no certificate", dgd.Namespace, name)
	}

	// Template items keep resolving; without items, only the SSH keys are projected.
	sshItems := func(items []corev1.KeyToPath) []corev1.KeyToPath {
		if len(items) > 0 {
			return items
		}
		return []corev1.KeyToPath{{Key: "private.key", Path: "private.key"}, {Key: "private.key.pub", Path: "private.key.pub"}}
	}
	// A recreated Secret holds new keys under the same name, so the marker names the certificate too.
	marker := fmt.Sprintf("%s/%x", name, sha256.Sum256(block.Bytes))
	for i := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		group := &pcs.Spec.Template.PodCliqueScalingGroupConfigs[i]
		if group.Annotations == nil {
			group.Annotations = make(map[string]string)
		}
		group.Annotations[clientIdentityAnnotation] = marker
	}
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Labels == nil {
			clique.Labels = make(map[string]string)
		}
		clique.Labels[consts.KubeLabelDynamoGraphDeploymentUID] = string(dgd.UID)
		if clique.Annotations == nil {
			clique.Annotations = make(map[string]string)
		}
		clique.Annotations[clientIdentityAnnotation] = marker
		for i := range clique.Spec.PodSpec.Volumes {
			volume := &clique.Spec.PodSpec.Volumes[i]
			if volume.Secret != nil && volume.Secret.SecretName == r.config.MPI.SSHSecretName {
				volume.Secret.SecretName, volume.Secret.Items = name, sshItems(volume.Secret.Items)
			}
			for _, source := range ptr.Deref(volume.Projected, corev1.ProjectedVolumeSource{}).Sources {
				if source.Secret != nil && source.Secret.Name == r.config.MPI.SSHSecretName {
					source.Secret.Name, source.Secret.Items = name, sshItems(source.Secret.Items)
				}
			}
		}
		if clique.Annotations[lpxv1alpha1.PodRoleAnnotation] != lpxv1alpha1.PodRoleAgent {
			// Only agents publish a certificate; one authored on any other role is dropped.
			delete(clique.Annotations, clientCertAnnotation)
			continue
		}
		clique.Annotations[clientCertAnnotation] = base64.StdEncoding.EncodeToString(block.Bytes)
		spec := &clique.Spec.PodSpec
		if slices.ContainsFunc(spec.Volumes, func(v corev1.Volume) bool { return v.Name == identityVolume }) {
			return fmt.Errorf("LPX agent template %s already defines volume %q, which the operator adds", clique.Name, identityVolume)
		}
		spec.Volumes = append(spec.Volumes, corev1.Volume{Name: identityVolume,
			Projected: &corev1.ProjectedVolumeSource{DefaultMode: new(int32(0o440)), Sources: []corev1.VolumeProjection{
				{Secret: &corev1.SecretProjection{
					Name:  name,
					Items: []corev1.KeyToPath{{Key: "client.pem", Path: "client.pem"}, {Key: "client.key", Path: "client.key"}},
				}},
				// Optional, so Pods start where the platform has not published it.
				{Secret: &corev1.SecretProjection{
					Name:     "lpx-server-ca",
					Items:    []corev1.KeyToPath{{Key: "ca.crt", Path: "ca.pem"}},
					Optional: new(true),
				}},
			}}})
		// Only the agent's main runtime authenticates to LPUD; a missing main must fail, never mount nowhere.
		container := common.FindContainerByName(spec.Containers, consts.MainContainerName)
		if container == nil {
			return fmt.Errorf("LPX agent template %s has no %q container", clique.Name, consts.MainContainerName)
		}
		if slices.ContainsFunc(container.VolumeMounts, func(m corev1.VolumeMount) bool { return path.Clean(m.MountPath) == identityMountPath }) {
			return fmt.Errorf("LPX agent template %s already mounts %s, which the operator owns", clique.Name, identityMountPath)
		}
		container.VolumeMounts = append(container.VolumeMounts,
			corev1.VolumeMount{Name: identityVolume, MountPath: identityMountPath, ReadOnly: true})
	}
	return checkPodCliqueSetSize(pcs)
}

// newClientIdentitySecret returns an Ed25519 key with its self-signed, non-CA client certificate,
// valid from one hour ago for ten years, and an SSH key pair.
func newClientIdentitySecret(namespace, name string) (*corev1.Secret, error) {
	now := time.Now()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 127))
	if err != nil {
		return nil, err
	}
	template := &x509.Certificate{
		SerialNumber:          serial,
		Subject:               pkix.Name{CommonName: name},
		NotBefore:             now.Add(-time.Hour),
		NotAfter:              now.AddDate(10, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}
	certDER, err := x509.CreateCertificate(rand.Reader, template, template, public, private)
	if err != nil {
		return nil, err
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		return nil, err
	}
	sshKey, sshPub, err := secret.GenerateSSHKeyPair()
	if err != nil {
		return nil, err
	}
	return &corev1.Secret{
		Name: name, Namespace: namespace,
		Type: corev1.SecretTypeOpaque,
		// The key is never rewritten, so nobody may swap it in place either.
		Immutable: new(true),
		Data: map[string][]byte{
			"client.pem": pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER}),
			"client.key": pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}),
			// The shared MPI SSH Secret's key names, so template items resolve unchanged.
			"private.key":     sshKey,
			"private.key.pub": sshPub,
		},
	}, nil
}

// markBuildCheckpointAdoption records that new input adopts hybrid builds' own
// checkpoints. The record is the checkpointDownload status, which persists the
// decision while build downloads hold rendering across reconciles that each
// advance observedGeneration. Input counts as new until a reconcile
// acknowledges its generation, including input that an earlier operator never
// acknowledged: rendering that input changes the workload anyway, so adopting
// its checkpoint adds no extra roll. A deployment whose generation is
// acknowledged and that has no record keeps its workload across an
// operator-only upgrade. deployment is non-nil.
func markBuildCheckpointAdoption(modelExpressURL string, deployment *v1alpha1.LPXGraphDeployment) {
	if modelExpressURL == "" || deployment.Status.CheckpointDownload != nil ||
		deployment.Generation == deployment.Status.ObservedGeneration {
		return
	}
	deployment.Status.CheckpointDownload = &v1alpha1.CheckpointDownloadStatus{}
}

// adoptBuildCheckpoints reports whether hybrid builds' own checkpoints reach the
// Cyborg conductor: Model Express is configured and the deployment records
// adoption. Without Model Express the operator cannot confirm that a checkpoint
// is in model storage, so it projects none: removing the Model Express URL
// renders the authored conductor environment again, and restoring the URL
// resumes projection from the retained record after another download check.
// deployment is non-nil.
func adoptBuildCheckpoints(modelExpressURL string, deployment *v1alpha1.LPXGraphDeployment) bool {
	return modelExpressURL != "" && deployment.Status.CheckpointDownload != nil
}
