/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func ownerReferenceMatchesDGD(owner *metav1.OwnerReference, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	return owner != nil &&
		dgd != nil &&
		owner.APIVersion == nvidiacomv1beta1.GroupVersion.String() &&
		owner.Kind == dynamoGraphDeploymentKind &&
		owner.Name == dgd.Name &&
		owner.UID != "" &&
		owner.UID == dgd.UID
}

func dgdControllerOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment) *metav1.OwnerReference {
	if dgd == nil || dgd.UID == "" {
		return nil
	}
	return &metav1.OwnerReference{
		APIVersion:         nvidiacomv1beta1.GroupVersion.String(),
		Kind:               dynamoGraphDeploymentKind,
		Name:               dgd.Name,
		UID:                dgd.UID,
		Controller:         ptr.To(true),
		BlockOwnerDeletion: ptr.To(true),
	}
}

func setDGDControllerOwnerReference(dgd *nvidiacomv1beta1.DynamoGraphDeployment, obj client.Object) {
	ownerRef := dgdControllerOwnerReference(dgd)
	if ownerRef == nil {
		return
	}
	ownerRefs := make([]metav1.OwnerReference, 0, len(obj.GetOwnerReferences())+1)
	for _, ref := range obj.GetOwnerReferences() {
		if ptr.Deref(ref.Controller, false) {
			continue
		}
		if ref.APIVersion == ownerRef.APIVersion && ref.Kind == ownerRef.Kind && ref.Name == ownerRef.Name {
			continue
		}
		ownerRefs = append(ownerRefs, ref)
	}
	ownerRefs = append(ownerRefs, *ownerRef)
	obj.SetOwnerReferences(ownerRefs)
}

func isControlledByBetaDGD(obj client.Object, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if obj == nil || dgd == nil {
		return false
	}
	if dgd.UID != "" {
		return metav1.IsControlledBy(obj, dgd)
	}
	controllerOwner := metav1.GetControllerOf(obj)
	return controllerOwner != nil &&
		controllerOwner.APIVersion == nvidiacomv1beta1.GroupVersion.String() &&
		controllerOwner.Kind == dynamoGraphDeploymentKind &&
		controllerOwner.Name == dgd.Name
}
