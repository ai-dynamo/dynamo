/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1alpha1

import (
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
)

// ConvertTo converts this DynamoCheckpoint (v1alpha1) to the Hub version (v1beta1).
func (src *DynamoCheckpoint) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoCheckpoint)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoCheckpoint but got %T", dstRaw)
	}

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Identity = convertCheckpointIdentityTo(src.Spec.Identity)
	dst.Spec.Job = v1beta1.DynamoCheckpointJobConfig{
		PodTemplateSpec:         src.Spec.Job.PodTemplateSpec,
		ActiveDeadlineSeconds:   src.Spec.Job.ActiveDeadlineSeconds,
		BackoffLimit:            src.Spec.Job.BackoffLimit,
		TTLSecondsAfterFinished: src.Spec.Job.TTLSecondsAfterFinished,
	}

	// Status
	dst.Status.Phase = v1beta1.DynamoCheckpointPhase(src.Status.Phase)
	dst.Status.IdentityHash = src.Status.IdentityHash
	dst.Status.Location = src.Status.Location
	dst.Status.StorageType = v1beta1.DynamoCheckpointStorageType(src.Status.StorageType)
	dst.Status.JobName = src.Status.JobName
	dst.Status.CreatedAt = src.Status.CreatedAt
	dst.Status.Message = src.Status.Message
	dst.Status.Conditions = src.Status.Conditions

	return nil
}

// ConvertFrom converts from the Hub version (v1beta1) to this DynamoCheckpoint (v1alpha1).
func (dst *DynamoCheckpoint) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoCheckpoint)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoCheckpoint but got %T", srcRaw)
	}

	// ObjectMeta
	dst.ObjectMeta = src.ObjectMeta

	// Spec
	dst.Spec.Identity = convertCheckpointIdentityFrom(src.Spec.Identity)
	dst.Spec.Job = DynamoCheckpointJobConfig{
		PodTemplateSpec:         src.Spec.Job.PodTemplateSpec,
		ActiveDeadlineSeconds:   src.Spec.Job.ActiveDeadlineSeconds,
		BackoffLimit:            src.Spec.Job.BackoffLimit,
		TTLSecondsAfterFinished: src.Spec.Job.TTLSecondsAfterFinished,
	}

	// Status
	dst.Status.Phase = DynamoCheckpointPhase(src.Status.Phase)
	dst.Status.IdentityHash = src.Status.IdentityHash
	dst.Status.Location = src.Status.Location
	dst.Status.StorageType = DynamoCheckpointStorageType(src.Status.StorageType)
	dst.Status.JobName = src.Status.JobName
	dst.Status.CreatedAt = src.Status.CreatedAt
	dst.Status.Message = src.Status.Message
	dst.Status.Conditions = src.Status.Conditions

	return nil
}

func convertCheckpointIdentityTo(src DynamoCheckpointIdentity) v1beta1.DynamoCheckpointIdentity {
	return v1beta1.DynamoCheckpointIdentity{
		Model:                src.Model,
		BackendFramework:     src.BackendFramework,
		DynamoVersion:        src.DynamoVersion,
		TensorParallelSize:   src.TensorParallelSize,
		PipelineParallelSize: src.PipelineParallelSize,
		Dtype:                src.Dtype,
		MaxModelLen:          src.MaxModelLen,
		ExtraParameters:      src.ExtraParameters,
	}
}

func convertCheckpointIdentityFrom(src v1beta1.DynamoCheckpointIdentity) DynamoCheckpointIdentity {
	return DynamoCheckpointIdentity{
		Model:                src.Model,
		BackendFramework:     src.BackendFramework,
		DynamoVersion:        src.DynamoVersion,
		TensorParallelSize:   src.TensorParallelSize,
		PipelineParallelSize: src.PipelineParallelSize,
		Dtype:                src.Dtype,
		MaxModelLen:          src.MaxModelLen,
		ExtraParameters:      src.ExtraParameters,
	}
}
