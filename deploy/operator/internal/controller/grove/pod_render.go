// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package grove

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// RenderPodMetadata stamps the graph and component identity on resolved Pod metadata.
// Inputs are not modified.
func RenderPodMetadata(graphName, selector string, metadata metav1.ObjectMeta) metav1.ObjectMeta {
	metadata = *metadata.DeepCopy()
	if metadata.Labels == nil {
		metadata.Labels = make(map[string]string)
	}
	metadata.Labels[consts.KubeLabelDynamoSelector] = selector
	metadata.Labels[consts.KubeLabelDynamoGraphDeploymentName] = graphName
	return metadata
}
