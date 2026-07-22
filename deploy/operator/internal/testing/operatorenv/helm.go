/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"fmt"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/webhookconfig"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
)

func addValidationBypassUsers(configurations []*admissionregistrationv1.ValidatingWebhookConfiguration, usernames []string) {
	for _, configuration := range configurations {
		for i := range configuration.Webhooks {
			for j, username := range usernames {
				configuration.Webhooks[i].MatchConditions = append(configuration.Webhooks[i].MatchConditions,
					admissionregistrationv1.MatchCondition{
						Name:       fmt.Sprintf("operatorenv-bypass-%d", j),
						Expression: "request.userInfo.username != " + strconv.Quote(username),
					})
			}
		}
	}
}

func helmWebhookConfigurations() ([]*admissionregistrationv1.MutatingWebhookConfiguration, []*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	return webhookconfig.HelmConfigurations()
}
