/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package webhookconfig

import admissionregistrationv1 "k8s.io/api/admissionregistration/v1"

// Configurations groups admission registrations whose handlers are installed
// in the test environment's webhook manager.
type Configurations struct {
	Mutating   []*admissionregistrationv1.MutatingWebhookConfiguration
	Validating []*admissionregistrationv1.ValidatingWebhookConfiguration
}
