// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package registry

import (
	"regexp"
)

var dnsLabel = regexp.MustCompile(`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`)

// IsDNSLabel reports whether s is an RFC 1123 DNS label: at most 63
// lowercase letters, digits and dashes, starting and ending with a letter or
// digit. Clusterset, cluster, namespace and Service names are DNS labels.
func IsDNSLabel(s string) bool {
	return len(s) <= 63 && dnsLabel.MatchString(s)
}
