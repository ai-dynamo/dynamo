# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shell helpers shared by the gen-dgd*.sh generators in this directory's siblings.
# Source it, do not run it:
#
#   source "$(dirname "$0")/../common/lib.sh"

# With NODE_NAME set, add a nodeSelector to each component's pod spec (the lines
# that are exactly "        spec:"); otherwise pass the manifest through unchanged.
add_node_selector() {
  if [ -z "${NODE_NAME:-}" ]; then
    cat
  else
    awk -v node="${NODE_NAME}" '{ print } $0 == "        spec:" {
      print "          nodeSelector:"
      print "            kubernetes.io/hostname: " node
    }'
  fi
}
