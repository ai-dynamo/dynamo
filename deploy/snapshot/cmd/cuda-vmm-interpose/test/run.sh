#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck disable=SC2016
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${BUILD_DIR:-$(mktemp -d)}"
trap 'rm -rf "$build"' EXIT

system_library=/lib/x86_64-linux-gnu/libm.so.6
sed "s|/usr/local/lib/dynamo/libdynamo_snapshot_cuda_vmm.so|$system_library|" \
  "$root/launch.sh" > "$build/launch"
chmod +x "$build/launch"
output="$(
  LD_PRELOAD="$system_library" "$build/launch" /bin/sh -c \
    'printf "%s|%s|%s" "$DYN_SNAPSHOT_CUDA_VMM_INTERPOSE" "$LD_PRELOAD" "$1"' \
    sh "argument with space"
)"
test "$output" = "1|$system_library:$system_library|argument with space"

cuda_include="${CUDA_INCLUDE:-/usr/local/cuda/include}"
test -f "$cuda_include/cuda.h"
test -f "$cuda_include/cuda_runtime_api.h"
test -f "$cuda_include/crt/host_defines.h"

mkdir -p "$build/control"
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  -I"$cuda_include" -o "$build/libcuda.so.1" "$root/test/fake_cuda.c" \
  -Wl,-Bsymbolic-functions -pthread
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -Wno-deprecated-declarations \
  -I"$cuda_include" -o "$build/libcuda.software" \
  "$root/test/fake_cuda_prefix.c" -Wl,-Bsymbolic-functions
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -Wno-deprecated-declarations -DDYN_VMM_TESTING -I"$cuda_include" \
  -o "$build/libdynamo_snapshot_cuda_vmm.so" "$root/interpose.c" \
  -Wl,-Bsymbolic-functions -ldl -pthread
cc -std=gnu11 -fPIC -shared -Wall -Wextra -Werror \
  -Wno-deprecated-declarations -I"$cuda_include" \
  -o "$build/libdynamo_snapshot_cuda_vmm.production.so" "$root/interpose.c" \
  -Wl,-Bsymbolic-functions -ldl -pthread
if nm -D --defined-only "$build/libdynamo_snapshot_cuda_vmm.production.so" |
  grep -q 'dyn_vmm_test_'; then
  echo "production CUDA VMM interposer exports test symbols" >&2
  exit 1
fi
cc -std=gnu11 -Wall -Wextra -Werror -Wno-deprecated-declarations \
  -I"$cuda_include" \
  -L"$build" -Wl,--no-as-needed -Wl,-rpath,'$ORIGIN' \
  -o "$build/resolver_test" "$root/test/resolver_test.c" -l:libcuda.so.1
cc -std=gnu11 -Wall -Wextra -Werror -Wno-deprecated-declarations \
  -I"$cuda_include" \
  -L"$build" -Wl,--no-as-needed -Wl,-rpath,'$ORIGIN' \
  -o "$build/lifecycle_test" "$root/test/lifecycle_test.c" -l:libcuda.so.1
cc -std=gnu11 -Wall -Wextra -Werror -Wno-deprecated-declarations \
  -I"$cuda_include" \
  -L"$build" -Wl,-rpath,'$ORIGIN' \
  -o "$build/explicit_loader_test" "$root/test/explicit_loader_test.c" -ldl
cc -std=gnu11 -Wall -Wextra -Werror \
  -L"$build" -Wl,-rpath,'$ORIGIN' \
  -o "$build/prefix_loader_test" "$root/test/prefix_loader_test.c" -ldl

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/resolver_test"

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/explicit_loader_test"

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/prefix_loader_test"

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/resolver_test" unshared

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/resolver_test" retained

for state in live released; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" "$state"
done

for shape in supported host other-device multi-device no-access; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" access-shape "$shape"
done

for shape in combined partial gap repeated allocation-failure; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" access-range "$shape"
done

DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
  "$build/lifecycle_test" owner-importer-success

for scenario in capability-self canonical-capability-path colliding-raw-identity cross-process; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" "$scenario"
done

for shape in \
  malformed unsealed truncated zero-uuid wrong-participant unavailable-owner \
  relative-path other-directory traversal-path alias-path duplicate-separator \
  invalid-basename zero-pid leading-zero-pid nondecimal-pid signed-pid suffix \
  raw; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" invalid-capability "$shape"
done

for stage in owner-create owner-map owner-access owner-copy owner-export owner-rebind; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" owner-failure "$stage"
done

for stage in importer-import importer-map importer-access importer-rebind; do
  DYN_SNAPSHOT_CUDA_VMM_INTERPOSE=1 \
  DYN_SNAPSHOT_CONTROL_DIR="$build/control" \
  LD_PRELOAD="$build/libdynamo_snapshot_cuda_vmm.so" \
    "$build/lifecycle_test" importer-failure "$stage"
done
