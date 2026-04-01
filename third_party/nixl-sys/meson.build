# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cargo = find_program('cargo', required: false)

if cargo.found() and get_option('rust')
  RUST_BINDINGS_NAME = 'nixl-sys'

  cargo_build_cmd = [
        cargo, 'build', '--target-dir',
        'src' / 'bindings' / 'rust' / RUST_BINDINGS_NAME,
  ]

  if get_option('buildtype') == 'release'
    cargo_build_cmd += ['--release']
  endif

  nixl_include_dir = include_directories('../../api/cpp', '../../infra', '../../core')
  nixl_dep = declare_dependency(link_with: nixl_lib, include_directories: nixl_include_dir)

  wrapper_sources = ['wrapper.cpp']
  wrapper_lib = static_library('nixl_wrapper',
      sources: wrapper_sources,
      include_directories: [nixl_include_dir, include_directories('/usr/include')],
      link_with: nixl_lib,
      dependencies: [nixl_dep]
    )

  rust_lib = custom_target(
      RUST_BINDINGS_NAME,
      command: cargo_build_cmd,
      input: [],
      output: RUST_BINDINGS_NAME,
      depends: [wrapper_lib],
      install: true,
      install_dir: '.',
      env: [
        'NIXL_PREFIX=' + get_option('prefix'),
      ]
    )
else
  nixl_rust_bindings_lib = disabler()
endif
