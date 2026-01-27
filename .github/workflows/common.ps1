# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set-strictmode -version latest

$global:_init_path = "${env:PWD}"
$global:_git_branch = $null
$global:_local = $null
$global:_local_srcdir = $null
$global:_repository_root = $null
$global:_verbosity = $null

$global:colors = @{
  error = 'Red'
  high = 'Cyan'
  low = 'DarkGray'
  medium = 'DarkBlue'
  test = @{
    failed = 'Red'
    passed = 'Green'
  }
  title = 'Blue'
  warning = 'Yellow'
}

function cleanup_after {
  write-debug "<cleanup_after>"
  $(reset_environment)
}

function configure_debug([bool] $enabled) {
  write-debug "<configure_debug> enabled = '${enabled}'."

  if ($enabled) {
    # Notify user if any environment variables which could affect the build outcome are set prior to the build script running.
    $overrides = @()

    foreach ($entry in $(& get-childitem env:)) {
      # We're only looking for environment variables which are used directly by the build scripts (starts with 'NVBUILD_`);
      # and we're looking at environment variables which would indirectly affect the build scripts (i.e. `PATH`).
      if ($entry.key.startswith('NVBUILD_')) {
        # No reason to display values which are displayed when the build start, often time provided by the user.
        if ($entry.key.endswith('_COMMAND') -or $entry.key.endswith('_VERBOSITY')) {
          continue
        }

        $overrides += $entry
      }
    }

    if ($overrides.count -gt 0) {
      write-low ' Overriding environment variables:'
      foreach ($entry in $overrides) {
        write-low "  $($entry.key) = $($entry.Value)"
      }
    }
  }
}

function create_directory([string] $path, [switch] $recreate) {
  write-debug "<create_directory> path = '${path}'."
  write-debug "<create_directory> recreate = ${recreate}"

  $path_local = $(to_local_path $path)
  write-debug "<ensure_directory> path_local = '${path_local}'."

  if (test-path $path_local -pathType Container) {
    if ($recreate) {
      remove-item $path_local -Recurse | out-null
      new-item $path_local -itemtype Directory | out-null
    }
  }
  else {
    new-item $path_local -itemtype Directory | out-null
  }
}

function default_git_branch {
  if (is_installed 'git') {
    $value = "$(git branch --show-current)"
  }
  else {
    $value = 'main'
  }
  write-debug "<default_git_branch> -> '${value}'."
  return $value
}

function default_local_srcdir {
  $value = $(& git rev-parse --show-toplevel)
  write-debug "<default_local_srcdir> -> '${value}'."
  return $value;
}

function default_verbosity {
  $value = 'NORMAL'
  write-debug "<default_verbosity> -> '${value}'."
  return $value
}

function env_get_git_branch {
  $value = $env:NVBUILD_GIT_BRANCH
  write-debug "<env_get_git_branch> -> '${value}'."
  return $value
}

function env_get_local_srcdir {
  $value = $env:NVBUILD_LOCAL_SRCDIR
  write-debug "<env_get_local_srcdir> -> '${value}'."
  return $value
}

function env_get_verbosity {
  $value = $env:NVBUILD_VERBOSITY
  write-debug "<env_get_verbosity> -> '${value}'."
  return $value
}

function env_set_git_branch([string] $value) {
  if ($null -eq $env:NVBUILD_NOSET) {
    write-debug "<env_set_git_branch> value: '${value}'."
    $env:NVBUILD_GIT_BRANCH = $value
  }
}

function env_set_local_srcdir([string] $value) {
  if ($null -eq $env:NVBUILD_NOSET) {
    write-debug "<env_set_local_srcdir> value: '${value}'."
    $env:NVBUILD_LOCAL_SRCDIR = $value
  }
}

function env_set_verbosity([string] $value) {
  if ($null -eq $env:NVBUILD_NOSET) {
    write-debug "<env_set_verbosity> value: '${value}'."
    $env:NVBUILD_VERBOSITY = $value
  }
}

function fatal_exit([string] $message) {
  write-error "fatal: ${message}"
  exit 1
}

function get_git_branch {
  if ($null -eq $global:_git_branch) {
    $value = $(env_set_git_branch)
    if ($null -ne $value) {
      set_git_branch $value
    }
    else {
      set_git_branch $(default_git_branch)
    }
  }
  write-debug "<get_git_branch> -> '${global:_git_branch}'."
  return $global:_git_branch
}

function get_local_srcdir {
  if ($null -eq $global:_local_srcdir) {
    $value = $(env_get_local_srcdir)
    if ($null -ne $value) {
      set_local_srcdir $value
    }
    else {
      set_local_srcdir $(default_local_srcdir)
    }
  }
  write-debug "<get_local_srcdir> -> '${global:_local_srcdir}'."
  return $global:_local_srcdir
}

function get_repository_root {
  if ($null -eq $global:_repository_root) {
    $path = $(& git rev-parse --show-toplevel)
    $global:_repository_root = $(normalize_path $path)
  }

  write-debug "<get_repository_root> '${global:_repository_root}'."
  return $global:_repository_root
}

function get_verbosity {
  if ($null -eq $global:_verbosity) {
    $value = $(env_get_verbosity)
    if ($null -ne $value) {
      set_verbosity $value
    }
    else {
      set_verbosity $(default_verbosity)
    }
  }
  write-debug "<get_verbosity> -> '${global:_verbosity}'."
  return $global:_verbosity
}

function is_debug {
  $value = $($null -ne $env:NVBUILD_DEBUG_TRACE)
  write-debug "<is_debug> -> ${value}."
  return $value
}

function is_empty([string] $value) {
  return [System.String]::IsNullOrWhiteSpace($value)
}

function is_git_ignored([string] $path) {
  $repo_root = $(get_repository_root)

  if ($path.startswith($repo_root)) {
    $path = $path.substring($repo_root.length)
  }
  if ($path.startswith('/')) {
    $path = $path.substring(1)
  }

  $result = $(& git check-ignore $path)
  return (0 -eq $result)
}

function is_installed([string] $command) {
  write-debug "<is_installed> command = '${command}'."
  $out = $null -ne $(get-command "${command}" -errorAction SilentlyContinue)
  write-debug "<is_installed> -> ${out}."
  return $out
}

function is_tty {
  return -not(([System.Console]::IsOutputRedirected) -or ([System.Console]::IsErrorRedirected))
}

function is_verbosity_valid([string] $value) {
  return (('NORMAL' -eq $value) -or ('MINIMAL' -eq $value) -or ('DETAILED' -eq $value))
}

function normalize_path([string] $path) {
  write-debug "<normalize-path> path: '${path}'."
  $out = "$(resolve-path "${path}" -erroraction 'Ignore')"
  if (($null -eq $out) -or ($out.length -eq 0)) {
    if ($path.startswith($(get_repository_root))) {
      $out = $path
    }
    else {
      $out = "$(get_repository_root)/${path}"
    }
  }
  if ($IsWindows) {
    if ($out -match "^[A-Z]:") {
      $out = $out.substring(2)
    }
    $out = $out.replace('\', '/')
  }
  write-debug "<normalize-path> '${path}' -> '${out}'."
  return $out
}

function read_content([string] $path, [switch] $lines, [switch] $bytes) {
  if (is_empty $path -or ($lines -and $bytes)) {
    write-error 'usage: read_content {path} [(-bytes|-lines)]' -category InvalidArgument
    write-error ' {path} file system path of the file to read contents from.'
    write-error ' -bytes when provided content is returned as an array of bytes. mutually exclusive with -lines.'
    write-error ' -lines when provided content is returned as an array of strings. mutually exclusive with -bytes.'
    write-error ' '
    usage_exit 'read_content {path} [(-bytes|-lines)]'
  }

  write-debug "<read_content> path: '${path}'."
  write-debug "<read_content> bytes: ${bytes}."
  write-debug "<read_content> lines: ${lines}."

  $path = normalize_path $path

  if ($bytes) {
    return get-content -path $path -asbytestream -raw
  }
  if ($lines) {
    return get-content -path $path
  }

  return get-content -path $path -raw
}

function reset_environment {
  write-debug "<reset_environment>"

  $overrides = @()

  foreach ($entry in $(& get-childitem env:)) {
    # We're only looking for environment variables which are used directly by the build scripts (starts with 'NVBUILD_`);
    # and we're looking at environment variables which would indirectly affect the build scripts (i.e. `PATH`).
    if ($entry.key.startswith('NVBUILD_')) {
      $overrides += $entry
    }
  }

  if ($overrides.count -gt 0) {
    foreach ($entry in $overrides) {
      $expression = '$env:' + "$($entry.Key)" + ' = $null'
      invoke-expression "${expression}"

      if ("$($entry.Key)" -ne 'NVBUILD_NOSET') {
        write-debug "<reset_environment> removed '$($entry.Key)'."
      }
    }
  }
}

function run([string] $command) {
  if ($null -eq $command) {
    write-error 'usage: run {command}' -category InvalidArgument
    write-error ' {command} is the command to execute.'
    write-error ' '
    usage_exit 'run {command}'
  }

  write-debug "<run> command = '${command}'."

  if ('MINIMAL' -ne $(get_verbosity)) {
    write-high "${command}"
  }

  invoke-expression "${command}" | out-default
  $exit_code = $LASTEXITCODE

  write-debug "<run> exit_code = ${exit_code}."

  if ($exit_code -ne 0) {
    write-error "fatal: Command ""${command}"" failed, returned ${exit_code}." -category fromStdErr
    exit $exit_code
  }
}

function set_git_branch([string] $value) {
  write-debug "<set_git_branch> value = '${value}'."

  $global:_git_branch = $value
  env_set_git_branch $value
}

function set_local_srcdir([string] $value) {
  write-debug "<set_local_srcdir> value: '${value}'."

  $global:_local_srcdir = $value
  env_set_local_srcdir $value
}

function set_verbosity([string] $value) {
  write-debug "<set_verbosity> '${value}'."

  if (-not(is_verbosity_valid $value)) {
    throw "Invalid verbosity value '${value}'."
  }

  $global:_verbosity = $value
  env_set_verbosity $value
}

function to_local_path([string] $path) {
  write-debug "<to_local_path> path: '${path}'."

  if ($null -eq $path) {
    return $(get_local_srcdir)
  }

  $out = $path.trim()
  $out = $out.trim('/','\')
  $out = join-path $(get_local_srcdir) $out
  $out = $(normalize_path $out)
  return $out
}

function typeof($object) {
  if ($null -eq $object) {
    return 'null'
  }

  return $object.gettype().name
}

function usage_exit([string] $message) {
  write-error "usage: $message"
  exit 254
}

function value_or_default([string] $value, [string] $default) {
  if (($null -eq $value) -or ($value.Length -eq 0)) {
    return $default
  }

  return $value
}

function write_content([string] $content, [string] $path, [switch] $overwrite) {
  if (($null -eq $path) -or ($path.length -eq 0)) {
    write-error 'usage: write_content {content} {path}'
    write-error ' {content} is the content to be written to a file.'
    write-error ' {path} is the path to file into which to write content.'
    usage_exit 'write_content {content} {path}'
  }

  write-debug "<write_content> content = $($content.length) bytes."
  write-debug "<write_content> path = '${path}'."
  $path_local = normalize_path $path
  write-debug "<write-content> '${path_local}'."

  if ($null -eq $content) {
    $content = ''
  }

  if ($overwrite -and (test-path $path_local)) {
    remove-item $path_local | out-null
  }

  $content | out-file $path_local
}

function __write([string] $value, [string] $color, [bool] $no_newline) {
  if (is_tty) {
    $opts = @{
      NoNewline = $no_newline
    }
    if (($null -ne $color) -and ($color.length -gt 0)) {
      $opts.ForegroundColor = $color
    }
    write-host $value @opts
  }
  else {
    if (-not($no_newline)) {
      $value = "${value}`n"
    }
    write-output $value
  }
}

function write-detailed {
  param([string] $value, [string] $color = $null, [switch] $no_newline)
  if ('DETAILED' -eq $(get_verbosity)) {
    __write $value $color $no_newline
  }
}

function write-error([string] $value) {
  $opts = @{
    color = $global:colors.error
    no_newline = $false
  }
  write-minimal $value @opts
}

function write-failed([string] $value) {
  if (is_tty) {
    write-normal '  [Failed]' $global:colors.test.failed -no_newline
    write-normal " ${value}"
  }
  else {
    write-output "  Test: [Failed] ${value}"
  }
}

function write-high {
  param([string] $value, [switch] $no_newline)
  $opts = @{
    color = $global:colors.high
    no_newline = $no_newline
  }
  write-minimal $value @opts
}

function write-low {
  param([string] $value, [switch] $no_newline)
  $opts = @{
    color      = $global:colors.low
    no_newline = $no_newline
  }
  write-detailed $value @opts
}

function write-medium {
  param([string] $value, [switch] $no_newline)
  $opts = @{
    color      = $global:colors.medium
    no_newline = $no_newline
  }
  write-normal $value @opts
}

function write-minimal {
  param([string] $value, [string] $color = $null, [switch] $no_newline)
  __write $value $color $no_newline
}

function write-normal {
  param([string] $value, [string] $color = $null, [switch] $no_newline)
  if ('MINIMAL' -ne $(get_verbosity)) {
    $opts = @{
      color      = $color
      no_newline = $no_newline
    }
    __write $value @opts
  }
}

function write-passed([string] $value) {
  if (is_tty) {
    write-detailed '  [Passed]' $global:colors.test.passed -no_newline
    write-detailed " ${value}"
  }
  else {
    write-output "  Test: [Passed] ${value}"
  }
}

function write-title([string] $value) {
  write-minimal $value $global:colors.title
}

function write-warning([string] $value) {
  write-minimal $value $global:colors.warning
}
