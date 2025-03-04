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

set-strictmode -version latest

$global:_init_path = "${env:PWD}"
$global:_is_debug = $null
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
  $global:DebugPreference = 'SilentlyContinue'
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

function default_is_debug {
  $value = $false
  write-debug "<default_is_debug> -> ${value}."
  return $value
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

function env_get_is_debug {
  $value = $env:NVBUILD_DEBUG_TRACE
  if (('true' -ieq $value) -or ('1' -eq $value) -or ('yes' -ieq $value)) {
    $value = $true
  }
  elseif (('false' -ieq $value) -or ('0' -eq $value) -or ('no' -ieq $value)) {
    $value = $false
  }
  # value can be $null, $true, or $false
  write-debug "<env_get_is_debug> -> $(value_or_default $value '<null>')."
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
  cleanup_after
  exit 1
}

function get_is_debug {
  if ($null -eq $global:_is_debug) {
    $value = $(env_get_is_debug)
    if ($null -ne $value) {
      set_is_debug $value
    }
    else {
      set_is_debug $(default_is_debug)
    }
  }
  write-debug "<get_is_debug> -> ${global:_is_debug}."
  return $global:_is_debug
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
    $global:_repository_root = $(& git rev-parse --show-toplevel)
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

function is_empty([string] $value) {
  return [System.String]::IsNullOrWhiteSpace($value)
}

function is_git_ignored([string] $path) {
  $repo_root = $(get_repository_root)

  if (starts_with $path $repo_root) {
    $path = $path.substring($(len $repo_root))
  }
  if (starts_with $path '/') {
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
  # $out = $path
  # if (-not [System.IO.Path]::IsPathRooted($path)) {
  #   $out = [System.IO.Path]::GetFullPath($path)
  # }
  $out = resolve-path "${path}"
  write-debug "<normalize-path> '${path}' -> '${out}'."
  return $out
}

function read_content([string] $path, [switch] $lines, [switch] $bytes) {
  if (is_empty $path) {
    throw 'Argument `path` cannot be `null` or empty.'
  }
  if ($lines -and $bytes) {
    throw 'Arguments `lines` and `bytes` are mutually exclusive.'
  }

  write-debug "<read_content> path: '${path}'."
  write-debug "<read_content> bytes: ${bytes}."
  write-debug "<read_content> lines: ${lines}."

  $path = $(to_local_path $path)

  if ($bytes) {
    return get-content -path $path -asbytestream -raw
  }
  if ($lines)
  {
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
    if (starts_with $entry.key 'NVBUILD_') {
      $overrides += $entry
    }
  }

  if ($(len $overrides) -gt 0) {
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
    throw 'Argument `command` cannot be `null`.'
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

function set_is_debug([bool] $value) {
  write-debug "<set_is_debug> value = '${value}'."

  $global:_is_debug = $value

  if ($value) {
    $global:DebugPreference = 'Continue'
  }
  else {
    $global:DebugPreference = 'SilentlyContinue'
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

function len([object] $value) {
  if ($null -eq $value) {
    return 0
  }
  $type = $(typeof $value)
  write-debug "<len> type: '${type}'."
  if ($type.endswith('[]') -or ('hashtable' -eq $type)) {
    return $value.count
  }
  if ('string' -eq $type) {
    return $value.length
  }
  return 0
}

function starts_with([string] $value, [string] $prefix) {
  if ('string' -ne $(typeof $value)) {
    throw 'Argument `value` must be a string.'
  }
  return $value.startswith($prefix)
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

function to_lower([string] $value) {
  if ('string' -ne $(typeof $value)) {
    return $value
  }
  return $value.tolower()
}

function typeof([object] $object, [switch] $full_name = $false) {
  if ($null -eq $object) {
    return 'null'
  }
  # Cannot use the `to_lower` function here, as it would cause a recursion failure.
  if ($full_name) {
    return $object.gettype().fullname.tolower()
  }
  return $object.gettype().name.tolower()
}

function usage_exit([string] $message) {
  write-error "usage: $message"
  cleanup_after
  exit 254
}

function value_or_default([object] $value, [object] $default) {
  if ($null -eq $value) {
    return $default
  }
  if (('int32' -eq $(typeof $value)) -and ($value -eq 0)) {
    return $default
  }
  if (('double' -eq $(typeof $value)) -and ($value -eq 0.0)) {
    return $default
  }
  if (('string' -eq $(typeof $value)) -and ($value.length -eq 0)) {
    return $default
  }
  if (('array' -eq $(typeof $value)) -and ($value.count -eq 0)) {
    return $default
  }
  if (('hashtable' -eq $(typeof $value)) -and ($value.count -eq 0)) {
    return $default
  }
  return $value
}

function write_content([string] $content, [string] $path, [switch] $overwrite) {
  if ($null -eq $content) {
    throw 'Argument `content` cannot be `null`.'
  }
  if (is_empty $path) {
    throw 'Argument `path` cannot be `null` or empty.'
  }

  write-debug "<write_content> content = $($content.length) bytes."
  write-debug "<write_content> path = '${path}'."
  $path_local = $(to_local_path $path)
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
  if ($null -eq $value) {
    return
  }
  if (is_tty) {
    $opts = @{
      NoNewline = $no_newline
    }
    if (($null -ne $color) -and ($(len $color) -gt 0)) {
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

function write-detailed([string] $value, [string] $color = $null, [switch] $no_newline) {
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

function write-high([string] $value, [switch] $no_newline) {
  $opts = @{
    color = $global:colors.high
    no_newline = $no_newline
  }
  write-minimal $value @opts
}

function write-low([string] $value, [switch] $no_newline) {
  $opts = @{
    color      = $global:colors.low
    no_newline = $no_newline
  }
  write-detailed $value @opts
}

function write-medium([string] $value, [switch] $no_newline) {
  $opts = @{
    color      = $global:colors.medium
    no_newline = $no_newline
  }
  write-normal $value @opts
}

function write-minimal([string] $value, [string] $color = $null, [switch] $no_newline) {
  __write $value $color $no_newline
}

function write-normal([string] $value, [string] $color = $null, [switch] $no_newline) {
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
