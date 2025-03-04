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

. "$(& git rev-parse --show-toplevel)/deploy/Kubernetes/_build/common.ps1"

# == begin common.ps1 extensions ==

$global:_print_template = $null

function default_print_template {
  $value = $false
  write-debug "<default_print_template> -> ${value}."
  return $value
}

function env_get_print_template {
  $value = $($null -ne $env:NVBUILD_PRINT_TEMPLATE)
  write-debug "<env_get_print_template> -> '${value}'."
  return $value
}

function env_set_print_template([bool] $value) {
  if ($null -eq $env:NVBUILD_NOSET) {
    write-debug "<env_set_print_template> value: ${value}."
    if ($value) {
      $env:NVBUILD_PRINT_TEMPLATE = '1'
    }
    else {
      $env:NVBUILD_PRINT_TEMPLATE = $null
    }
  }
}

function get_print_template {
  if ($null -eq $global:_print_template) {
    $value = $(env_get_print_template)
    if ($null -ne $value) {
      set_print_template $value
    }
    else {
      set_print_template $(default_print_template)
    }
  }
  write-debug "<get_print_template> -> ${global:_print_template}."
  return $global:_print_template
}

function set_print_template([bool] $value) {
  write-debug "<set_print_template> value: ${value}."

  $global:_print_template = $value
  env_set_print_template $value
}

# === end common.ps1 extensions ===

function initialize_test([string[]]$params, [object[]] $tests) {
  if (($null -eq $params) -or ($null -eq $tests)) {
    write-error 'usage: initialize_test {params} {tests}'
    write-error ' {params} list of argument passed to the script.'
    write-error ' {tests} list of test objects.'
    write-error ' '
    usage_exit 'initialize_test {params} {tests}.'
  }

  write-debug "<initialize_test> params: [$(len $params)]."
  write-debug "<initialize_test> tests: [$(len $tests)]."

  $command = $null
  $is_debug = $false
  $is_verbosity_specified = $false
  $test_filter = @()

  if (0 -eq $(len $params)) {
    write-title './test-chart <command> [<options>]'
    write-high 'commands:'
    write-normal '  list            Prints a list of available tests and quits.'
    write-normal '  test            Executes available tests. (default)'
    write-normal ''
    write-high 'options:'
    write-normal '  --print|-p      Prints the output of the ''helm template'' command to the terminal.'
    write-normal '  -t:<test>       Specifies which tests to run. When not provided all tests will be run.'
    write-normal '                  Use ''list'' to determine which tests are available.'
    write-normal '  -v:<verbosity>  Enables verbose output from the test scripts.'
    write-normal '                  verbosity:'
    write-normal '                    minimal|m:  Sets build-system verbosity to minimal. (default)'
    write-normal '                    normal|n:   Sets build-system verbosity to normal.'
    write-normal '                    detailed|d: Sets build-system verbosity to detailed.'
    write-normal '  --debug         Enables verbose build script tracing; this has no effect on build-system verbosity.'
    write-normal ''
    cleanup_after
    exit 0
  }

  for ($i = 0 ; $i -lt $(len $params) ; $i += 1) {
    $arg = $params[$i]
    $arg2 = $null
    $pair = $arg -split ':'

    if ($(len $pair) -gt 1) {
      $arg = $pair[0]
      if ($(len $pair[1]) -gt 0) {
        $arg2 = $pair[1]
      }
    }

    if ($i -eq 0) {
      if ('list' -ieq $arg)
      {
        $command = 'LIST'
        continue
      }
      elseif ('test' -ieq $arg) {
        $command = 'TEST'
        continue
      }
      else {
        $command = 'TEST'
      }
    }

    if ('--debug' -ieq $arg) {
      $is_debug = $true
    }
    elseif (('--print' -ieq $arg) -or ('-p' -ieq $arg)) {
      if ('TEST' -ne $command) {
        usage_exit "Option '${arg}' not supported by command 'list'."
      }
      if (get_print_template) {
        usage_exit "Option '${arg}' already specified."
      }
      set_print_template($true)
    }
    elseif (('--test' -ieq $arg) -or ('-t' -ieq $arg))
    {
      if ($null -eq $arg2)
      {
        if ($i + 1 -ge $(len $params)) {
          usage_exit "Expected value following ""{$arg}""."
        }

        $i += 1
        $test_name = $params[$i]
      }
      else
      {
        $test_name = $arg2
      }

      $test_found = $false

      $parts = $test_name.split('/')
      if ($(len $parts) -gt 1) {
        $test_name = $parts[$(len $parts) - 1]
      }

      foreach ($test in $tests) {
        if ($test.name -ieq $test_name) {
          $test_found = $true
          break
        }
      }

      if (-not $test_found) {
        usage_exit "Unknown test name ""${test_name}"" provided."
      }

      $test_filter += $test_name
    }
    elseif (('--verbosity' -ieq $arg) -or ('-v' -ieq $arg)) {
      if ($null -eq $arg2)
      {
        if ($i + 1 -ge $(len $params)) {
          usage_exit "Expected value following ""{$arg}""."
        }

        $i += 1
        $value = $params[$i]
      }
      else
      {
        $value = $arg2
      }

      if (('minimal' -ieq $value) -or ('m' -ieq $value)) {
        $verbosity = 'MINIMAL'
      }
      elseif (('normal' -ieq $value) -or ('n' -ieq $value)) {
        $verbosity = 'NORMAL'
      }
      elseif (('detailed' -ieq $value) -or ('d' -ieq $value)) {
        $verbosity = 'DETAILED'
      }
      else {
        usage_exit "Invalid verbosity option ""${arg}""."
      }

      $(set_verbosity $verbosity)
      $is_verbosity_specified = $true
    }
    else {
      usage_exit "Unknown option '${arg}'."
    }
  }

  $is_debug = $is_debug -or $(get_is_debug)
  set_is_debug $is_debug

  $tests_path = split-path -parent $myinvocation.pscommandpath
  $root_path = split-path -parent $tests_path
  $chart_path = join-path $root_path 'chart'

  if (-not $(test-path $root_path)) {
    fatal_exit "Expected path '${root_path}' not found or inaccessible."
  }
  if (-not $(test-path $chart_path)) {
    fatal_exit "Expected path '${chart_path}' not found or inaccessible."
  }
  if (-not $(test-path $tests_path)) {
    fatal_exit "Expected path '${tests_path}' not found or inaccessible."
  }

  write-debug "<initialize_test> root_path = '${root_path}'."
  write-debug "<initialize_test> chart_path = '${chart_path}'."
  write-debug "<initialize_test> tests_path = '${tests_path}'."

  # When a subset of tests has been requested, filter out the not requested tests.
  if ($(len $test_filter) -gt 0) {
    write-debug "<initialize_test> selected: [$(len $test_filter)]."

    $replace = @()

    # Find the test that matches each selected item and add it to a replacement list.
    foreach ($filter in $test_filter) {
      foreach ($test in $tests) {
        if ($test.name -ieq $filter) {
          $replace += $test
          break
        }
      }
    }

    # Replace the test list with the replacement list.
    $tests = $replace
    write-debug "<initialize_test> tests = [$(len $tests)]."
  }

  if ((-not $is_verbosity_specified) -and (-not $(is_tty))) {
    write-debug "<initialize_test> override verbosity with 'detailed' when TTY not detected."
    set_verbosity 'DETAILED'
  }

  return @{
    chart_path = $root_path
    command = $command
    tests = $tests
  }
}

function list_helm_tests([object] $config) {
  if ($null -eq $config.tests) {
    write-error 'usage: list_helm_tests {config}' -category InvalidArgument
    write-error ' {config} configuration object returned by `initialize_test`.'
    write-error ' '
    usage_exit 'list_helm_tests {config}.'
  }
  if (($null -eq $config.tests) -or ($null -eq $config.command) -or ($null -eq $config.chart_path)) {
    fatal_exit 'invalid configuration object received.'
  }

  write-debug "<list_helm_tests> config.chart_path = '$($config.chart_path)'."
  write-debug "<list_helm_tests> config.command = '$($config.command)'."
  write-debug "<list_helm_tests> config = [$(len $config.tests)]"

  if ('LIST' -ne $config.command) {
    throw "List method called when command was 'test'."
  }

  write-title "Available tests:"

  foreach ($test in $config.tests) {
    if ('MINIMAL' -ne $(get_verbosity)) {
      write-high "- $($test.name):"

      if ('DETAILED' -eq $(get_verbosity)) {
        write-detailed '  matches:'
        if (len $test.matches -gt 0) {
          foreach ($match in $test.matches) {
            $regex = generate_regex $match
            write-low "    ${regex}"
          }
        }
        else {
          write-low '    <none>'
        }

        write-detailed '  options:'
        if (len $test.options -gt 0) {
          foreach ($option in $test.options) {
            write-low "    ${option}"
          }
        }
        else{
          write-low '    <none>'
        }
      }
      else {
        $matches_count = 0
        if (($null -ne $test.matches)) {
          $matches_count = $(len $test.matches)
        }
        $options_count = 0
        if (($null -ne $test.options)) {
          $options_count = $(len $test.options)
        }
        write-normal "  matches: ${matches_count}"
        if ($options_count -gt 0) {
          write-normal "  options: " -no_newline
          write-normal "${options_count}" $global:colors.low
        }
      }

      write-normal '  values:'
      if ($(len $test.values) -gt 0) {
        foreach($value in $test.values) {
          write-normal "    ${value}"
        }
      }
      else {
        write-normal '    <none>'
      }
    }
    else {
      write-minimal "- $($test.name)"
    }
  }

  $(cleanup_after)
}

function test_helm_chart([object] $config) {
  write-debug "<test_helm_chart> config.chart_path = '$($config.chart_path)'."
  write-debug "<test_helm_chart> config.command = '$($config.command)'."
  write-debug "<test_helm_chart> config = [$(len $config.tests)]."

  if ('LIST' -eq $config.command) {
    list_helm_tests $config
    return $true
  }

  $timer = [System.Diagnostics.Stopwatch]::StartNew()

  push-location $config.chart_path

  try {
    $fail_count = 0
    $pass_count = 0
    $total_fail_checks = 0
    $total_pass_checks = 0

    foreach ($test in $config.tests) {
      $fail_checks = 0
      $pass_checks = 0

      $values_path = resolve-path $(join-path 'chart' 'values.yaml') -relative
      write-debug "<test_helm_chart> values_path = '${values_path}'."

      $helm_command = "helm template test -f ${values_path}"
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      # First add all values files to the command.
      if ($(len $test.values) -gt 0) {
        foreach ($value in $test.values) {
          write-debug "<test_helm-chart> value = '${value}'."
          $values_path = $(resolve-path $(join-path 'tests' $value) -relative)
          write-debug "<test_helm_chart> values_path = '${values_path}'."
          $helm_command = "${helm_command} -f $values_path"
        }
        write-debug "<test_helm_chart> helm_command = '${helm_command}'."
      }

      # Second add all --set options to the command.
      if ($(len $test.options) -gt 0) {
        foreach ($option in $test.options) {
          write-debug "<test_helm_chart> option = '${option}'."
          $helm_command = "${helm_command} --set `"${option}`""
        }
      }

      $helm_command = "${helm_command} ./chart/."
      write-debug "<test_helm_chart> helm_command = '${helm_command}'."

      $captured = invoke-expression "${helm_command} 2>&1" | out-string
      $exit_code = $LASTEXITCODE
      write-debug "<test_helm_chart> expected = $($test.expected)."
      write-debug "<test_helm_chart> actual = ${exit_code}."

      $is_pass = $test.expected -eq $exit_code

      if (-not $is_pass) {
        write-normal ">> Failed: exit code ${exit_code} did not match expected $($test.expected)."  $global:colors.low
        # When the exit code is an unexpected non-zero value, print Helm's output.
        if ($exit_code -ne 0)
        {
          # Disable template printing to avoid a double print.
          set_print_template $false
          write-minimal "Helm Template Output" $global:colors.high
          write-minimal $captured $global:colors.low
        }
      }

      if ($(len $test.matches) -gt 0) {
        foreach ($match in $test.matches) {
          $regex = generate_regex $match

          write-debug "<test_helm_chart> regex = '${regex}'."
          $is_match = $captured -match $regex
          write-debug "<test_helm_chart> is_match = ${is_match}."

          if (-not $is_match) {
            write-normal ">> Failed: output did not match: ""${regex}""." $global:colors.low
          }

          if ($is_match) {
            $pass_checks += 1
          }
          else {
            $fail_checks += 1
            $is_pass = $false
          }
        }
      }

      $total_fail_checks += $fail_checks
      $total_pass_checks += $pass_checks

      if (get_print_template) {
        write-normal "Helm Template Output" $global:colors.high
        write-normal $captured $global:colors.low
      }

      if ($is_pass) {
        $pass_count += 1
        write-passed "$($test.name) (passed ${pass_checks} of $($fail_checks + $pass_checks) checks)"
      }
      else {
        $fail_count += 1
        write-failed "$($test.name) (failed ${fail_checks} of $($fail_checks + $pass_checks) checks)"
        write-low "  command: $($config.chart_path)> ${helm_command}"
      }
    }
  }
  catch {
    pop-location

    throw $_
  }

  pop-location

  $timer.stop()

  if ($fail_count -gt 0) {
    write-minimal "Failed: ${fail_count}" $global:colors.test.failed -no_newline
    write-normal ", Passed: ${pass_count} ($total_pass_checks) [${total_fail_checks}]" $global:colors.test.failed -no_newline
    write-minimal ", Passed: ${pass_count} ($total_pass_checks)" $global:colors.test.failed -no_newline
    write-normal ", Tests: $(len $config.tests) [$($total_fail_checks + $total_pass_checks)]" $global:colors.test.failed -no_newline
    write-normal " ($('{0:0.000}' -f $timer.elapsed.totalseconds) seconds)" $global:colors.low -no_newline
    write-minimal ''
    return $false
  }
  else
  {
    write-minimal "Passed: ${pass_count}" $global:colors.test.passed -no_newline
    write-normal ", Tests: $(len $config.tests) [${total_pass_checks}]" $global:colors.test.passed -no_newline
    write-minimal ", Tests: $(len $config.tests)" $global:colors.test.passed -no_newline
    write-normal ", [$($total_fail_checks + $total_pass_checks)]" $global:colors.test.passed -no_newline
    write-normal " ($('{0:0.000}' -f $timer.elapsed.totalseconds) seconds)" $global:colors.low
    write-minimal ''
    return $true
  }

  $(cleanup_after)
}

function generate_regex([object] $match) {
  $regex = ''
  if ('hashtable' -eq $(typeof $match)) {
    write-debug "<generate_regex> match is hashtable"
    write-debug "<generate_regex> indent: $($match.indent)."
    write-debug "<generate_regex> match.lines: [$(len $match.lines)]."

    if ($match.indent -gt 0) {
      $prefix = "\s{$($match.indent)}"
    }
    else {
      $prefix = ''
    }

    foreach ($line in $match.lines) {
      $line = $line -replace '([\.\*\+\?\^\$\{\}\(\)|[\]\\])', '\$1'
      $regex = "${regex}${prefix}${line}\s*[\n\r]{1,2}"
    }
  }
  else {
    $regex = $match
  }

  write-debug "<generate_regex> -> '${regex}'."
  return $regex
}
