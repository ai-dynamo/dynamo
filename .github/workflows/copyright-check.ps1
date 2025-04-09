# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

. "$(& git rev-parse --show-toplevel)/.github/workflows/common.ps1"

# == begin common.ps1 extensions ==

$date_key = '%%DATE%%'
$date_regex = '(?>(?>\d{4})-)?(?<year>\d{4})'

$timer = [System.Diagnostics.Stopwatch]::StartNew()

$global:copyright_matchers = @(
  @{
    files = @('.containerfile', '.dockerignore', '.pbtxt', '.ps1', '.py', '.sh', '.toml', '.tpl', '.txt', '.yaml', '.yml', 'Dockerfile')
    found_missing = $false
    matches = @(
      '# SPDX-FileCopyrightText: Copyright (c) ' + $date_key + ' NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
      '# SPDX-License-Identifier: Apache-2.0'
      '# Licensed under the Apache License, Version 2.0 (the "License");'
      '# you may not use this file except in compliance with the License.'
      '# You may obtain a copy of the License at'
      '# http://www.apache.org/licenses/LICENSE-2.0'
      '# Unless required by applicable law or agreed to in writing, software'
      '# distributed under the License is distributed on an "AS IS" BASIS,'
      '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'
      '# See the License for the specific language governing permissions and'
      '# limitations under the License.'
    )
    name = 'basic'
    regex = $null
    vertical_spacer = '#'
  }
  @{
    files = @('.json')
    found_missing = $false
    matches = @(
      '"copyright": ['
      '  "SPDX-FileCopyrightText: Copyright (c) ' + $date_key + ' NVIDIA CORPORATION & AFFILIATES. All rights reserved.",'
      '  "SPDX-License-Identifier: Apache-2.0",'
      '  "Licensed under the Apache License, Version 2.0 (the \"License\");",'
      '  "you may not use this file except in compliance with the License.",'
      '  "You may obtain a copy of the License at",'
      '  "http://www.apache.org/licenses/LICENSE-2.0",'
      '  "Unless required by applicable law or agreed to in writing, software",'
      '  "distributed under the License is distributed on an \"AS IS\" BASIS,",'
      '  "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",'
      '  "See the License for the specific language governing permissions and",'
      '  "limitations under the License."'
      '],'
    )
    name = 'json'
    regex = $null
    vertical_spacer = $null
  }
  @{
    files = @('.md')
    found_missing = $false
    matches = @(
      '<!--'
      'SPDX-FileCopyrightText: Copyright (c) ' + $date_key + ' NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
      'SPDX-License-Identifier: Apache-2.0'
      'Licensed under the Apache License, Version 2.0 (the "License");'
      'you may not use this file except in compliance with the License.'
      'You may obtain a copy of the License at'
      'http://www.apache.org/licenses/LICENSE-2.0'
      'Unless required by applicable law or agreed to in writing, software'
      'distributed under the License is distributed on an "AS IS" BASIS,'
      'WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'
      'See the License for the specific language governing permissions and'
      'limitations under the License.'
      '-->'
    )
    name = 'markdown'
    regex = $null
    vertical_spacer = ''
  }
  @{
    files = @('.proto', '.rs')
    found_missing = $false
    matches = @(
      '// SPDX-FileCopyrightText: Copyright (c) ' + $date_key + ' NVIDIA CORPORATION & AFFILIATES. All rights reserved.'
      '// SPDX-License-Identifier: Apache-2.0'
      '// Licensed under the Apache License, Version 2.0 (the "License");'
      '// you may not use this file except in compliance with the License.'
      '// You may obtain a copy of the License at'
      '// http://www.apache.org/licenses/LICENSE-2.0'
      '// Unless required by applicable law or agreed to in writing, software'
      '// distributed under the License is distributed on an "AS IS" BASIS,'
      '// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.'
      '// See the License for the specific language governing permissions and'
      '// limitations under the License.'
    )
    name = 'c-like'
    regex = $null
    vertical_spacer = '//'
  }
)
$global:copyright_results = @{
  failed_date = @()
  failed_header = @()
  passed = @()
  skipped = @()
  unsupported = @()
}

# === end common.ps1 extensions ===

$ignored_files = @('.clang-format', '.gitattributes', '.gitignore', '.gitkeep', '.patch', 'Cargo.lock', 'LICENSE', 'uv.lock', 'rust-toolchain.toml', 'codespell.txt', 'exclusions.txt')
write-debug "<copyright-check> ignored_files = ['$($ignored_files -join "','")']."
$ignored_paths = @('.github', '.mypy_cache', '.pytest_cache', 'lib/llm/tests/data/sample-models')
write-debug "<copyright-check> ignored_paths = ['$($ignored_paths -join "','")']."
$ignored_types = @('.bat', '.gif', '.ico', '.ipynb', '.jpg', '.jpeg', '.patch', '.png', '.pyc', '.pyi', '.rst', '.zip', '.md', '.json')
write-debug "<copyright-check> ignored_types = ['$($ignored_types -join "', '")']."
$ignored_folders = @('.git', '__pycache__')

function is_ignored([string] $path) {
  # write-debug "<copyright-check/is_ignored> path: \"${path}\"."
  if (($null -eq $path) -or ($path.length -eq 0) -or ($path.endswith('/'))) {
    write-debug "<copyright-check/is_ignored> ignored: true."
    return $true
  }

  foreach ($ignored_path in $ignored_paths) {
    if ($path.startswith($ignored_path)) {
      write-debug "<copyright-check/is_ignored> ignored: true."
      return $true
    }
  }

  foreach ($ignored_extension in $ignored_types) {
    if ($file.endswith($ignored_extension)) {
      write-debug "<copyright-check/is_ignored> ignore = true."
      return $true
    }
  }

  foreach ($ignored_file in $ignored_files) {
    if ($file.endswith($ignored_file)) {
      write-debug "<copyright-check/is_ignored> ignore = true."
      return $true
    }
  }

  $normalized_path = normalize_path $file

  foreach ($ignored_folder in $ignored_folders) {
    if ($normalized_path -contains "/${ignored_folder}/") {
      write-debug "<copyright-check/is_ignored> ignore = true."
      return $true
    }
  }

  if (-not(test-path "${normalized_path}" -pathtype 'Leaf')) {
    write-debug "<copyright-check/is_ignored> ignore = true."
    return $true
  }

  write-debug "<copyright-check/is_ignored> ignore = false."
    return $false
}

function build_regex([object] $matcher) {
  write-debug "<copyright-check/build_regex> matcher.name: $($matcher.name)."

  $regex = ''
  foreach ($match in $matcher.matches) {
    $match = $match -replace '([\(\)\[\]\.\+\*\\])', '\$1'
    $match = $match -replace '\s+', '\s+'
    # Given the amount of inconsistency between using http and https, we'll just regex it away.
    $match = $match -replace 'https?://', 'https?://'
    # Replace the date matcher placeholder w/ the actual regex we'll need.
    $match = $match -replace $date_key, $date_regex

    $regex = "${regex}${match}[\n\r\s]+"
    if ($null -ne $matcher.vertical_spacer) {
      $regex = "${regex}(?>$($matcher.vertical_spacer)[\n\r\s]+)*"
    }
  }

  write-debug "<copyright-check/build_regex> -> '${regex}'."
  return $regex
}

function check_header([string] $path, [object] $matcher) {
  write-debug "<copyright-check/check_header> path: ""${path}""."
  write-debug "<copyright-check/check_header> matcher: ""$($matcher.name)""."

  $command = "git log -1 --pretty=""%cs"" -- ${file}"
  $output = invoke-expression $command | out-string
  $output = $output.trim()
  $last_modified = $output.substring(0, 4) -as [int]

  write-debug "<copyright-check/check_header> last_modified: ${last_modified}."

  if ($null -eq $matcher.regex) {
    $matcher.regex = $(build_regex $matcher)
  }
  $regex = $matcher.regex

  write-debug "<copyright-check/check_header> regex: ""${regex}""."

  $contents = read_content $path
  if (($null -eq $contents) -or ($contents.length -le 0)) {
    $global:copyright_results.skipped += $file
    write-detailed "  [SKIP] ${file}" 'DarkGray'
    return
  }

  if ($contents -match $regex) {
    $capture_date = $Matches.year -as [int]

    if ($capture_date -lt $last_modified) {
      $global:copyright_results.failed_date += $file
      write-error "  [FAIL] Incorrect Date in Header: ${path} (${capture_date})"
    }
    else {
      $global:copyright_results.passed += $file
      write-normal "  [PASS] ${file}"
    }
  }
  else {
    $global:copyright_results.failed_header += $file
    write-error "  [FAIL] Invalid/Missing Header: ${file}"
    $matcher.found_missing = $true
  }
}

function check_file([string] $file) {
  write-debug "<copyright-check/check_file> file: ""${file}""."

  $path = normalize_path $file

  if (test-path $path -pathtype 'Leaf') {
    write-debug "<copyright-check/check_file> path: ""${path}""."

    $is_checked = $false
    foreach ($matcher in $global:copyright_matchers)
    {
      foreach ($ext in $matcher.files) {
        if ($path.endswith($ext)) {
          check_header $path $matcher
          $is_checked = $true
          break
        }
      }
      if ($is_checked) {
        return
      }
    }

    write-warning "  [WARN] Unsupported: ${file}"
    $global:copyright_results.unsupported += $file
  }
}

$current_year = "$(get-date -format 'yyyy')" -as [int]
write-debug "<copyright-check> current_year = ${current_year}."

foreach ($file in $(git ls-tree -r --name-only HEAD)) {
  $file = $file.trim()
  write-debug "<copyright-check> file: ""${file}""."

  if (is_ignored $file) {
    write-detailed "  [SKIP] ${file}" 'DarkGray'
    $global:copyright_results.skipped += $file
    continue
  }

  check_file $file
}

function generate_report() {
  $reports_path = $env:NVBUILD_REPORTS_PATH
  if ($null -eq $reports_path) {
    return
  }

  if (-not (test-path $reports_path -pathtype 'Container')) {
    if (test-path $reports_path -pathtype 'Leaf') {
      return
    }
    new-item $reports_path -itemtype 'Directory' | out-null
  }

  write-debug "<copyright-check/generate_report> Generating check report."

  $check_results = "<copyright-verification>`n"

  if ($global:copyright_results.failed_header.count -gt 0) {
    $check_results += "  <failed reason=""Invalid or Missing Header"">`n"

    foreach ($file in $global:copyright_results.failed_header) {
      $check_results += "    <file>${file}</file>`n"
    }

    $check_results += "  </failed>`n"
  }

  if ($global:copyright_results.failed_date.count -gt 0) {
    $check_results += "  <failed reason=""Incorrect Date"">`n"

    foreach ($file in $global:copyright_results.failed_date) {
      $check_results += "    <file>${file}</file>`n"
    }

    $check_results += "  </failed>`n"
  }

  if ($global:copyright_results.unsupported.count -gt 0) {
    $check_results += "  <skipped reason=""Unsupported File Type"">`n"

    foreach ($file in $global:copyright_results.unsupported) {
      $check_results += "    <file>${file}</file>`n"
    }

    $check_results += "  </skipped>`n"
  }

  if ($global:copyright_results.passed.count -gt 0) {
    $check_results += "  <passed>`n"

    foreach ($file in $global:copyright_results.passed) {
      $check_results += "    <file>${file}</file>`n"
    }

    $check_results += "  </passed>`n"
  }

  if ($global:copyright_results.skipped.count -gt 0) {
    $check_results += "  <skipped reason=""Ignored by Path"">`n"

    foreach ($file in $global:copyright_results.skipped) {
      $check_results += "    <file>${file}</file>`n"
    }

    $check_results += "  </skipped>`n"
  }

  $check_results += "</copyright-verification>`n"
  $output_path = "${reports_path}/copyright-check.xml"

  write_content $check_results $output_path -overwrite

  write-minimal ''
  write-minimal "Copyright check report -> ${output_path}"
}
write-normal ''

$timer.Stop()

write-high "Pass: $($global:copyright_results.passed.count), Fail: $($global:copyright_results.failed_date.count + $global:copyright_results.failed_header.count)" -no_newline
if ($global:copyright_results.skipped.count -gt 0) {
  write-high ", Skipped: $($global:copyright_results.skipped.count)" -no_newline
}
if ($global:copyright_results.unsupported.count -gt 0) {
  write-high ", Unsupported: $($global:copyright_results.unsupported.count)" -no_newline
}
write-minimal " ($($timer.Elapsed.TotalSeconds.ToString("0.000")) seconds)" $global:colors.low -no_newline
write-minimal ''

if ($global:copyright_results.failed_header.count -gt 0) {
  write-low ''
  write-low 'Copyright checkers detected missing or invalid copyright headers:'
  write-low ''
  foreach ($matcher in $global:copyright_matchers) {
    if ($matcher.found_missing) {
      write-low "  name: $($matcher.name)"
      write-low "  files: $($matcher.files -join ", ")"
      write-low "  pattern:`n    $($matcher.regex)"
      write-low ''
    }
  }
}


if (($global:copyright_results.failed_date.count -gt 0) -or ($global:copyright_results.failed_header.count -gt 0)) {
  write-high 'Files out of compliance:'
  # Final, end of output list of errors.
  foreach ($path in $global:copyright_results.failed_header) {
    write-error " [FAIL] invalid/missing header: ${path}"
  }
  foreach ($path in $global:copyright_results.failed_date) {
    write-error " [FAIL] incorrect date: ${path}"
  }
  exit(-1)
}
