#!/usr/bin/env node
/**
 * Test script for .github/filters.yaml pattern matching.
 * Reads patterns directly from filters.yaml and validates behavior.
 *
 * Usage:
 *   cd .github/scripts
 *   npm install
 *   npm test                    # Run pattern tests only
 *   npm run coverage            # Check full repo coverage
 *   npm test -- --coverage      # Run both
 *
 * This validates that tj-actions/changed-files will correctly:
 * - Match backend-specific files to their respective filters (vllm, sglang, trtllm)
 * - Route docs-artifact tests to recipe-check without core E2E checks
 * - Route sidecar files to sidecar/Rust checks without backend or core E2E checks
 * - Exclude doc files (*.md, *.rst, *.txt) from core via negation patterns
 * - Match CI/infrastructure changes to core
 * - (with --coverage) Ensure all files in repo are covered by at least one filter
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const micromatch = require('micromatch');
const YAML = require('yaml');

const runCoverage = process.argv.includes('--coverage');

// Find filters.yaml relative to this script
const scriptDir = path.dirname(__filename);
const filtersPath = path.resolve(scriptDir, '../filters.yaml');

console.log(`Reading filters from: ${filtersPath}\n`);

// Parse YAML (handles anchors/aliases automatically)
const filtersYaml = fs.readFileSync(filtersPath, 'utf8');
const filters = YAML.parse(filtersYaml);

// Flatten nested arrays (YAML anchors create nested arrays)
function flattenPatterns(patterns) {
  if (!patterns || !Array.isArray(patterns)) return [];
  return patterns.flat(Infinity).filter(p => typeof p === 'string');
}

// Simulate tj-actions/changed-files behavior with negation
function checkFilter(file, patterns) {
  const flat = flattenPatterns(patterns);
  if (flat.length === 0) return false;

  const positive = flat.filter(p => !p.startsWith('!'));
  const negative = flat.filter(p => p.startsWith('!')).map(p => p.slice(1));

  const matchesPositive = micromatch.isMatch(file, positive);
  const matchesNegative = negative.length > 0 && micromatch.isMatch(file, negative);

  return matchesPositive && !matchesNegative;
}

// Test cases: [file, expectations, description]
// expectations: { filterName: expectedValue, ... }
const testCases = [
  // dev/local-dev templates build only the dev images -- they must NOT pull in
  // `core` (all runtime builds + the GPU test matrix), and must not be silently
  // uncovered the way they were when they sat in `ignore`.
  {
    file: 'container/templates/dev.Dockerfile',
    expect: { dev_images: true, core: false },
    desc: 'dev.Dockerfile triggers dev image builds only'
  },
  {
    file: 'container/templates/local_dev.Dockerfile',
    expect: { dev_images: true, core: false },
    desc: 'local_dev.Dockerfile triggers dev image builds only'
  },
  // Backend-specific files should only trigger their backend
  {
    file: 'examples/backends/vllm/launch/dsr1_dep.sh',
    expect: { core: false, vllm: true, sglang: false, trtllm: false },
    desc: 'vllm script triggers only vllm'
  },
  {
    file: 'examples/backends/sglang/example.py',
    expect: { core: false, vllm: false, sglang: true, trtllm: false },
    desc: 'sglang script triggers only sglang'
  },
  {
    file: 'examples/backends/trtllm/example.py',
    expect: { core: false, vllm: false, sglang: false, trtllm: true },
    desc: 'trtllm script triggers only trtllm'
  },
  {
    file: 'recipes/qwen3-32b/vllm/cloud-providers/.kustomize-matrix.yaml',
    expect: { core: false, examples: true },
    desc: 'recipe matrix dotfile triggers recipe check without core'
  },
  {
    file: 'scripts/kustomize-matrix.py',
    expect: { core: false, examples: true },
    desc: 'recipe generator triggers recipe check without core'
  },
  {
    file: 'tests/test_kustomize_matrix.py',
    expect: { core: false, examples: true },
    desc: 'recipe generator test triggers recipe check without core'
  },
  {
    file: 'docs/tests/nested/__ci_filter_probe__',
    expect: { core: false, docs: true, examples: true },
    desc: 'any docs/tests descendant triggers recipe check without core'
  },
  {
    file: 'components/src/dynamo/vllm/worker.py',
    expect: { core: false, vllm: true },
    desc: 'vllm component triggers only vllm'
  },

  // Sidecar Rust and proto files should trigger Rust checks without unrelated E2E
  {
    file: 'lib/sidecar/common/src/lib.rs',
    expect: { sidecar: true, rust: true, core: false, frontend: false, vllm: false, sglang: false, trtllm: false },
    desc: 'common sidecar source avoids unrelated build and E2E filters'
  },
  {
    file: 'lib/sidecar/vllm/proto/vllm_grpc.proto',
    expect: { sidecar: true, rust: true, core: false, frontend: false, vllm: false, sglang: false, trtllm: false },
    desc: 'vllm sidecar proto triggers Rust checks without backend E2E'
  },
  {
    file: 'lib/sidecar/sglang/src/lib.rs',
    expect: { sidecar: true, rust: true, core: false, frontend: false, vllm: false, sglang: false, trtllm: false },
    desc: 'sglang sidecar source avoids backend E2E'
  },
  {
    file: 'lib/sidecar/trtllm/src/lib.rs',
    expect: { sidecar: true, rust: true, core: false, frontend: false, vllm: false, sglang: false, trtllm: false },
    desc: 'trtllm sidecar source does not route to sglang or trtllm E2E'
  },
  {
    file: 'lib/sidecar/vllm/deploy/agg.yaml',
    expect: { sidecar: true, rust: false, core: false, frontend: false, vllm: false, sglang: false, trtllm: false },
    desc: 'sidecar deployment config avoids Rust and E2E checks'
  },
  {
    file: 'lib/sidecar/README.md',
    expect: { sidecar: false, ignore: true, rust: false, core: false, frontend: false, docs: false, vllm: false, sglang: false, trtllm: false },
    desc: 'sidecar README is classification-only and triggers no build'
  },
  {
    file: '.github/workflows/shared-build-sidecar.yml',
    expect: { sidecar: true },
    desc: 'sidecar workflow triggers its own job'
  },
  {
    file: 'container/compliance/policy/licenses.toml',
    expect: { sidecar: true },
    desc: 'compliance policy changes validate the sidecar image gate'
  },

  // Doc files should be excluded from core (negation patterns)
  {
    file: 'lib/README.md',
    expect: { core: false, vllm: false, docs: true },
    desc: 'lib README excluded from core, matches docs'
  },
  {
    file: 'tests/README.md',
    expect: { core: false, docs: true },
    desc: 'tests README excluded from core'
  },
  {
    file: 'lib/docs/guide.txt',
    expect: { core: false, docs: true },
    desc: 'txt file excluded from core'
  },
  {
    file: 'docs/guide.md',
    expect: { core: false, docs: true },
    desc: 'docs folder matches docs filter'
  },

  // Code files should trigger core
  {
    file: 'lib/runtime/src/main.rs',
    expect: { core: true, vllm: false },
    desc: 'rust file triggers core'
  },
  {
    file: 'lib/runtime/Cargo.toml',
    expect: { core: true },
    desc: 'Cargo.toml triggers core'
  },
  {
    file: 'tests/test_something.py',
    expect: { core: true },
    desc: 'python test triggers core'
  },
  {
    file: 'components/src/dynamo/router/router.py',
    expect: { core: true },
    desc: 'router triggers core'
  },
  {
    file: 'components/src/dynamo/frontend/server.py',
    expect: { core: true },
    desc: 'frontend triggers core'
  },

  // CI files should trigger core
  {
    file: '.github/workflows/ci.yml',
    expect: { core: true },
    desc: 'workflow triggers core'
  },
  {
    file: '.github/filters.yaml',
    expect: { core: true },
    desc: 'filters.yaml triggers core'
  },
  {
    file: '.github/actions/docker-build/action.yml',
    expect: { core: true },
    desc: 'action triggers core'
  },

  // Root level files
  {
    file: 'pyproject.toml',
    expect: { core: true },
    desc: 'root toml triggers core'
  },
  {
    file: 'setup.py',
    expect: { core: true },
    desc: 'root py triggers core'
  },

  // Operator and deploy
  {
    file: 'deploy/operator/cmd/main.go',
    expect: { core: false, operator: true },
    desc: 'operator file triggers operator'
  },
  {
    file: 'deploy/helm/charts/platform/values.yaml',
    expect: { core: false, deploy: true },
    desc: 'helm file triggers deploy'
  },

  // Framework snapshot lifecycle: backend filter + that framework's checkpoint filter
  {
    file: 'components/src/dynamo/vllm/snapshot.py',
    expect: {
      vllm: true,
      snapshot: false,
      snapshot_vllm: true,
      snapshot_sglang: false,
      snapshot_trtllm: false,
    },
    desc: 'vllm snapshot.py gates only vllm checkpoint tests'
  },
  {
    file: 'components/src/dynamo/sglang/snapshot.py',
    expect: {
      sglang: true,
      snapshot: false,
      snapshot_vllm: false,
      snapshot_sglang: true,
      snapshot_trtllm: false,
    },
    desc: 'sglang snapshot.py gates only sglang checkpoint tests'
  },
  {
    file: 'components/src/dynamo/trtllm/snapshot.py',
    expect: {
      trtllm: true,
      snapshot: false,
      snapshot_vllm: false,
      snapshot_sglang: false,
      snapshot_trtllm: true,
    },
    desc: 'trtllm snapshot.py gates only trtllm checkpoint tests'
  },
  {
    file: 'components/src/dynamo/trtllm/tests/test_trtllm_snapshot.py',
    expect: { trtllm: true, snapshot_trtllm: true, snapshot: false },
    desc: 'trtllm snapshot unit test gates only trtllm checkpoint tests'
  },
  {
    file: 'components/src/dynamo/common/snapshot/lifecycle.py',
    expect: { snapshot: true, snapshot_vllm: false, core: true },
    desc: 'shared snapshot lifecycle triggers shared snapshot filter'
  },
  {
    file: 'tests/deploy/test_dynamocheckpoint.py',
    expect: { snapshot: true, deploy: true },
    desc: 'Checkpoint deploy test triggers shared snapshot integration'
  },
  {
    file: 'deploy/operator/api/v1beta1/dynamographdeployment_types.go',
    expect: { snapshot: true, operator: true },
    desc: 'Operator API changes trigger snapshot contract tests'
  },
  {
    file: 'deploy/operator/config/crd/bases/nvidia.com_dynamographdeployments.yaml',
    expect: { snapshot: true, operator: true },
    desc: 'Operator CRD changes trigger snapshot contract tests'
  },
  {
    file: 'deploy/operator/config/rbac/role.yaml',
    expect: { snapshot: true, operator: true },
    desc: 'Operator RBAC changes trigger snapshot contract tests'
  },
  {
    file: 'deploy/operator/api/v1beta1/dynamomodel_types.go',
    expect: { snapshot: false, operator: true },
    desc: 'Unrelated operator APIs do not trigger snapshot integration'
  },
  {
    file: 'deploy/operator/config/crd/bases/nvidia.com_dynamomodels.yaml',
    expect: { snapshot: false, operator: true },
    desc: 'Unrelated operator CRDs do not trigger snapshot integration'
  },
  {
    file: 'deploy/operator/go.mod',
    expect: { snapshot: true, operator: true },
    desc: 'Operator dependency changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/consts/consts.go',
    expect: { snapshot: true, operator: true },
    desc: 'Restore protocol constants trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/features/gates.go',
    expect: { snapshot: true, operator: true },
    desc: 'Snapshot API feature detection triggers snapshot integration'
  },
  {
    file: 'deploy/operator/internal/controller/dgd_component_workloads_reconciler.go',
    expect: { snapshot: true, operator: true },
    desc: 'DGD workload reconciliation changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/controller/dynamographdeployment_controller.go',
    expect: { snapshot: true, operator: true },
    desc: 'DGD controller changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/controller/dynamocomponentdeployment_renderer.go',
    expect: { snapshot: true, operator: true },
    desc: 'DCD rendering changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/controller/gms_pod_replacement_controller.go',
    expect: { snapshot: true, operator: true },
    desc: 'GMS replacement changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/controller/failover_cascade_controller.go',
    expect: { snapshot: true, operator: true },
    desc: 'Failover cascade behavior triggers snapshot integration'
  },
  {
    file: 'deploy/operator/internal/webhook/mutation/pod_checkpoint_restore_handler.go',
    expect: { snapshot: true, operator: true },
    desc: 'Restore admission changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/webhook/setup/setup.go',
    expect: { snapshot: true, operator: true },
    desc: 'Restore webhook registration triggers snapshot integration'
  },
  {
    file: 'deploy/operator/internal/webhook/validation/shared_v1beta1.go',
    expect: { snapshot: true, operator: true },
    desc: 'Checkpoint API validation triggers snapshot integration'
  },
  {
    file: 'deploy/operator/internal/podcache/transform.go',
    expect: { snapshot: true, operator: true },
    desc: 'Pod cache projection changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/gms/gms.go',
    expect: { snapshot: true, operator: true },
    desc: 'GMS compatibility changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/hash.go',
    expect: { snapshot: true, operator: true },
    desc: 'Worker compatibility hash changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/graph.go',
    expect: { snapshot: true, operator: true },
    desc: 'Graph workload shaping changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/component_worker.go',
    expect: { snapshot: true, operator: true },
    desc: 'Worker Pod shaping changes trigger snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/failover.go',
    expect: { snapshot: true, operator: true },
    desc: 'Failover Pod shaping changes trigger snapshot integration'
  },
  {
    file: 'components/src/dynamo/vllm/handlers.py',
    expect: { snapshot: false, snapshot_vllm: true, vllm: true },
    desc: 'vLLM pause and resume changes trigger vLLM snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/backend_vllm.go',
    expect: { snapshot: false, snapshot_vllm: true, operator: true },
    desc: 'vLLM Pod rendering changes trigger vLLM snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/backend_sglang.go',
    expect: { snapshot: false, snapshot_sglang: true, operator: true },
    desc: 'SGLang Pod rendering changes trigger SGLang snapshot integration'
  },
  {
    file: 'deploy/operator/internal/dynamo/backend_trtllm.go',
    expect: { snapshot: false, snapshot_trtllm: true, operator: true },
    desc: 'TensorRT-LLM Pod rendering changes trigger TensorRT-LLM snapshot integration'
  },
  {
    file: 'deploy/helm/charts/platform/components/operator/templates/deployment.yaml',
    expect: { snapshot: true, deploy: true, operator: true },
    desc: 'Operator Helm changes trigger snapshot integration'
  },
];

// Print available filters
console.log('Loaded filters:', Object.keys(filters).join(', '));
console.log('');

console.log('Testing filter patterns\n');
console.log('File                                           | Result');
console.log('-----------------------------------------------|--------');

let passed = 0;
let failed = 0;

testCases.forEach(({ file, expect, desc }) => {
  const results = {};
  let allMatch = true;

  // Check each expected filter
  for (const [filterName, expectedValue] of Object.entries(expect)) {
    const actual = checkFilter(file, filters[filterName]);
    results[filterName] = actual;
    if (actual !== expectedValue) {
      allMatch = false;
    }
  }

  if (allMatch) {
    passed++;
    const matchedFilters = Object.entries(results)
      .filter(([_, v]) => v)
      .map(([k, _]) => k)
      .join(', ') || 'none';
    console.log(`✓ ${file.padEnd(45)} | ${matchedFilters}`);
  } else {
    failed++;
    console.log(`✗ ${file.padEnd(45)} | FAIL`);
    console.log(`  ${desc}`);
    for (const [filterName, expectedValue] of Object.entries(expect)) {
      const actual = results[filterName];
      if (actual !== expectedValue) {
        console.log(`  ${filterName}: expected=${expectedValue}, got=${actual}`);
      }
    }
  }
});

console.log(`\n${passed}/${testCases.length} tests passed`);

if (failed > 0) {
  console.error(`\n${failed} test(s) failed!`);
  process.exit(1);
}

console.log('\nAll filter tests passed! ✓');

// --- Coverage Check ---
// Validates that all files in the repo are covered by at least one specific filter

if (runCoverage) {
  console.log('\n' + '='.repeat(60));
  console.log('Running full repository coverage check...\n');

  // Get repo root (two levels up from .github/scripts)
  const repoRoot = path.resolve(scriptDir, '../..');

  // Get all tracked files using git
  let allFiles;
  try {
    const output = execSync('git ls-files', { cwd: repoRoot, encoding: 'utf8' });
    allFiles = output.trim().split('\n').filter(f => f.length > 0);
  } catch (err) {
    console.error('Failed to run git ls-files. Are you in a git repository?');
    process.exit(1);
  }

  console.log(`Found ${allFiles.length} tracked files in repository\n`);

  // Specific filters to check (exclude 'all' since it matches everything)
  const specificFilters = Object.keys(filters).filter(f => f !== 'all');

  // Check each file
  const uncoveredFiles = [];

  for (const file of allFiles) {
    let covered = false;
    for (const filterName of specificFilters) {
      if (checkFilter(file, filters[filterName])) {
        covered = true;
        break;
      }
    }
    if (!covered) {
      uncoveredFiles.push(file);
    }
  }

  if (uncoveredFiles.length > 0) {
    console.error(`ERROR: ${uncoveredFiles.length} file(s) not covered by any CI filter:\n`);

    // Group by directory for readability
    const byDir = {};
    for (const file of uncoveredFiles) {
      const dir = path.dirname(file) || '.';
      if (!byDir[dir]) byDir[dir] = [];
      byDir[dir].push(path.basename(file));
    }

    for (const [dir, files] of Object.entries(byDir).sort()) {
      console.log(`  ${dir}/`);
      for (const file of files.slice(0, 10)) {
        console.log(`    - ${file}`);
      }
      if (files.length > 10) {
        console.log(`    ... and ${files.length - 10} more`);
      }
    }

    console.log('\nPlease add patterns for these files to .github/filters.yaml');
    process.exit(1);
  }

  console.log(`All ${allFiles.length} files are covered by CI filters! ✓`);
}
