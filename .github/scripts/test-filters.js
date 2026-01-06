#!/usr/bin/env node
/**
 * Test script for .github/filters.yaml pattern matching.
 * Reads patterns directly from filters.yaml and validates behavior.
 *
 * Usage:
 *   cd .github/scripts
 *   npm install
 *   npm test
 *
 * This validates that tj-actions/changed-files will correctly:
 * - Match backend-specific files to their respective filters (vllm, sglang, trtllm)
 * - Exclude doc files (*.md, *.rst, *.txt) from core via negation patterns
 * - Match CI/infrastructure changes to core
 */

const fs = require('fs');
const path = require('path');
const micromatch = require('micromatch');
const YAML = require('yaml');

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
    file: 'components/src/dynamo/vllm/worker.py',
    expect: { core: false, vllm: true },
    desc: 'vllm component triggers only vllm'
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
    file: 'deploy/cloud/operator/main.go',
    expect: { core: false, operator: true },
    desc: 'operator file triggers operator'
  },
  {
    file: 'deploy/cloud/helm/values.yaml',
    expect: { core: false, deploy: true },
    desc: 'helm file triggers deploy'
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
