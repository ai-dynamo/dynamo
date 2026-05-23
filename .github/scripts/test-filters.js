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
 * - Exclude doc files (*.md, *.rst, *.txt) from core via negation patterns
 * - Match CI/infrastructure changes to core
 * - Match parser-specific files to parser-only filters
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
const repoRoot = path.resolve(scriptDir, '../..');
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
    file: 'deploy/operator/cmd/main.go',
    expect: { core: false, operator: true },
    desc: 'operator file triggers operator'
  },
  {
    file: 'deploy/helm/charts/platform/values.yaml',
    expect: { core: false, deploy: true },
    desc: 'helm file triggers deploy'
  },
];

function listFilesRecursive(relativeDir) {
  const root = path.resolve(repoRoot, relativeDir);
  if (!fs.existsSync(root)) return [];

  const out = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
      } else if (entry.isFile()) {
        out.push(path.relative(repoRoot, fullPath).replaceAll(path.sep, '/'));
      }
    }
  }
  return out.sort();
}

for (const file of listFilesRecursive('lib/parsers').filter(file => file.endsWith('.rs'))) {
  testCases.push({
    file,
    expect: { core: false, rust: false, parser: true, parser_rust: true },
    desc: 'parser Rust file triggers parser Rust checks only',
  });
}

for (const file of listFilesRecursive('tests/parity/parser').filter(file => file.endsWith('.py'))) {
  const expect = {
    core: false,
    parser: true,
    parser_rust: false,
    parser_vllm: false,
    parser_sglang: false,
  };
  if (file === 'tests/parity/parser/test_parity_parser.py') {
    expect.parser_vllm = true;
    expect.parser_sglang = true;
  } else if (file === 'tests/parity/parser/vllm.py') {
    expect.parser = false;
    expect.parser_vllm = true;
  } else if (file === 'tests/parity/parser/sglang.py') {
    expect.parser = false;
    expect.parser_sglang = true;
  }
  testCases.push({
    file,
    expect,
    desc: 'parser parity Python file triggers parser tests only',
  });
}

const parserFixtureYamlFiles = listFilesRecursive('tests/parity/parser/fixtures').filter(
  file => file.endsWith('.yaml') || file.endsWith('.yml')
);

testCases.push(
  {
    file: 'tests/parity/common.py',
    expect: {
      core: false,
      parser: true,
      parser_vllm: true,
      parser_sglang: true,
      parser_rust: false,
    },
    desc: 'shared parity helpers trigger all parser parity lanes',
  },
  {
    file: 'tests/parity/__init__.py',
    expect: {
      core: false,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      parser_rust: false,
    },
    desc: 'parity package init triggers parser utilities',
  },
  {
    file: 'tests/parity/generate_parity_table.py',
    expect: {
      core: false,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      parser_rust: false,
    },
    desc: 'shared parity table CLI triggers parser utilities',
  },
  {
    file: 'tests/parity/parity_table.html.j2',
    expect: {
      core: false,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      parser_rust: false,
    },
    desc: 'shared parity table template triggers parser utilities',
  },
  {
    file: 'tests/parity/README.md',
    expect: { core: false, parser: false, parser_vllm: false, parser_sglang: false, docs: true },
    desc: 'parity README remains docs only',
  },
  {
    file: 'lib/parsers/README.md',
    expect: { core: false, parser: false, parser_rust: false, docs: true },
    desc: 'parser README remains docs only',
  },
  {
    file: 'lib/parsers/Cargo.toml',
    expect: { core: false, rust: false, parser: true, parser_rust: true },
    desc: 'parser Cargo.toml triggers parser Rust checks only',
  },
  {
    file: 'lib/bindings/python/rust/parsers.rs',
    expect: {
      core: false,
      frontend: false,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      rust: true,
      parser_rust: false,
    },
    desc: 'python parser binding triggers parser and normal Rust checks',
  },
  {
    file: 'lib/bindings/python/src/dynamo/_core.pyi',
    expect: {
      core: true,
      frontend: true,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      rust: false,
      parser_rust: false,
    },
    desc: '_core.pyi has shared runtime stubs beyond parser APIs — triggers core + frontend + parser so non-parser stub changes still get full validation',
  },
  {
    file: 'lib/bindings/python/tests/test_parsers.py',
    expect: {
      core: false,
      frontend: false,
      parser: true,
      parser_vllm: false,
      parser_sglang: false,
      parser_rust: false,
    },
    desc: 'python parser binding test triggers parser only',
  },
  {
    file: 'tests/parity/parser/PARITY.html',
    expect: { core: false, parser: false, ignore: true },
    desc: 'generated parser parity HTML is ignored',
  },
);

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

let fixtureAssertions = 0;
let fixtureFailures = 0;
for (const file of parserFixtureYamlFiles) {
  const expect = {
    core: false,
    docs: false,
    parser: true,
    parser_vllm: true,
    parser_sglang: true,
    parser_rust: false,
  };
  for (const [filterName, expectedValue] of Object.entries(expect)) {
    const actual = checkFilter(file, filters[filterName]);
    if (actual !== expectedValue) {
      fixtureFailures++;
      console.log(
        `✗ ${file} | ${filterName}: expected=${expectedValue}, got=${actual}`
      );
    } else {
      fixtureAssertions++;
    }
  }
}

console.log(`\n${passed}/${testCases.length} tests passed`);
console.log(
  `Parser fixture sweep: ${parserFixtureYamlFiles.length} files, ${fixtureAssertions} assertions passed`
);

if (failed > 0 || fixtureFailures > 0) {
  console.error(`\n${failed + fixtureFailures} test(s) failed!`);
  process.exit(1);
}

console.log('\nAll filter tests passed! ✓');

// --- Coverage Check ---
// Validates that all files in the repo are covered by at least one specific filter

if (runCoverage) {
  console.log('\n' + '='.repeat(60));
  console.log('Running full repository coverage check...\n');

  // Get all tracked files using git
  let allFiles;
  try {
    const output = execSync('git ls-files', {
      cwd: repoRoot,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
    });
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
