// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';

const fs = require('node:fs');
const micromatch = require('micromatch');
const YAML = require('yaml');

const REPOSITORY = 'ai-dynamo/dynamo';
const REGISTRY = '210086341041.dkr.ecr.us-west-2.amazonaws.com';
const SHA = /^[a-f0-9]{40}$/;
const STANDING = new Set(['trusted_contributor', 'maintainer', 'core_maintainer', 'project_leader']);
const EXCLUSIONS = ['deployment tests', 'ACR copies', 'NGC publication', 'Triton Artifactory QA'];

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function eligibleAuthors(policy) {
  requireValue(Array.isArray(policy?.contributors), 'Missing contributor policy');
  const authors = new Set();
  const seen = new Set();
  for (const entry of policy.contributors) {
    requireValue(/^[a-z0-9][a-z0-9-]{0,38}$/i.test(entry.github), 'Invalid contributor login');
    const login = entry.github.toLowerCase();
    requireValue(!seen.has(login), 'Duplicate contributor login');
    seen.add(login);
    requireValue(Array.isArray(entry.roles) && entry.roles.length === 1, 'Invalid contributor role');
    const role = entry.roles[0];
    requireValue(role.repo === 'dynamo', 'Unexpected contributor repository');
    const standing = String(role.role).toLowerCase().replace(/[\s-]+/g, '_');
    requireValue(STANDING.has(standing) || standing === 'contributor', 'Unknown contributor standing');
    if (STANDING.has(standing)) authors.add(login);
  }
  return authors;
}

function matches(file, patterns = []) {
  const flat = patterns.flat(Infinity);
  return micromatch.isMatch(file, flat.filter(p => !p.startsWith('!')))
    && !micromatch.isMatch(file, flat.filter(p => p.startsWith('!')).map(p => p.slice(1)));
}

function plan({pr, files, filters, runId, attempt, mergeSha}) {
  requireValue(pr.state === 'open' && pr.base.repo.full_name === REPOSITORY && pr.base.ref === 'main',
    'PR must target Dynamo main');
  requireValue(Number.isSafeInteger(pr.number) && pr.number > 0, 'Invalid PR number');
  requireValue(SHA.test(pr.head.sha) && SHA.test(mergeSha) && pr.merge_commit_sha === mergeSha,
    'PR merge identity changed');
  requireValue(/^[1-9][0-9]*$/.test(String(runId)) && /^[1-9][0-9]*$/.test(String(attempt)), 'Invalid run identity');
  requireValue(Array.isArray(files) && files.length <= 3000, 'Incomplete changed-file list');
  requireValue(!files.some(file => file.filename.startsWith('.github/')
    || file.previous_filename?.startsWith('.github/')), 'Workflow edits require the maintainer CI lane');
  const paths = files.flatMap(file => [file.filename, file.previous_filename].filter(Boolean));
  const flags = Object.fromEntries(Object.entries(filters)
    .filter(([key]) => key !== 'all')
    .map(([key, patterns]) => [key, paths.some(file => matches(file, patterns))]));
  const uncovered = paths.filter(file => !Object.entries(filters)
    .some(([key, patterns]) => key !== 'all' && matches(file, patterns)));
  requireValue(!uncovered.length, `Uncovered CI paths: ${JSON.stringify(uncovered)}`);
  const components = [];
  for (const framework of ['vllm', 'sglang', 'trtllm', 'triton']) {
    if (flags.core || flags[framework]) components.push({framework, target: 'runtime'});
  }
  if (flags.core || flags.planner) components.push({framework: 'dynamo', target: 'planner'});
  if (flags.core || flags.frontend) components.push({framework: 'dynamo', target: 'frontend'});
  if (flags.core || flags.benchmarks || flags.sample) components.push({framework: 'dynamo', target: 'runtime'});
  if (flags.sidecar) components.push({framework: 'sidecar', target: 'runtime'});
  const builds = [];
  const tests = [];
  for (const component of components) {
    for (const arch of component.framework === 'triton' ? ['amd64'] : ['amd64', 'arm64']) {
      const cuda = component.framework === 'trtllm' || component.framework === 'triton' ? '13.1' : '13.0';
      const key = `${component.framework}-${component.target}-${arch}`;
      const tag = `pr-${pr.number}-${pr.head.sha}-r${runId}-a${attempt}-${key}`;
      const image = `${REGISTRY}/dynamo-pr-ci:${tag}`;
      builds.push({...component, arch, cuda, key, image,
        cache: `${REGISTRY}/dynamo-pr-ci/cache:pr-${pr.number}-${key}`});
      if (component.framework === 'sidecar') continue;
      const suite = component.target === 'runtime' ? component.framework : component.target;
      const backend = component.framework === 'dynamo' ? '(not (vllm or sglang or trtllm or triton))' : component.framework;
      const common = `pre_merge and ${backend} and not triton_qa and not sidecar`;
      tests.push({key: `${key}-cpu`, suite, framework: component.framework, arch, tier: 'cpu', image: `${image}-test`,
        labels: ['trusted_contributors', 'cpu', arch, 'tester'],
        check: component.target === 'frontend' ? 'frontend' : 'pytest',
        markers: `${common}${component.target === 'planner' ? ' and planner' : ''} and gpu_0`});
      if (arch === 'amd64' && ['vllm', 'sglang', 'trtllm'].includes(component.framework)) {
        for (const count of [1, 2]) tests.push({key: `${key}-gpu-${count}`, suite, framework: component.framework,
          arch, tier: `gpu-${count}`, image: `${image}-test`,
          labels: ['trusted_contributors', `gpu-${count}`, arch, 'tester'],
          markers: `${common} and gpu_${count}`, check: 'pytest'});
      }
    }
  }
  return {builds: {include: builds}, tests: {include: tests}, has_builds: builds.length > 0,
    has_tests: tests.length > 0, head: pr.head.sha, source: mergeSha, exclusions: EXCLUSIONS};
}

async function api(path) {
  requireValue(path.startsWith(`/repos/${REPOSITORY}/`), 'Unexpected API repository');
  const response = await fetch(`https://api.github.com${path}`, {headers: {
    Authorization: `Bearer ${process.env.GH_TOKEN}`, Accept: 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28'}, signal: AbortSignal.timeout(20000)});
  requireValue(response.ok, `GitHub API failed (${response.status})`);
  return response.json();
}

async function filesFor(pr) {
  requireValue(pr.changed_files <= 3000, 'PR exceeds the changed-file API limit');
  const files = [];
  for (let page = 1; page <= 30 && files.length < pr.changed_files; page++) {
    const next = await api(`/repos/${REPOSITORY}/pulls/${pr.number}/files?per_page=100&page=${page}`);
    requireValue(Array.isArray(next) && next.length, 'Truncated changed-file response');
    files.push(...next);
  }
  requireValue(files.length === pr.changed_files && new Set(files.map(f => f.filename)).size === files.length,
    'Changed-file enumeration is incomplete');
  return files;
}

async function main() {
  const event = JSON.parse(fs.readFileSync(process.env.GITHUB_EVENT_PATH, 'utf8'));
  requireValue(process.env.GITHUB_REPOSITORY === REPOSITORY && process.env.GITHUB_EVENT_NAME === 'pull_request',
    'Only native Dynamo PR runs are supported');
  requireValue(Number.isSafeInteger(event.number), 'Missing PR identity');
  const pr = await api(`/repos/${REPOSITORY}/pulls/${event.number}`);
  requireValue(pr.head.sha === event.pull_request.head.sha
    && pr.head.repo.full_name === event.pull_request.head.repo.full_name, 'PR head changed');
  const policy = YAML.parse(fs.readFileSync('contributors.yaml', 'utf8'));
  const candidate = eligibleAuthors(policy).has(pr.user.login.toLowerCase())
    && pr.head.repo.full_name !== REPOSITORY;
  const output = {candidate};
  if (candidate) Object.assign(output, plan({pr, files: await filesFor(pr),
    filters: YAML.parse(fs.readFileSync('.github/filters.yaml', 'utf8')),
    runId: process.env.GITHUB_RUN_ID, attempt: process.env.GITHUB_RUN_ATTEMPT, mergeSha: process.env.GITHUB_SHA}));
  for (const [key, value] of Object.entries(output)) {
    fs.appendFileSync(process.env.GITHUB_OUTPUT, `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}\n`);
  }
  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY,
    `Trusted contributor lane candidate: ${candidate}. Hardware admission independently checks every commit author and signature.\n\nExcluded: ${EXCLUSIONS.join(', ')}.\n`);
}

module.exports = {eligibleAuthors, matches, plan, REGISTRY};
if (require.main === module) main().catch(error => { console.error(error.message); process.exitCode = 1; });
