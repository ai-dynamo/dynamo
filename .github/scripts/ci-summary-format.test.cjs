// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {extractFailureExcerpt, renderSummary} = require('./ci-summary-format.cjs');

const sha = '0123456789abcdef0123456789abcdef01234567';
const pullRequest = {number: 42, head: {sha}};
const run = {
  id: 1234, html_url: 'https://github.com/ai-dynamo/dynamo/actions/runs/1234',
  head_sha: sha, status: 'completed', conclusion: 'failure', run_attempt: 2,
};
const job = (id, conclusion, extra = {}) => ({
  id, name: `Job ${id}`, status: 'completed', conclusion,
  html_url: `${run.html_url}/job/${id}`, ...extra,
});

function summary(workflows) {
  return renderSummary({pullRequest, workflows});
}

test('extracts pytest failures instead of repeated generic process exits', () => {
  const generic = '##[error]Process completed with exit code 1.\n'.repeat(100);
  const excerpt = extractFailureExcerpt(`setup\n${generic}E   AssertionError: expected 2, got 1\nFAILED tests/test_router.py::test_route - AssertionError\n${generic}`);
  assert.match(excerpt, /FAILED tests\/test_router.py::test_route/);
  assert.match(excerpt, /AssertionError: expected 2/);
  assert.ok(excerpt.split('\n').length <= 12);
  assert.ok(excerpt.length <= 2000);
});

test('preserves Rust panic and compiler error evidence', () => {
  const excerpt = extractFailureExcerpt("thread 'tokio-runtime-worker' panicked at lib/router.rs:42:9:\nassertion `left == right` failed\n  left: 1\n right: 2\nerror[E0308]: mismatched types\n --> lib/main.rs:3:5\nProcess completed with exit code 101");
  assert.match(excerpt, /panicked at lib\/router.rs:42:9/);
  assert.match(excerpt, /left: 1/);
  assert.match(excerpt, /error\[E0308\]: mismatched types/);
});

test('strips terminal escapes, timestamps and control characters', () => {
  const excerpt = extractFailureExcerpt('\u001b]0;secret title\u0007\u001b[31m2026-09-28T13:04:05.1234567Z ERROR tests/test.py - bad\u001b[0m\u0000\u202e\n2026-09-28T13:04:06Z E\tAssertionError: nope');
  assert.equal(excerpt, 'ERROR tests/test.py - bad\nE  AssertionError: nope');
});

test('redacts token formats, quoted credentials, auth headers and credential URLs', () => {
  const log = [
    'ERROR hf_abcdefghijklmnopqrstuvwxyz ghp_abcdefghijklmnopqrstuvwxyz github_pat_1234_abcd',
    'ERROR PASSWORD="two secret words" access_token=unquoted-secret api_key: abcdef',
    'ERROR Authorization: Bearer opaque-secret',
    'ERROR {"password": "json-secret", "token": "json-token"}',
    'ERROR https://user:password@example.com/repo @maintainers',
  ].join('\n');
  const excerpt = extractFailureExcerpt(log);
  for (const secret of ['hf_abcdefgh', 'ghp_abcdef', 'github_pat_', 'two secret words', 'unquoted-secret', 'abcdef', 'opaque-secret', 'json-secret', 'json-token', 'user:password']) {
    assert.ok(!excerpt.includes(secret), `leaked ${secret}`);
  }
  assert.match(excerpt, /\[REDACTED\]/);
  assert.ok(excerpt.includes('@\u200bmaintainers'));
});

test('bounds oversized logs and excerpts and handles empty logs', () => {
  assert.equal(extractFailureExcerpt(null), '');
  assert.equal(extractFailureExcerpt('\n \n'), '');
  const excerpt = extractFailureExcerpt(`FAILED tests/test_start.py::test_one\n${'noise\n'.repeat(400000)}error: final compilation failed\n${'x'.repeat(5000)}`);
  assert.match(excerpt, /FAILED tests\/test_start.py::test_one/);
  assert.match(excerpt, /final compilation failed/);
  assert.ok(excerpt.length <= 2000);
});

test('reports missing full CI, optional XPU and Pre Merge independently', () => {
  const body = summary(['PR', 'PR-XPU', 'Pre Merge'].map(name => ({name, run: null, jobs: [], failures: []})));
  assert.ok(body.startsWith('<!-- dynamo-ci-summary -->'));
  assert.ok(body.includes(`<code>${sha}</code>`));
  assert.match(body, /No run for this head; full CI requires approval/);
  assert.match(body, /Optional \/ not run/);
  assert.match(body, /Pre Merge \| — Not run/);
});

test('success summary counts every job without treating skips or cancellations as passing', () => {
  const jobs = [job(1, 'success'), job(2, 'skipped'), job(3, 'cancelled'), job(4, 'neutral')];
  const body = summary([{name: 'PR', run: {...run, conclusion: 'success'}, jobs, failures: []}]);
  assert.match(body, /PR \| ✅ Pass/);
  assert.match(body, /4 total: 1 passed, 0 failed, 1 skipped, 1 cancelled, 1 other/);
  assert.match(body, /Run 1234 · attempt 2/);
  assert.ok(body.includes(run.html_url));
  assert.ok(!body.includes('<details>'));
});

test('failed jobs include step links and excerpts, including unavailable logs', () => {
  const failed = job(10, 'failure', {steps: [{name: 'Run tests', number: 4, conclusion: 'failure'}]});
  const timeout = job(11, 'timed_out');
  const body = summary([{name: 'PR', run, jobs: [failed, timeout, job(12, 'skipped')], failures: [
    {job: failed, excerpt: 'FAILED tests/test_router.py::test_route - AssertionError'},
    {job: timeout, unavailable: true},
  ]}]);
  assert.match(body, /PR \| ❌ Failed/);
  assert.match(body, /Job 10 — ❌ Failed<\/summary>/);
  assert.match(body, /Job 11 — ⏱ Timed out<\/summary>/);
  assert.match(body, /\(❌ Failed\)/);
  assert.match(body, /3 total: 0 passed, 2 failed, 1 skipped, 0 cancelled/);
  assert.ok(body.includes(`${failed.html_url}#step:4:1`));
  assert.match(body, /<pre>FAILED tests\/test_router.py::test_route - AssertionError<\/pre>/);
  assert.match(body, /Log excerpt unavailable/);
});

test('pending workflows distinguish queued and running jobs', () => {
  const jobs = [job(1, null, {status: 'queued'}), job(2, null, {status: 'in_progress'})];
  const body = summary([{name: 'PR', run: {...run, status: 'in_progress', conclusion: null}, jobs}]);
  assert.match(body, /1 running, 1 queued/);
  assert.match(body, /PR \| 🔄 Running/);
});

test('discovers failed jobs when the collector has no excerpt entry', () => {
  const body = summary([{name: 'PR', run, jobs: [job(1, 'action_required')]}]);
  assert.match(body, /Job 1 — ⚠️ Action required<\/summary>/);
  assert.match(body, /1 failed/);
  assert.match(body, /Log excerpt unavailable/);
});

test('escapes adversarial names and logs and suppresses mentions', () => {
  const name = '</summary><script>alert(1)</script> [x](javascript:evil) | @team **bold** &';
  const failed = job(1, 'failure', {name, html_url: 'javascript:alert(1)', steps: [{name, number: 1, conclusion: 'failure'}]});
  const body = summary([{name, run: {...run, conclusion: name}, jobs: [failed], failures: [{job: failed, excerpt: 'ERROR </pre><img src=x onerror=alert(1)> @team ```\nHF_TOKEN=hf_supersecret'}]}]);
  assert.ok(!body.includes('<script>'));
  assert.ok(!body.includes('<img'));
  assert.ok(!body.includes('](javascript:'));
  assert.ok(!body.includes('javascript:alert(1)'));
  assert.ok(!body.includes('@team'));
  assert.ok(!body.includes('hf_supersecret'));
  assert.ok(body.includes('&lt;/pre&gt;'));
  assert.ok(body.includes('❔ Unknown ('));
  assert.ok(body.includes('&#124;'));
  assert.ok(body.includes('&#42;&#42;bold&#42;&#42;'));
  assert.ok(!body.includes('&&#35;'));
});

test('caps failure details with an explicit omission and full-run link', () => {
  const jobs = Array.from({length: 50}, (_, index) => job(index, 'failure'));
  const body = summary([{name: 'PR', run, jobs, failures: jobs.map(failed => ({job: failed, excerpt: 'ERROR compilation failed'}))}]);
  assert.equal((body.match(/<details>/g) || []).length, 20);
  assert.match(body, /30 additional failed job\(s\) omitted/);
  assert.ok(body.includes(`[View the full run](${run.html_url})`));
});

test('bounds the complete comment even when escaping expands all failure data', () => {
  const noisy = '<&'.repeat(2000);
  const jobs = Array.from({length: 100}, (_, index) => job(index, 'failure', {
    name: noisy,
    steps: Array.from({length: 20}, (_, number) => ({name: noisy, number, conclusion: 'failure'})),
  }));
  const workflows = Array.from({length: 30}, (_, index) => ({
    name: noisy, run: {...run, id: index, html_url: `${run.html_url}/${'x'.repeat(1000)}`}, jobs,
    failures: jobs.map(failed => ({job: failed, excerpt: `ERROR ${noisy}`})),
  }));
  const body = summary(workflows);
  assert.ok(body.length <= 60000, `comment is ${body.length} characters`);
  assert.equal((body.match(/<details>/g) || []).length, (body.match(/<\/details>/g) || []).length);
  assert.match(body, /additional failed job\(s\) omitted/);
  assert.match(body, /additional workflows omitted/);
  assert.ok(body.endsWith('Open the linked job or run for the full logs.\n'));
});


test('reserves excerpt budget for the primary diagnostic before long earlier errors', () => {
  const error = `ERROR from Kubernetes: ${'noisy context '.repeat(200)}`;
  const primary = 'FAILED tests/deploy/test.py::test_restore - TimeoutError: Deployment failed to reach Ready=True within 900 seconds';
  const excerpt = extractFailureExcerpt(`${error}\n${error}\n${error}\n${primary}\n${error}\n${error}\n${error}\n${error}\n${error}\n${error}\n${error}\nProcess completed with exit code 1`);
  assert.ok(excerpt.includes(primary));
  assert.ok(excerpt.length <= 2000);
  assert.ok(excerpt.split('\n').length <= 12);
});

test('preserves both ends of a long diagnostic and does not rerank extracted snippets', () => {
  const excerpt = extractFailureExcerpt(`FAILED tests/test.py::test_long[${'param'.repeat(300)}] - AssertionError: expected 10, got 1`);
  assert.ok(excerpt.startsWith('FAILED tests/test.py::test_long'));
  assert.ok(excerpt.endsWith('AssertionError: expected 10, got 1'));
  const failed = job(1, 'failure');
  const snippet = 'ERROR first context\nFAILED tests/test.py::test_long - AssertionError\nERROR last context';
  const body = summary([{name: 'PR', run, jobs: [failed], failures: [{job: failed, excerpt: snippet}]}]);
  assert.ok(body.includes(`<pre>${snippet}</pre>`));
});

test('includes sanitized reasons when log fetching is unavailable', () => {
  const failed = job(1, 'failure');
  const body = summary([{name: 'PR', run, jobs: [failed], failures: [{
    job: failed, unavailable: 'Log budget exhausted <script>@team</script> token=private-value',
  }]}]);
  assert.match(body, /Log budget exhausted/);
  assert.ok(!body.includes('<script>'));
  assert.ok(!body.includes('@team'));
  assert.ok(!body.includes('private-value'));
});


test('redacts escaped quotes inside credential assignments', () => {
  const excerpt = extractFailureExcerpt(String.raw`ERROR PASSWORD="first \"quoted-secret\" trailing-secret"`);
  assert.ok(!excerpt.includes('quoted-secret'));
  assert.ok(!excerpt.includes('trailing-secret'));
  assert.match(excerpt, /\[REDACTED\]/);
});
