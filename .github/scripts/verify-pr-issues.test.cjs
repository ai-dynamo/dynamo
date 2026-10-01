// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { references, verify } = require('./verify-pr-issues.cjs');

const repo = { owner: 'ai-dynamo', repo: 'dynamo' };

test('recognizes and deduplicates GitHub and Linear reference formats', () => {
  const refs = references(`Closes #123, GH-123,
    https://github.com/ai-dynamo/dynamo/issues/123#issuecomment-789
    other/repo#456 https://github.com/other/repo/issues/456
    DYN-321 https://linear.app/team/issue/DYN-321/a-title`, repo);
  assert.deepEqual(refs.map(ref => ref.ref), ['ai-dynamo/dynamo#123', 'other/repo#456', 'DYN-321']);
});

test('ignores template examples, placeholders and unrelated URL fragments', () => {
  assert.deepEqual(references(`<!-- Closes #8480 --> Closes #XXXX
    https://github.com/ai-dynamo/dynamo/pull/123#456
    https://example.com/path#789 <!-- DYN-321`, repo), []);
});

function harness(text, { title = '', issue, githubStatus, linear, linearStatus = 200, key = 'test-key', fetchError } = {}) {
  const result = { logs: [], errors: [], calls: [], html: '', failure: null };
  const core = {
    info: value => result.logs.push(value),
    error: value => result.errors.push(value),
    setFailed: value => { result.failure = value; },
    summary: {
      addHeading() { return this; },
      addRaw(value) { result.html += value; return this; },
      async write() {},
    },
  };
  const github = { rest: { issues: { async get(params) {
    result.calls.push(params);
    if (githubStatus) throw Object.assign(new Error('secret error details'), { status: githubStatus });
    return { data: issue || { title: 'GitHub issue title', state: 'closed' } };
  } } } };
  const fetchImpl = async (url, options) => {
    result.calls.push({ url, options });
    if (fetchError) throw new Error('secret request headers');
    return {
      ok: linearStatus === 200, status: linearStatus,
      async json() { return linear || { data: { issue: { identifier: 'DYN-321', title: 'Linear ticket title' } } }; },
    };
  };
  result.run = () => verify({ github, core, key, fetchImpl, context: { repo, payload: { pull_request: { title, body: text } } } });
  return result;
}

test('fails explicitly without a reference, making no API requests', async () => {
  const h = harness(null);
  await h.run();
  assert.match(h.failure, /Failed: no issue listed/);
  assert.match(h.html, /Failed: no issue listed/);
  assert.equal(h.calls.length, 0);
});

test('accepts a closed GitHub issue referenced in the title', async () => {
  const h = harness('', { title: 'fix: bug #123' });
  await h.run();
  assert.equal(h.failure, null);
  assert.match(h.html, /GitHub issue title/);
  assert.match(h.logs[0], /Found: ai-dynamo\/dynamo#123/);
  assert.deepEqual(h.calls[0], { owner: repo.owner, repo: repo.repo, issue_number: 123 });
});

test('verifies Linear with a GraphQL variable and reports its title', async () => {
  const h = harness('DYN-321');
  await h.run();
  assert.equal(h.failure, null);
  assert.match(h.html, /Linear ticket title/);
  assert.equal(h.calls[0].url, 'https://api.linear.app/graphql');
  assert.deepEqual(JSON.parse(h.calls[0].options.body).variables, { id: 'DYN-321' });
  assert.equal(h.calls[0].options.headers.Authorization, 'test-key');
  assert.equal(h.calls[0].options.redirect, 'error');
});

test('supports both trackers and escapes issue titles in the summary', async () => {
  const h = harness('#123 DYN-321', { issue: { title: '<script>alert("x")</script> & | title' } });
  await h.run();
  assert.equal(h.failure, null);
  assert.match(h.html, /2 of 2 referenced issues verified/);
  assert.match(h.html, /&lt;script&gt;/);
  assert.ok(!h.html.includes('<script>'));
});

test('rejects nonexistent or inaccessible GitHub issues and pull requests', async () => {
  for (const [options, expected] of [
    [{ githubStatus: 404 }, /does not exist or is not accessible/],
    [{ issue: { title: 'PR', pull_request: {} } }, /is a pull request, not an issue/],
    [{ githubStatus: 403 }, /GitHub API 403/],
    [{ githubStatus: 500 }, /GitHub API 500/],
  ]) {
    const h = harness('#123', options);
    await h.run();
    assert.match(h.failure, expected);
    assert.match(h.html, expected);
    assert.ok(!h.failure.includes('secret error details'));
  }
});

test('reports Linear missing credentials, missing issues and API failures distinctly', async () => {
  for (const [options, expected] of [
    [{ key: '' }, /LINEAR_ACCESS_KEY secret is missing/],
    [{ linear: { data: { issue: null } } }, /does not exist or is not accessible/],
    [{ linear: { errors: [{ message: 'Entity not found' }] } }, /does not exist or is not accessible/],
    [{ linearStatus: 401 }, /Linear API HTTP 401/],
    [{ linearStatus: 429 }, /Linear API HTTP 429/],
    [{ linear: { errors: [{ message: 'secret response detail' }], data: { issue: { title: 'partial data' } } } }, /Linear GraphQL error/],
    [{ linear: { data: {} } }, /unexpected Linear API response/],
    [{ fetchError: true }, /request failed or timed out/],
  ]) {
    const h = harness('DYN-321', options);
    await h.run();
    assert.match(h.failure, expected);
    assert.match(h.html, expected);
    assert.ok(!h.failure.includes('secret response detail'));
    assert.ok(!h.failure.includes('secret request headers'));
  }
});

test('one valid reference does not hide an invalid reference', async () => {
  const h = harness('#123 DYN-321', { linear: { data: { issue: null } } });
  await h.run();
  assert.match(h.failure, /Failed: issue listed DYN-321/);
  assert.match(h.html, /GitHub issue title/);
  assert.match(h.html, /1 of 2 referenced issues verified/);
});

test('bounds API requests for oversized PR text', async () => {
  const h = harness(Array.from({ length: 51 }, (_, i) => `#${i + 1}`).join(' '));
  await h.run();
  assert.match(h.failure, /more than 50/);
  assert.equal(h.calls.length, 0);
});
