// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { test } = require('node:test');
const rerun = require('./rerun-ci');

const NOW = Date.parse('2026-11-19T20:00:00Z');
const DAY = 24 * 60 * 60 * 1000;
const SHA = 'a'.repeat(40);
const marker = id => `<!-- dynamo-rerun-ci:run:${id}:attempt:1 -->`;

function fixture(t, { age = DAY, count = 1 } = {}) {
  t.mock.method(Date, 'now', () => NOW);
  const comments = [];
  const requested = [];
  const warnings = [];
  const failures = [];
  const pull = { state: 'open', head: { sha: SHA, ref: 'feature' }, base: { ref: 'main' } };
  const runs = Array.from({ length: count }, (_, index) => ({
    id: index + 1, path: `.github/workflows/fixture-${index}.yml`,
    html_url: `https://example.invalid/runs/${index + 1}`,
    event: 'pull_request', head_sha: SHA, head_branch: 'feature',
    pull_requests: [{ number: 123, head: { sha: SHA }, base: { ref: 'main' } }],
    status: 'completed', conclusion: 'failure', run_attempt: 1,
    created_at: new Date(NOW - age).toISOString(),
    updated_at: new Date(NOW - 1000).toISOString(),
  }));
  const context = {
    repo: { owner: 'fixture', repo: 'fixture' }, eventName: 'issue_comment',
    payload: {
      action: 'created', issue: { number: 123, pull_request: {} },
      comment: { id: 456, body: '/rerun ci', created_at: new Date(NOW).toISOString(), user: { login: 'writer' } },
    },
  };
  const rest = {
    issues: {
      listComments: async () => structuredClone(comments),
      createComment: async ({ body }) => {
        const comment = { id: comments.length + 1, body, user: { id: 41898282, type: 'Bot' } };
        comments.push(comment);
        return { data: comment };
      },
      updateComment: async ({ comment_id, body }) => { comments.find(c => c.id === comment_id).body = body; },
    },
    repos: {
      getCollaboratorPermissionLevel: async () => ({ data: { permission: 'write' } }),
      getBranch: async () => { throw Object.assign(new Error('Missing mirror'), { status: 404 }); },
    },
    reactions: { createForIssueComment: async () => {} },
    pulls: { get: async () => ({ data: pull }) },
    actions: {
      listWorkflowRunsForRepo: async ({ head_sha, branch, event }) => runs.filter(run =>
        run.head_sha === head_sha && run.head_branch === branch && run.event === event),
      listJobsForWorkflowRun: async () => [{ conclusion: 'failure' }],
      getWorkflowRun: async ({ run_id }) => ({ data: runs.find(r => r.id === run_id) }),
      reRunWorkflowFailedJobs: async ({ run_id }) => {
        assert(comments.some(c => c.body.includes(marker(run_id))), 'Persist a reservation before each POST');
        requested.push(run_id);
      },
    },
  };
  const core = {
    info: () => {}, warning: message => warnings.push(message), setFailed: message => failures.push(message),
  };
  return {
    rest, runs, pull, context, comments, requested, warnings, failures,
    invoke: () => rerun({ github: { rest, paginate: (method, args) => method(args) }, context, core }),
    reply: () => comments.at(-1).body,
  };
}

for (const [name, age, accepted] of [
  ['just under seven days', 7 * DAY - 1, true],
  ['exactly seven days', 7 * DAY, false],
  ['eight days', 8 * DAY, false],
]) {
  test(`retry window: ${name}`, async t => {
    const f = fixture(t, { age });
    await f.invoke();
    assert.deepEqual(f.requested, accepted ? [1] : []);
    if (!accepted) assert.match(f.reply(), /seven-day retry window/);
  });
}

test('reaction failure does not prevent retries', async t => {
  const f = fixture(t);
  f.rest.reactions.createForIssueComment = async () => { throw new Error('Reaction unavailable'); };
  await f.invoke();
  assert.deepEqual(f.requested, [1]);
  assert.equal(f.warnings.length, 1);
  assert.deepEqual(f.failures, []);
});

for (const [stage, message] of [
  ['listJobsForWorkflowRun', /could not inspect jobs/],
  ['getWorkflowRun', /could not inspect the current attempt/],
]) {
  test(`${stage} failure does not block another workflow`, async t => {
    const f = fixture(t, { count: 2 });
    const original = f.rest.actions[stage];
    f.rest.actions[stage] = async args => {
      if (args.run_id === 1) throw Object.assign(new Error('API unavailable'), { status: 503 });
      return original(args);
    };
    await f.invoke();
    assert.deepEqual(f.requested, [2]);
    assert.match(f.reply(), message);
    assert.match(f.warnings[0], /workflow run 1 \(HTTP 503\)/);
    assert(!f.reply().includes(marker(1)));
    assert.equal(f.failures.length, 1);
  });
}

for (const [name, status, retained] of [
  ['definite rejection', 403, false],
  ['server error', 503, true],
  ['network error', undefined, true],
  ['conflict', 409, true],
  ['request timeout', 408, true],
]) {
  test(`${name}: continue the batch and ${retained ? 'retain' : 'release'} the reservation`, async t => {
    const f = fixture(t, { count: 2 });
    const original = f.rest.actions.reRunWorkflowFailedJobs;
    f.rest.actions.reRunWorkflowFailedJobs = async args => {
      if (args.run_id === 1) throw Object.assign(new Error(name), { status });
      return original(args);
    };
    await f.invoke();
    assert.deepEqual(f.requested, [2]);
    assert.equal(f.reply().includes(marker(1)), retained);
    assert(f.reply().includes(marker(2)));
    assert.match(f.reply(), retained ? /could not confirm the retry/ : /retry rejected \(HTTP 403\)/);
    assert.equal(f.failures.length, 1);
    assert(!f.reply().includes('not requested yet'));

    f.context.payload.comment.id++;
    f.rest.actions.reRunWorkflowFailedJobs = original;
    await f.invoke();
    assert.deepEqual(f.requested, retained ? [2] : [2, 1]);
  });
}

test('replayed commands do not issue another POST', async t => {
  const f = fixture(t);
  await f.invoke();
  await f.invoke();
  assert.deepEqual(f.requested, [1]);
});

test('a failed reservation write prevents the POST', async t => {
  const f = fixture(t);
  f.rest.issues.updateComment = async () => { throw new Error('Comment unavailable'); };
  await assert.rejects(f.invoke(), /Comment unavailable/);
  assert.deepEqual(f.requested, []);
});

test('a changed PR head stops the remaining batch', async t => {
  const f = fixture(t, { count: 2 });
  const original = f.rest.actions.reRunWorkflowFailedJobs;
  f.rest.actions.reRunWorkflowFailedJobs = async args => {
    await original(args);
    f.pull.head.sha = 'b'.repeat(40);
  };
  await f.invoke();
  assert.deepEqual(f.requested, [1]);
  assert.match(f.reply(), /not requested because the PR revision/);
  assert(!f.reply().includes(marker(2)));
});

test('users without write access cannot request retries', async t => {
  const f = fixture(t);
  f.rest.repos.getCollaboratorPermissionLevel = async () => ({ data: { permission: 'read' } });
  await f.invoke();
  assert.deepEqual(f.requested, []);
  assert.match(f.reply(), /requires repository write access/);
});

for (const [approval, accepted] of [['current', true], ['missing', false], ['different commit', false]]) {
  test(`copied-branch approval is ${approval}`, async t => {
    const f = fixture(t);
    Object.assign(f.runs[0], {
      event: 'push', head_branch: 'pull-request/123', actor: { id: 138252374 }, pull_requests: [],
    });
    if (approval !== 'missing') {
      f.rest.repos.getBranch = async () => ({ data: { commit: { sha: accepted ? SHA : 'b'.repeat(40) } } });
    }
    await f.invoke();
    assert.deepEqual(f.requested, accepted ? [1] : []);
    assert.equal(f.reply().includes(marker(1)), accepted);
  });
}

test('closed PRs are rejected before any retry request', async t => {
  const f = fixture(t);
  f.pull.state = 'closed';
  await f.invoke();
  assert.deepEqual(f.requested, []);
  assert(!f.reply().includes(marker(1)));
});

test('skipped jobs do not cause retry requests', async t => {
  const f = fixture(t);
  f.runs[0].conclusion = 'skipped';
  f.rest.actions.listJobsForWorkflowRun = async () => [{ conclusion: 'skipped' }];
  await f.invoke();
  assert.deepEqual(f.requested, []);
  assert.match(f.reply(), /no failed jobs/);
});
