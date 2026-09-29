// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { test } = require('node:test');
const {
  resolvePullRequest, collectSummary, publishSummary, readJobLog,
} = require('./ci-summary.cjs');

const REPO = { owner: 'ai-dynamo', repo: 'dynamo' };
const FULL_NAME = 'ai-dynamo/dynamo';
const SHA = 'a'.repeat(40);
const MARKER = '<!-- dynamo-ci-summary -->';
const WORKFLOW_FILES = ['pre-merge.yml', 'pr.yaml', 'pr-xpu.yaml'];

function pullRequest(overrides = {}) {
  return {
    number: 4542, state: 'open', html_url: 'https://github.com/ai-dynamo/dynamo/pull/4542',
    base: { ref: 'main', repo: { full_name: FULL_NAME } },
    head: { sha: SHA, ref: 'feature', repo: { full_name: 'contributor/dynamo' } },
    ...overrides,
  };
}

function workflowRun(file = 'pr.yaml', overrides = {}) {
  const isPull = file === 'pre-merge.yml';
  return {
    id: 100, name: isPull ? 'Pre Merge' : file === 'pr.yaml' ? 'PR' : 'PR-XPU',
    path: `.github/workflows/${file}`, event: isPull ? 'pull_request' : 'push',
    repository: { full_name: FULL_NAME },
    head_repository: { full_name: isPull ? 'contributor/dynamo' : FULL_NAME },
    head_sha: SHA, head_branch: isPull ? 'feature' : 'pull-request/4542',
    pull_requests: [{ number: 4542 }], run_attempt: 1,
    status: 'completed', conclusion: 'failure',
    html_url: 'https://github.com/ai-dynamo/dynamo/actions/runs/100',
    created_at: '2026-09-28T12:00:00Z', updated_at: '2026-09-28T12:05:00Z',
    ...overrides,
  };
}

function job(id = 1, overrides = {}) {
  return {
    id, name: `unit-tests-${id}`, status: 'completed', conclusion: 'failure',
    html_url: `https://github.com/ai-dynamo/dynamo/actions/runs/100/job/${id}`,
    steps: [{ number: 1, name: 'Run tests', status: 'completed', conclusion: 'failure' }],
    ...overrides,
  };
}

function harness(options = {}) {
  const requests = [];
  const paginations = [];
  const writes = [];
  const infos = [];
  const pr = options.pr || pullRequest();
  let pullReads = 0;
  const listReads = {};
  const endpoint = (name, implementation) => Object.assign(async args => {
    requests.push({ name, args });
    return { data: await implementation(args) };
  }, { endpointName: name });
  const github = {
    rest: {
      actions: {
        getWorkflowRun: endpoint('getWorkflowRun', () => options.run || workflowRun()),
        listWorkflowRuns: endpoint('listWorkflowRuns', args => {
          const count = listReads[args.workflow_id] = (listReads[args.workflow_id] || 0) + 1;
          return options.getRuns ? options.getRuns(args, count) : (options.runs?.[args.workflow_id] || []);
        }),
        listJobsForWorkflowRun: endpoint('listJobsForWorkflowRun', args => options.jobs?.[args.run_id] || []),
      },
      repos: {
        listPullRequestsAssociatedWithCommit: endpoint('listPullRequestsAssociatedWithCommit', () => options.associations || []),
      },
      pulls: {
        list: endpoint('listPullRequests', () => options.sourcePRs || []),
        get: endpoint('getPullRequest', args => {
          pullReads++;
          return options.getPullRequest ? options.getPullRequest(args, pullReads) : pr;
        }),
      },
      issues: {
        listComments: endpoint('listComments', () => options.comments || []),
        updateComment: endpoint('updateComment', args => { writes.push({ type: 'update', ...args }); }),
        createComment: endpoint('createComment', args => { writes.push({ type: 'create', ...args }); }),
      },
    },
    paginate: async (method, args) => {
      paginations.push({ name: method.endpointName, args });
      return (await method(args)).data;
    },
  };
  return {
    github, requests, paginations, writes, infos,
    context: { repo: REPO, payload: { workflow_run: { id: 100 } } },
    core: { info: message => infos.push(message) },
    pullRequest: pr, pullNumber: pr.number,
    loadLog: options.loadLog || (async () => 'AssertionError: expected 2, received 1'),
  };
}

function botComment(id, body, user = {}) {
  return { id, body, user: { login: 'github-actions[bot]', type: 'Bot', ...user } };
}

function logResponse(chunks, onCancel = () => {}) {
  let index = 0;
  return {
    ok: true,
    body: { getReader: () => ({
      read: async () => index < chunks.length ? { done: false, value: chunks[index++] } : { done: true },
      cancel: async () => onCancel(),
    }) },
  };
}

test('copied-branch runs resolve the branch PR and ignore supplied associations', async () => {
  const h = harness({ run: workflowRun('pr.yaml', { pull_requests: [{ number: 999 }] }) });
  assert.equal((await resolvePullRequest(h)).number, 4542);
  assert.deepEqual(h.requests.filter(r => r.name === 'getPullRequest').map(r => r.args.pull_number), [4542]);
  assert.equal(h.paginations.length, 0);
});

test('fork pre-merge runs fall back to paginated commit associations when payload is empty', async () => {
  const h = harness({ run: workflowRun('pre-merge.yml', { pull_requests: [] }), associations: [{ number: 4542 }] });
  assert.equal((await resolvePullRequest(h)).number, 4542);
  assert.deepEqual(h.paginations, [{
    name: 'listPullRequestsAssociatedWithCommit',
    args: { ...REPO, commit_sha: SHA, per_page: 100 },
  }]);
});

test('empty fork associations fall back to the exact source owner and branch', async () => {
  const h = harness({ run: workflowRun('pre-merge.yml', { pull_requests: [] }), sourcePRs: [pullRequest()] });
  assert.equal((await resolvePullRequest(h)).number, 4542);
  assert.deepEqual(h.paginations.map(r => r.name), ['listPullRequestsAssociatedWithCommit', 'listPullRequests']);
  assert.deepEqual(h.paginations[1].args, { ...REPO, state: 'open', head: 'contributor:feature', per_page: 100 });
});

test('fork source fallback still rejects stale and ambiguous PRs', async t => {
  await t.test('stale head', async () => {
    const stale = pullRequest({ head: { ...pullRequest().head, sha: 'b'.repeat(40) } });
    const h = harness({ run: workflowRun('pre-merge.yml', { pull_requests: [] }), sourcePRs: [stale], pr: stale });
    assert.equal(await resolvePullRequest(h), null);
  });
  await t.test('different bases', async () => {
    const h = harness({
      run: workflowRun('pre-merge.yml', { pull_requests: [] }),
      sourcePRs: [pullRequest(), pullRequest({ number: 4543, base: { ref: 'release', repo: { full_name: FULL_NAME } } })],
      getPullRequest: args => pullRequest({ number: args.pull_number }),
    });
    assert.equal(await resolvePullRequest(h), null);
  });
});

test('pre-merge payload associations avoid a fallback and duplicate associations are deduplicated', async () => {
  const h = harness({ run: workflowRun('pre-merge.yml', { pull_requests: [{ number: 4542 }, { number: 4542 }] }) });
  assert.equal((await resolvePullRequest(h)).number, 4542);
  assert.equal(h.requests.filter(r => r.name === 'getPullRequest').length, 1);
  assert.equal(h.paginations.length, 0);
});

test('ambiguous matching PR associations refuse publication', async () => {
  const h = harness({
    run: workflowRun('pre-merge.yml', { pull_requests: [{ number: 4542 }, { number: 4543 }] }),
    getPullRequest: args => pullRequest({ number: args.pull_number }),
  });
  assert.equal(await resolvePullRequest(h), null);
});

test('resolution refuses stale, closed, foreign and invalid workflow runs', async t => {
  const cases = [
    ['stale head', { pr: pullRequest({ head: { ...pullRequest().head, sha: 'b'.repeat(40) } }) }],
    ['closed PR', { pr: pullRequest({ state: 'closed' }) }],
    ['foreign base', { pr: pullRequest({ base: { repo: { full_name: 'other/dynamo' } } }) }],
    ['foreign workflow repository', { run: workflowRun('pr.yaml', { repository: { full_name: 'other/dynamo' } }) }],
    ['foreign copied-branch head repository', { run: workflowRun('pr.yaml', { head_repository: { full_name: 'other/dynamo' } }) }],
    ['unrecognized workflow', { run: workflowRun('dangerous.yml') }],
    ['wrong workflow event', { run: workflowRun('pr.yaml', { event: 'pull_request_target' }) }],
    ['ordinary push branch', { run: workflowRun('pr.yaml', { head_branch: 'main' }) }],
    ['noncanonical copied branch', { run: workflowRun('pr.yaml', { head_branch: 'pull-request/04542' }) }],
    ['wrong fork owner', { run: workflowRun('pre-merge.yml', { head_repository: { full_name: 'other/dynamo' } }) }],
    ['wrong fork branch', { run: workflowRun('pre-merge.yml', { head_branch: 'other-feature' }) }],
  ];
  for (const [name, options] of cases) {
    await t.test(name, async () => assert.equal(await resolvePullRequest(harness(options)), null));
  }
});

test('collection selects the newest matching run per workflow and paginates every source', async () => {
  const runs = Object.fromEntries(WORKFLOW_FILES.map((file, index) => [file, [
    workflowRun(file, { id: 100 + index }),
    workflowRun(file, { id: 200 + index }),
    workflowRun(file, { id: 999, head_sha: 'b'.repeat(40) }),
    workflowRun(file, { id: 998, repository: { full_name: 'other/dynamo' } }),
  ]]));
  const jobs = { 200: [job(1)], 201: [job(2, { conclusion: 'success' })], 202: [] };
  const h = harness({ runs, jobs });
  const summary = await collectSummary(h);
  assert.deepEqual(summary.workflows.map(w => w.run.id), [200, 201, 202]);
  assert.deepEqual(summary.workflows.map(w => w.failures.length), [1, 0, 0]);
  const runPages = h.paginations.filter(r => r.name === 'listWorkflowRuns');
  assert.equal(runPages.length, 3);
  for (const request of runPages) {
    assert.equal(request.args.per_page, 100);
    assert.equal(request.args.head_sha, SHA);
    assert.equal(request.args.branch, request.args.workflow_id === 'pre-merge.yml' ? 'feature' : 'pull-request/4542');
  }
  assert.deepEqual(h.paginations.filter(r => r.name === 'listJobsForWorkflowRun').map(r => r.args),
    [200, 201, 202].map(run_id => ({ ...REPO, run_id, filter: 'latest', per_page: 100 })));
});

test('pre-merge collection excludes runs explicitly associated with another PR or base', async () => {
  const h = harness({ runs: { 'pre-merge.yml': [
    workflowRun('pre-merge.yml', { id: 300, pull_requests: [{ number: 999 }] }),
    workflowRun('pre-merge.yml', { id: 200, pull_requests: [{ number: 4542, base: { ref: 'release' } }] }),
    workflowRun('pre-merge.yml', { id: 100, pull_requests: [{ number: 4542, base: { ref: 'main' } }] }),
  ] } });
  assert.equal((await collectSummary(h)).workflows[0].run.id, 100);
});

test('unassociated pre-merge collection requires an unambiguous current source PR', async t => {
  await t.test('unique source PR', async () => {
    const h = harness({
      runs: { 'pre-merge.yml': [workflowRun('pre-merge.yml', { pull_requests: [] })] }, sourcePRs: [pullRequest()],
    });
    assert.equal((await collectSummary(h)).workflows[0].run.id, 100);
    assert.equal(h.paginations.find(r => r.name === 'listPullRequests').args.head, 'contributor:feature');
  });
  await t.test('ambiguous sources fall back to the older explicitly associated run', async () => {
    const h = harness({
      runs: { 'pre-merge.yml': [
        workflowRun('pre-merge.yml', { id: 200, pull_requests: [] }), workflowRun('pre-merge.yml', { id: 100 }),
      ] },
      sourcePRs: [pullRequest(), pullRequest({ number: 4543 })],
    });
    assert.equal((await collectSummary(h)).workflows[0].run.id, 100);
  });
});

test('collection preserves absent workflows without requesting their jobs', async () => {
  const h = harness();
  const summary = await collectSummary(h);
  assert.equal(summary.workflows.length, 3);
  assert(summary.workflows.every(w => w.run === null && w.jobs.length === 0 && w.failures.length === 0));
  assert(!h.requests.some(r => r.name === 'listJobsForWorkflowRun'));
});

test('latest rerun jobs include prior successes without downloading their logs', async () => {
  const loaded = [];
  const h = harness({
    runs: { 'pr.yaml': [workflowRun('pr.yaml', { run_attempt: 2 })] },
    jobs: { 100: [job(1, { conclusion: 'success' }), job(2, { conclusion: 'failure' })] },
    loadLog: async id => { loaded.push(id); return 'AssertionError: test failed'; },
  });
  const summary = await collectSummary(h);
  assert.equal(summary.workflows[1].jobs.length, 2);
  assert.deepEqual(loaded, [2]);
  assert.equal(h.paginations.find(r => r.name === 'listJobsForWorkflowRun').args.filter, 'latest');
});

test('transient log failures retain failed jobs and still publish a summary', async () => {
  const h = harness({
    runs: { 'pr.yaml': [workflowRun()] }, jobs: { 100: [job()] },
    loadLog: async () => { throw new Error('https://storage.invalid/?secret=signed-token'); },
  });
  const body = await publishSummary(h);
  assert.equal(h.writes.length, 1);
  assert(body.includes(job().html_url));
  assert.match(body, /1 failed/);
  assert(!body.includes('signed-token'));
  assert(h.infos.some(message => message.includes('job 1')));
  assert(h.infos.every(message => !message.includes('signed-token')));
});

test('log download cap is shared across workflows and retains all failures', async () => {
  const loaded = [];
  const h = harness({
    runs: { 'pre-merge.yml': [workflowRun('pre-merge.yml')], 'pr.yaml': [workflowRun('pr.yaml', { id: 101 })] },
    jobs: { 100: Array.from({ length: 7 }, (_, n) => job(n + 1)), 101: Array.from({ length: 7 }, (_, n) => job(n + 8)) },
    loadLog: async id => { loaded.push(id); return 'AssertionError: failed'; },
  });
  const summary = await collectSummary(h);
  assert.deepEqual(loaded, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  assert.equal(summary.workflows.flatMap(w => w.failures).length, 14);
  assert(summary.workflows[1].failures.slice(3).every(f => f.unavailable.includes('limit')));
});

test('sticky comments require the leading marker and exact bot identity', async () => {
  const h = harness({ comments: [
    botComment(1, `${MARKER}\nHuman text`, { login: 'maintainer', type: 'User' }),
    botComment(2, `${MARKER}\nImpersonator`, { type: 'User' }),
    botComment(3, `${MARKER}\nAnother bot`, { login: 'other[bot]' }),
    botComment(4, `Human discussion\n${MARKER}`),
    botComment(5, `${MARKER}\nOld summary`),
  ] });
  const body = await publishSummary(h);
  assert.deepEqual(h.writes, [{ type: 'update', ...REPO, comment_id: 5, body }]);
  assert.equal(h.paginations.find(r => r.name === 'listComments').args.per_page, 100);
});

test('a copied marker on a human comment creates a separate bot summary', async () => {
  const h = harness({ comments: [botComment(1, `${MARKER}\nMy analysis`, { login: 'maintainer', type: 'User' })] });
  await publishSummary(h);
  assert.equal(h.writes.length, 1);
  assert.equal(h.writes[0].type, 'create');
  assert.equal(h.writes[0].issue_number, 4542);
});

test('identical sticky comment is a no-op and dry-run never writes', async () => {
  const draft = harness();
  const body = await publishSummary({ ...draft, dryRun: true });
  assert.equal(draft.writes.length, 0);
  assert(!draft.requests.some(r => r.name === 'listComments'));
  const h = harness({ comments: [botComment(1, body)] });
  assert.equal(await publishSummary(h), body);
  assert.equal(h.writes.length, 0);
});

test('publisher refuses an already-closed PR before collecting CI', async () => {
  const h = harness({ pr: pullRequest({ state: 'closed' }) });
  assert.equal(await publishSummary(h), null);
  assert.equal(h.paginations.length, 0);
  assert.equal(h.writes.length, 0);
});

test('publisher skips writes if PR head or state changes during collection', async t => {
  for (const [name, current] of [
    ['head', pullRequest({ head: { ...pullRequest().head, sha: 'b'.repeat(40) } })],
    ['state', pullRequest({ state: 'closed' })],
  ]) {
    await t.test(name, async () => {
      const h = harness({ getPullRequest: (_, count) => count === 1 ? pullRequest() : current });
      assert.equal(await publishSummary(h), null);
      assert.equal(h.writes.length, 0);
    });
  }
});

test('publisher skips writes when run identity, attempt or completion changes', async t => {
  for (const [name, changed] of [
    ['identity', { id: 101 }], ['attempt', { run_attempt: 2 }],
    ['status', { status: 'in_progress', conclusion: null }], ['conclusion', { conclusion: 'success' }],
  ]) {
    await t.test(name, async () => {
      const h = harness({
        getRuns: (args, count) => args.workflow_id === 'pr.yaml' ? [workflowRun('pr.yaml', count > 1 ? changed : {})] : [],
      });
      assert.equal(await publishSummary(h), null);
      assert.equal(h.writes.length, 0);
    });
  }
});

test('signed log downloads use a bounded range and never forward the API credential', async () => {
  const calls = [];
  let cancellations = 0;
  const storageUrl = 'https://logs.example.invalid/job?signature=private';
  const result = await readJobLog({
    repo: REPO, jobId: 42, token: 'api-secret',
    fetchImpl: async (url, options) => {
      calls.push({ url, options });
      return calls.length === 1 ? { status: 302, headers: new Headers({ location: storageUrl }) }
        : logResponse([Buffer.from('first\n'), Buffer.from('last\n')], () => cancellations++);
    },
  });
  assert.equal(result, 'first\nlast\n');
  assert.equal(calls[0].url, 'https://api.github.com/repos/ai-dynamo/dynamo/actions/jobs/42/logs');
  assert.equal(calls[0].options.headers.Authorization, 'Bearer api-secret');
  assert.equal(calls[0].options.redirect, 'manual');
  assert.equal(calls[1].url, storageUrl);
  assert.deepEqual(calls[1].options.headers, { Range: 'bytes=-1048576' });
  assert(!JSON.stringify(calls[1]).includes('api-secret'));
  assert(calls.every(call => call.options.signal instanceof AbortSignal));
  assert.equal(cancellations, 1);
});

test('log reader aborts and cancels storage responses that ignore the size bound', async () => {
  let calls = 0;
  let cancellations = 0;
  await assert.rejects(readJobLog({
    repo: REPO, jobId: 42, token: 'api-secret',
    fetchImpl: async () => ++calls === 1
      ? { status: 302, headers: new Headers({ location: 'https://logs.example.invalid/job' }) }
      : logResponse([Buffer.alloc(1024 * 1024), Buffer.from('overflow')], () => cancellations++),
  }), /download limit/);
  assert.equal(cancellations, 1);
});

test('log reader refuses insecure redirects and unsuccessful storage responses', async t => {
  await t.test('insecure location', async () => {
    let calls = 0;
    await assert.rejects(readJobLog({
      repo: REPO, jobId: 42, token: 'api-secret',
      fetchImpl: async () => { calls++; return { status: 302, headers: new Headers({ location: 'http://logs.example.invalid/job' }) }; },
    }), /unavailable/);
    assert.equal(calls, 1);
  });
  await t.test('storage failure', async () => {
    let calls = 0;
    await assert.rejects(readJobLog({
      repo: REPO, jobId: 42, token: 'api-secret',
      fetchImpl: async () => ++calls === 1
        ? { status: 302, headers: new Headers({ location: 'https://logs.example.invalid/job' }) }
        : { ok: false, body: null },
    }), /download failed/);
  });
});

test('suffix log ranges discard a partial first line without exposing split credentials', async t => {
  const partial = 'partial-secret-value\r\nAssertionError: test failed\n';
  const cases = [
    { name: 'positive range start', status: 206, start: 100, input: partial, expected: 'AssertionError: test failed\n' },
    { name: 'missing content range', status: 206, input: partial, expected: 'AssertionError: test failed\n' },
    { name: 'partial range without newline', status: 206, start: 100, input: 'partial-secret-value', expected: '' },
    { name: 'unknown range without newline', status: 206, input: 'partial-secret-value', expected: '' },
    { name: 'range starting at byte zero', status: 206, start: 0, input: partial, expected: partial },
    { name: 'complete response', status: 200, input: partial, expected: partial },
  ];
  for (const { name, status, start, input, expected } of cases) {
    await t.test(name, async () => {
      let calls = 0;
      const headers = new Headers();
      if (start !== undefined) {
        const end = start + Buffer.byteLength(input) - 1;
        headers.set('content-range', `bytes ${start}-${end}/${end + 1}`);
      }
      const result = await readJobLog({
        repo: REPO, jobId: 42, token: 'api-secret',
        fetchImpl: async () => ++calls === 1
          ? { status: 302, headers: new Headers({ location: 'https://logs.example.invalid/job' }) }
          : { ...logResponse([Buffer.from(input.slice(0, 3)), Buffer.from(input.slice(3))]), status, headers },
      });
      assert.equal(result, expected);
    });
  }
});
