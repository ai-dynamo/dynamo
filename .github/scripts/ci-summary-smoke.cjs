// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { publishSummary, readJobLog } = require('./ci-summary.cjs');
const { renderSummary, extractFailureExcerpt } = require('./ci-summary-format.cjs');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const MARKER = '<!-- dynamo-ci-summary -->';
const EXAMPLE = '<!-- dynamo-ci-summary-smoke-example -->';

module.exports = async ({ github, context, core }) => {
  const pullNumber = 15363;
  const comments = async marker => (await github.paginate(github.rest.issues.listComments, {
    ...context.repo, issue_number: pullNumber, per_page: 100,
  })).filter(c => c.user?.login === 'github-actions[bot]' && c.user.type === 'Bot' && c.body?.startsWith(marker));
  const publish = async () => {
    for (let attempt = 0; attempt < 6; attempt++) {
      const body = await publishSummary({ github, context, core, pullNumber });
      if (body) return body;
      await sleep(5000);
    }
    throw new Error('PR/run snapshot kept changing during publication');
  };

  await publish();
  let reports = await comments(MARKER);
  assert.equal(reports.length, 1, 'exactly one real CI summary');
  const id = reports[0].id;
  const sentinel = '<!-- ci-summary-smoke-refresh-check -->';
  await github.rest.issues.updateComment({
    ...context.repo, comment_id: id, body: `${reports[0].body}\n${sentinel}`,
  });
  await publish();
  reports = await comments(MARKER);
  assert.equal(reports.length, 1);
  assert.equal(reports[0].id, id, 'update preserves comment ID');
  assert.ok(!reports[0].body.includes(sentinel), 'reporter replaced the outdated comment');
  await publish();
  reports = await comments(MARKER);
  assert.equal(reports.length, 1, 'repeat publication does not duplicate the comment');
  assert.equal(reports[0].id, id);

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...context.repo, run_id: context.runId, filter: 'latest', per_page: 100,
  });
  const fixture = jobs.find(job => job.name === 'Intentional CPU failure fixture');
  assert.equal(fixture?.conclusion, 'failure', 'fixture really failed on a CPU runner');
  let excerpt;
  for (let attempt = 0; attempt < 12; attempt++) {
    try {
      excerpt = extractFailureExcerpt(await readJobLog({
        repo: context.repo, jobId: fixture.id, token: process.env.GH_TOKEN,
      }));
      break;
    } catch {
      await sleep(5000);
    }
  }
  assert.ok(excerpt?.includes('ci_summary_smoke::intentional_failure'), 'real job log was extracted');
  assert.ok(excerpt.includes('[REDACTED]'), 'credential-shaped value was redacted');
  assert.ok(!excerpt.includes('smoke-only-placeholder'));
  const { data: pr } = await github.rest.pulls.get({ ...context.repo, pull_number: pullNumber });
  const { data: run } = await github.rest.actions.getWorkflowRun({ ...context.repo, run_id: context.runId });
  const rendered = renderSummary({ pullRequest: pr, workflows: [{
    name: 'Intentional CPU fixture (smoke test only)', run, jobs: [fixture],
    failures: [{ job: fixture, excerpt }],
  }] });
  assert.ok(rendered.includes('&lt;tag&gt;'), 'diagnostic HTML is escaped');
  const exampleBody = `${EXAMPLE}\n**Intentional smoke-test fixture, separate from this PR's CI results.**\n\n${rendered.replace(`${MARKER}\n`, '')}`;
  const examples = await comments(EXAMPLE);
  let example;
  if (examples.length) {
    ({ data: example } = await github.rest.issues.updateComment({ ...context.repo, comment_id: examples[0].id, body: exampleBody }));
  } else {
    ({ data: example } = await github.rest.issues.createComment({ ...context.repo, issue_number: pullNumber, body: exampleBody }));
  }
  const { data: saved } = await github.rest.issues.getComment({ ...context.repo, comment_id: example.id });
  assert.equal(saved.body, exampleBody, 'failure example persisted');
  core.info(`Real CI summary: ${reports[0].html_url}`);
  core.info(`Intentional failure example: ${saved.html_url}`);
  await core.summary.addHeading('CI summary smoke test passed').addList([
    `Created/refreshed exactly one bot summary: ${reports[0].html_url}`,
    'Verified an outdated comment is updated in place, then repeated publication without duplicates.',
    `Verified extraction, redaction and HTML escaping of a real CPU failure: ${saved.html_url}`,
    'Production reporter source is unchanged; only the default-branch workflow_run trigger remains untested.',
  ]).write();
};
