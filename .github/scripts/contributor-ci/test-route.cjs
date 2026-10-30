// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const {eligibleAuthors, plan} = require('./route.cjs');
const sha = 'a'.repeat(40);
function fixture() {
  return {pr: {state: 'open', number: 7, base: {ref: 'main', repo: {full_name: 'ai-dynamo/dynamo'}},
    head: {sha}, merge_commit_sha: 'b'.repeat(40)}, files: [{filename: 'components/model.py'}],
  filters: {core: ['components/**'], ignore: ['*.md']}, runId: '123', attempt: '1', mergeSha: 'b'.repeat(40)};
}
test('runtime matrix uses contributor repositories and separate CPU/GPU pools', () => {
  const p = plan(fixture());
  assert.ok(p.builds.include.length);
  assert.ok(p.tests.include.some(t => t.tier === 'gpu-2'));
  for (const build of p.builds.include) {
    assert.match(build.image, /amazonaws.com\/dynamo-pr-ci:pr-7-a{40}-r123-a1-/);
    assert.match(build.cache, /\/dynamo-pr-ci\/cache:pr-7-/);
  }
  for (const t of p.tests.include) {
    assert.ok(t.labels.includes('trusted_contributors'));
    assert.match(t.markers, /not triton_qa/);
    assert.ok(!t.labels.some(label => label.startsWith('prod-')));
  }
});
test('rerun image tags do not overwrite an immutable prior attempt', () => {
  const a = fixture(), b = fixture(); b.attempt = '2';
  assert.notEqual(plan(a).builds.include[0].image, plan(b).builds.include[0].image);
});
test('workflow changes and renamed workflow files reject routing', () => {
  for (const file of [{filename: '.github/workflows/pr.yaml'}, {filename: 'gone.txt', previous_filename: '.github/old.yml'}]) {
    const input = fixture(); input.files = [file];
    assert.throws(() => plan(input), /Workflow edits/);
  }
});
test('stale merge and foreign base reject routing', () => {
  const stale = fixture(); stale.mergeSha = 'c'.repeat(40);
  assert.throws(() => plan(stale), /identity changed/);
  const foreign = fixture(); foreign.pr.base.repo.full_name = 'someone/dynamo';
  assert.throws(() => plan(foreign), /Dynamo main/);
});
test('unclassified changes fail instead of silently omitting CI', () => {
  const input = fixture(); input.files = [{filename: 'unknown.bin'}];
  assert.throws(() => plan(input), /Uncovered/);
});
test('docs changes produce an explicit empty hardware plan', () => {
  const input = fixture(); input.files = [{filename: 'README.md'}];
  assert.equal(plan(input).has_builds, false);
});
test('standing is repository scoped and duplicate identities fail', () => {
  const author = {github: 'Contributor', roles: [{repo: 'dynamo', role: 'trusted_contributor'}]};
  assert.deepEqual([...eligibleAuthors({contributors: [author]})], ['contributor']);
  assert.throws(() => eligibleAuthors({contributors: [author, {...author, github: 'contributor'}]}), /Duplicate/);
  assert.throws(() => eligibleAuthors({contributors: [{...author, roles: [{repo: 'other', role: 'maintainer'}]}]}), /repository/);
});
