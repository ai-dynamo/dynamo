// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const {spawnSync} = require('node:child_process');
const test = require('node:test');
const {REGISTRY} = require('./route.cjs');

function runBuild(endpoint, target = 'runtime') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'contributor-build-'));
  try {
    const bin = path.join(root, 'bin'); fs.mkdirSync(bin);
    const script = path.join(root, '.github/scripts'); fs.mkdirSync(script, {recursive: true});
    fs.writeFileSync(path.join(script, 'route_buildkit.sh'),
      `#!/bin/bash\nprintf 'general_amd64=${endpoint}\\n' >> "$GITHUB_OUTPUT"\n`);
    for (const command of ['docker', 'aws', 'python3']) {
      const executable = path.join(bin, command);
      fs.writeFileSync(executable, command === 'docker'
        ? '#!/bin/bash\nprintf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
        : '#!/bin/bash\nexit 0\n');
      fs.chmodSync(executable, 0o755);
    }
    fs.writeFileSync(path.join(root, 'identity'), '');
    const result = spawnSync('bash', [path.join(__dirname, 'build.sh')], {
      cwd: root, encoding: 'utf8', env: {...process.env, PATH: `${bin}:${process.env.PATH}`,
        BUILD_FRAMEWORK: target === 'frontend' ? 'dynamo' : 'vllm', BUILD_TARGET: target,
        BUILD_ARCH: 'amd64', BUILD_CUDA: '13.0', BUILD_IMAGE: `${REGISTRY}/dynamo-pr-ci:pr-1-fixture`,
        BUILD_CACHE: `${REGISTRY}/dynamo-pr-ci/cache:pr-1-fixture`, GITHUB_RUN_ID: '123',
        GITHUB_RUN_ATTEMPT: '1', AWS_WEB_IDENTITY_TOKEN_FILE: path.join(root, 'identity'),
        AWS_ROLE_ARN: 'arn:aws:iam::210086341041:role/dynamo-pr-ci-builder',
        COMMAND_LOG: path.join(root, 'commands')}});
    return {...result, commands: fs.existsSync(path.join(root, 'commands'))
      ? fs.readFileSync(path.join(root, 'commands'), 'utf8') : ''};
  } finally { fs.rmSync(root, {recursive: true, force: true}); }
}

test('build and test images use isolated cache and scoped build secrets', () => {
  const result = runBuild('tcp://buildkit-amd64-0.buildkit-amd64-headless.dynamo-pr-buildkit.svc.cluster.local:1234');
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.commands, /--driver remote/);
  assert.match(result.commands, /dynamo-pr-ci\/cache:pr-1-fixture/);
  assert.match(result.commands, /id=aws-web-identity-token/);
  assert.match(result.commands, /dynamo-pr-ci:pr-1-fixture-test/);
  assert.doesNotMatch(result.commands, /--driver kubernetes|nvcr.io/);
});
test('a production worker endpoint cannot become a fallback', () => {
  const result = runBuild('tcp://buildkit-amd64-0.buildkit-amd64-headless.buildkit.svc.cluster.local:1234');
  assert.notEqual(result.status, 0);
  assert.equal(result.commands, '');
});
test('frontend builds its EPP dependency in the contributor repository', () => {
  const result = runBuild('tcp://buildkit-amd64-0.buildkit-amd64-headless.dynamo-pr-buildkit.svc.cluster.local:1234', 'frontend');
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.commands, /--build-context dynamo=\./);
  assert.match(result.commands, /EPP_IMAGE=.*dynamo-pr-ci:pr-1-fixture-epp/);
});
