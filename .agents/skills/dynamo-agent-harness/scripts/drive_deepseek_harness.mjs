#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Run one pinned DSH headless task through a capture relay in front of Dynamo. */

import { createHash } from 'node:crypto'
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { spawn } from 'node:child_process'
import process from 'node:process'
import { pathToFileURL } from 'node:url'

const DEFAULT_DSH_PACKAGE = '@deepseek-ai/dsh@0.1.0-rc.8'
const DEFAULT_PNPM_VERSION = '11.7.0'
const HOP_BY_HOP_HEADERS = new Set([
  'connection',
  'content-length',
  'host',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
])

function usage() {
  return `Usage: drive_deepseek_harness.mjs --base-url URL --model MODEL --task TASK [options]

Options:
  --capture PATH             Redacted JSONL request evidence (default: dsh-request-trace.jsonl)
  --cwd PATH                 DSH workspace (default: current directory)
  --dsh-bin PATH             Built patched DSH bin.js instead of the pinned npm package
  --dsh-home PATH            Empty persistent DSH home (default: a removed temporary directory)
  --dsh-package SPEC         Package used by npx (default: ${DEFAULT_DSH_PACKAGE})
  --final-timeout-ms N       Terminal request timeout (default: 5000)
  --max-tokens N             DSH output limit (default: 4096)
  --pnpm-version VERSION     Corepack pnpm used for the published package (default: ${DEFAULT_PNPM_VERSION})
  --session-final            Drain observed sessions through ThunderAgent on exit
  --help                     Show this help
`
}

function valueAfter(argv, index, name) {
  const value = argv[index + 1]
  if (value === undefined || value.startsWith('--')) throw new Error(`${name} requires a value`)
  return value
}

export function parseArgs(argv) {
  const options = {
    baseUrl: undefined,
    capture: resolve('dsh-request-trace.jsonl'),
    cwd: process.cwd(),
    dshBin: undefined,
    dshHome: undefined,
    dshPackage: DEFAULT_DSH_PACKAGE,
    finalTimeoutMs: 5_000,
    maxTokens: 4_096,
    model: undefined,
    pnpmVersion: DEFAULT_PNPM_VERSION,
    sessionFinal: false,
    task: undefined,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === '--help') return { ...options, help: true }
    if (argument === '--session-final') {
      options.sessionFinal = true
      continue
    }
    const name = argument
    const value = valueAfter(argv, index, name)
    index += 1
    switch (name) {
      case '--base-url': options.baseUrl = value; break
      case '--capture': options.capture = resolve(value); break
      case '--cwd': options.cwd = resolve(value); break
      case '--dsh-bin': options.dshBin = resolve(value); break
      case '--dsh-home': options.dshHome = resolve(value); break
      case '--dsh-package': options.dshPackage = value; break
      case '--final-timeout-ms': options.finalTimeoutMs = Number(value); break
      case '--max-tokens': options.maxTokens = Number(value); break
      case '--model': options.model = value; break
      case '--pnpm-version': options.pnpmVersion = value; break
      case '--task': options.task = value; break
      default: throw new Error(`unknown argument: ${name}`)
    }
  }
  if (!options.baseUrl) throw new Error('--base-url is required')
  if (!options.model?.trim()) throw new Error('--model is required')
  if (!options.task?.trim()) throw new Error('--task is required')
  if (!Number.isSafeInteger(options.maxTokens) || options.maxTokens <= 0) throw new Error('--max-tokens must be a positive integer')
  if (!Number.isSafeInteger(options.finalTimeoutMs) || options.finalTimeoutMs <= 0) throw new Error('--final-timeout-ms must be a positive integer')
  if (!existsSync(options.cwd) || !statSync(options.cwd).isDirectory()) throw new Error(`--cwd is not a directory: ${options.cwd}`)
  if (options.dshBin !== undefined && !existsSync(options.dshBin)) throw new Error(`--dsh-bin does not exist: ${options.dshBin}`)
  return options
}

export function normalizeBaseUrl(value) {
  const parsed = new URL(value)
  if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('--base-url must use HTTP(S)')
  parsed.search = ''
  parsed.hash = ''
  parsed.pathname = parsed.pathname.replace(/\/$/, '')
  if (!parsed.pathname.endsWith('/v1')) parsed.pathname += '/v1'
  return parsed
}

function hashValue(value) {
  return `sha256:${createHash('sha256').update(value).digest('hex')}`
}

function redactedHeaders(headers) {
  const output = {}
  for (const [name, rawValue] of Object.entries(headers)) {
    const value = Array.isArray(rawValue) ? rawValue.join(', ') : rawValue
    if (value === undefined) continue
    const lower = name.toLowerCase()
    if (lower === 'authorization' || lower === 'x-api-key') output[lower] = '<redacted>'
    else if (lower === 'x-deepseek-harness-user-id') output[lower] = hashValue(value)
    else if (lower.startsWith('x-deepseek-harness-') || ['content-type', 'user-agent'].includes(lower)) output[lower] = value
  }
  return output
}

function parseBody(body) {
  if (body.length === 0) return null
  try {
    return JSON.parse(body.toString('utf8'))
  } catch {
    return { encoding: 'base64', data: body.toString('base64') }
  }
}

function evidenceWriter(path) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, '', { mode: 0o600 })
  return value => appendFileSync(path, `${JSON.stringify({ timestamp: new Date().toISOString(), ...value })}\n`)
}

function readRequest(request) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = []
    request.on('data', chunk => chunks.push(chunk))
    request.on('end', () => resolveBody(Buffer.concat(chunks)))
    request.on('error', rejectBody)
  })
}

function forwardHeaders(headers) {
  const output = new Headers()
  for (const [name, rawValue] of Object.entries(headers)) {
    if (HOP_BY_HOP_HEADERS.has(name.toLowerCase()) || rawValue === undefined) continue
    output.set(name, Array.isArray(rawValue) ? rawValue.join(', ') : rawValue)
  }
  return output
}

function responseHeaders(headers) {
  const output = {}
  for (const [name, value] of headers.entries()) {
    if (!HOP_BY_HOP_HEADERS.has(name.toLowerCase())) output[name] = value
  }
  return output
}

async function relayBody(upstream, downstream) {
  if (upstream.body === null) {
    downstream.end()
    return
  }
  const reader = upstream.body.getReader()
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (!downstream.write(value)) await new Promise(resolveDrain => downstream.once('drain', resolveDrain))
    }
  } finally {
    reader.releaseLock()
  }
  downstream.end()
}

export async function startRelay({ baseUrl, record }) {
  const upstreamOrigin = baseUrl.origin
  const sessions = new Map()
  const controllers = new Set()
  const server = createServer((request, response) => {
    const controller = new AbortController()
    controllers.add(controller)
    void (async () => {
      const body = await readRequest(request)
      const sessionId = request.headers['x-deepseek-harness-session-id']
      const parentSessionId = request.headers['x-deepseek-harness-parent-session-id']
      if (typeof sessionId === 'string') sessions.set(sessionId, typeof parentSessionId === 'string' ? parentSessionId : null)
      record({
        kind: 'request',
        method: request.method,
        path: request.url,
        headers: redactedHeaders(request.headers),
        body: parseBody(body),
      })
      const target = new URL(request.url ?? '/', upstreamOrigin)
      const upstream = await fetch(target, {
        method: request.method,
        headers: forwardHeaders(request.headers),
        body: ['GET', 'HEAD'].includes(request.method ?? '') ? undefined : body,
        redirect: 'manual',
        signal: controller.signal,
      })
      record({ kind: 'response', path: request.url, status: upstream.status })
      response.writeHead(upstream.status, responseHeaders(upstream.headers))
      await relayBody(upstream, response)
    })().catch(error => {
      record({ kind: 'relay_error', path: request.url, error: error instanceof Error ? error.message : String(error) })
      if (!response.headersSent) response.writeHead(502, { 'content-type': 'application/json' })
      response.end(JSON.stringify({ error: 'Dynamo relay failed' }))
    }).finally(() => controllers.delete(controller))
  })
  await new Promise((resolveListen, rejectListen) => {
    server.once('error', rejectListen)
    server.listen(0, '127.0.0.1', resolveListen)
  })
  const address = server.address()
  if (address === null || typeof address === 'string') throw new Error('relay did not bind a TCP port')
  const proxyBaseUrl = new URL(baseUrl.pathname, `http://127.0.0.1:${address.port}`)
  return {
    abortRequests: () => { for (const controller of controllers) controller.abort() },
    close: () => new Promise(resolveClose => server.close(resolveClose)),
    proxyBaseUrl,
    sessions,
  }
}

function prepareDshHome(options, proxyBaseUrl) {
  const temporary = options.dshHome === undefined
  const home = options.dshHome ?? mkdtempSync(join(tmpdir(), 'dsh-dynamo-'))
  if (!temporary) {
    if (existsSync(home) && readdirSync(home).length > 0) throw new Error(`--dsh-home must be empty: ${home}`)
    mkdirSync(home, { recursive: true })
  }
  const settings = {
    'agent-default-model': {
      provider: 'deepseek-official',
      model: options.model,
      reasoningEffort: 'off',
    },
    'llm-deepseek': {
      baseURL: proxyBaseUrl.toString().replace(/\/$/, ''),
      thinking: 'disabled',
      maxTokens: options.maxTokens,
      models: [{ id: options.model, name: options.model, maxTokens: options.maxTokens }],
    },
  }
  writeFileSync(join(home, 'settings.yaml'), `${JSON.stringify(settings, null, 2)}\n`, { mode: 0o600 })
  return { home, temporary }
}

function dshCommand(options) {
  if (options.dshBin !== undefined) return [process.execPath, options.dshBin, '--profile', 'headless', options.task]
  return ['corepack', `pnpm@${options.pnpmVersion}`, 'dlx', options.dshPackage, '--profile', 'headless', options.task]
}

function runChild(command, environment, cwd, onSignalReady) {
  return new Promise((resolveExit, rejectExit) => {
    const child = spawn(command[0], command.slice(1), { cwd, env: environment, stdio: 'inherit' })
    child.once('error', rejectExit)
    child.once('close', (code, signal) => resolveExit({ code, signal }))
    onSignalReady(child)
  })
}

async function sendSessionFinal({ apiKey, baseUrl, model, record, sessionId, timeoutMs }) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const endpoint = new URL(`${baseUrl.pathname.replace(/\/$/, '')}/chat/completions`, baseUrl.origin)
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${apiKey}`,
        'content-type': 'application/json',
        'x-dynamo-session-final': 'true',
        'x-dynamo-session-id': sessionId,
      },
      body: JSON.stringify({ model, messages: [{ role: 'user', content: '.' }], max_tokens: 1, stream: false }),
      signal: controller.signal,
    })
    const text = await response.text()
    record({ kind: 'session_final', session_id: sessionId, status: response.status })
    if (!response.ok) throw new Error(`session final for ${sessionId} returned ${response.status}: ${text.slice(0, 300)}`)
  } finally {
    clearTimeout(timeout)
  }
}

export async function run(options) {
  const baseUrl = normalizeBaseUrl(options.baseUrl)
  const record = evidenceWriter(options.capture)
  const relay = await startRelay({ baseUrl, record })
  let preparedHome
  try {
    preparedHome = prepareDshHome(options, relay.proxyBaseUrl)
  } catch (error) {
    await relay.close()
    throw error
  }
  const apiKey = process.env.DEEPSEEK_API_KEY || process.env.DYNAMO_API_KEY || 'dummy'
  const environment = {
    ...process.env,
    DEEPSEEK_API_KEY: apiKey,
    DEEPSEEK_BASE_URL: relay.proxyBaseUrl.toString().replace(/\/$/, ''),
    DSH_HOME: preparedHome.home,
    DSH_TELEMETRY_DISABLED: '1',
  }
  const command = dshCommand(options)
  record({
    kind: 'run_start',
    dsh: options.dshBin === undefined ? `${options.dshPackage} via pnpm@${options.pnpmVersion}` : options.dshBin,
    model: options.model,
    upstream: baseUrl.toString().replace(/\/$/, ''),
    session_final: options.sessionFinal,
  })
  let receivedSignal = null
  let childProcess = null
  let forceKillTimer = null
  const handlers = new Map()
  for (const signal of ['SIGINT', 'SIGTERM']) {
    const handler = () => {
      if (receivedSignal !== null) return
      receivedSignal = signal
      relay.abortRequests()
      childProcess?.kill(signal)
      forceKillTimer = setTimeout(() => childProcess?.kill('SIGKILL'), 5_000)
    }
    handlers.set(signal, handler)
    process.on(signal, handler)
  }

  let childExit = { code: 1, signal: null }
  let finalFailed = false
  try {
    try {
      childExit = await runChild(command, environment, options.cwd, child => { childProcess = child })
    } finally {
      await relay.close()
    }
    if (options.sessionFinal) {
      for (const sessionId of [...relay.sessions.keys()].sort()) {
        try {
          await sendSessionFinal({ apiKey, baseUrl, model: options.model, record, sessionId, timeoutMs: options.finalTimeoutMs })
        } catch (error) {
          finalFailed = true
          record({ kind: 'session_final_error', session_id: sessionId, error: error instanceof Error ? error.message : String(error) })
          console.error(`dsh Dynamo lifecycle error: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    }
  } finally {
    if (forceKillTimer !== null) clearTimeout(forceKillTimer)
    for (const [signal, handler] of handlers) process.off(signal, handler)
    if (preparedHome.temporary) rmSync(preparedHome.home, { recursive: true, force: true })
  }
  record({ kind: 'run_end', child_code: childExit.code, child_signal: childExit.signal, received_signal: receivedSignal, final_failed: finalFailed })
  if (finalFailed) return 1
  if (receivedSignal === 'SIGINT') return 130
  if (receivedSignal === 'SIGTERM') return 0
  return childExit.code ?? 1
}

export async function main(argv = process.argv.slice(2)) {
  try {
    const options = parseArgs(argv)
    if (options.help) {
      process.stdout.write(usage())
      return 0
    }
    return await run(options)
  } catch (error) {
    console.error(`drive_deepseek_harness: ${error instanceof Error ? error.message : String(error)}`)
    return 1
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  process.exitCode = await main()
}
