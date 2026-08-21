#!/usr/bin/env node
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Run one pinned DSH headless task through a capture relay in front of Dynamo. */

import { createHash } from 'node:crypto'
import { execFileSync, spawn } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { constants as osConstants, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { pathToFileURL } from 'node:url'

const DEFAULT_DSH_PACKAGE = '@deepseek-ai/dsh@0.1.0-rc.8'
const DEFAULT_PNPM_VERSION = '11.7.0'
const CHILD_ENV_ALLOWLIST = [
  'LANG',
  'LC_ALL',
  'LC_CTYPE',
  'NO_COLOR',
  'PATH',
  'SSL_CERT_DIR',
  'SSL_CERT_FILE',
  'TERM',
  'TMP',
  'TMPDIR',
  'TEMP',
  'TZ',
]
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
  --api-key-env NAME         Explicit credential env var (default: DYNAMO_API_KEY)
  --canonicalize-dynamo-headers
                             Also send canonical identity headers to older Dynamo
  --capture PATH             Redacted JSONL request evidence (default: dsh-request-trace.jsonl)
  --cwd PATH                 DSH workspace (default: current directory)
  --dsh-bin PATH             Built patched DSH bin.js instead of the pinned npm package
  --dsh-home PATH            Empty persistent DSH home (default: a removed temporary directory)
  --dsh-package SPEC         Package used by pnpm dlx (default: ${DEFAULT_DSH_PACKAGE})
  --final-timeout-ms N       Terminal request timeout (default: 5000)
  --max-tokens N             DSH output limit (default: 4096)
  --pnpm-version VERSION     Corepack pnpm used for the published package (default: ${DEFAULT_PNPM_VERSION})
  --session-final            Drain observed sessions through ThunderAgent on exit
  --overwrite-capture        Replace an existing capture file
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
    apiKeyEnv: 'DYNAMO_API_KEY',
    apiKeyEnvExplicit: false,
    baseUrl: undefined,
    canonicalizeDynamoHeaders: false,
    capture: resolve('dsh-request-trace.jsonl'),
    cwd: process.cwd(),
    dshBin: undefined,
    dshHome: undefined,
    dshPackage: DEFAULT_DSH_PACKAGE,
    finalTimeoutMs: 5_000,
    maxTokens: 4_096,
    model: undefined,
    overwriteCapture: false,
    pnpmVersion: DEFAULT_PNPM_VERSION,
    sessionFinal: false,
    task: undefined,
  }
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index]
    if (argument === '--help') return { ...options, help: true }
    if (argument === '--canonicalize-dynamo-headers' || argument === '--overwrite-capture' || argument === '--session-final') {
      if (argument === '--canonicalize-dynamo-headers') options.canonicalizeDynamoHeaders = true
      if (argument === '--overwrite-capture') options.overwriteCapture = true
      if (argument === '--session-final') options.sessionFinal = true
      continue
    }
    const name = argument
    const value = valueAfter(argv, index, name)
    index += 1
    switch (name) {
      case '--api-key-env': options.apiKeyEnv = value; options.apiKeyEnvExplicit = true; break
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
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(options.apiKeyEnv)) throw new Error('--api-key-env must be an environment variable name')
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
  const pathname = parsed.pathname.replace(/\/+$/, '')
  parsed.pathname = pathname.endsWith('/v1') ? pathname : `${pathname}/v1`
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

function evidenceWriter(path, overwrite) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, '', { flag: overwrite ? 'w' : 'wx', mode: 0o600 })
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

function forwardHeaders(headers, canonicalizeDynamoHeaders) {
  const output = new Headers()
  for (const [name, rawValue] of Object.entries(headers)) {
    if (HOP_BY_HOP_HEADERS.has(name.toLowerCase()) || rawValue === undefined) continue
    output.set(name, Array.isArray(rawValue) ? rawValue.join(', ') : rawValue)
  }
  if (canonicalizeDynamoHeaders) {
    const sessionId = headers['x-deepseek-harness-session-id']
    const parentSessionId = headers['x-deepseek-harness-parent-session-id']
    if (typeof sessionId === 'string' && sessionId.trim() !== '') {
      output.set('x-dynamo-session-id', sessionId)
    }
    if (typeof parentSessionId === 'string' && parentSessionId.trim() !== '') {
      output.set('x-dynamo-parent-session-id', parentSessionId)
    }
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

export async function startRelay({ baseUrl, canonicalizeDynamoHeaders, record }) {
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
        headers: forwardHeaders(request.headers, canonicalizeDynamoHeaders),
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
    const child = spawn(command[0], command.slice(1), {
      cwd,
      detached: process.platform !== 'win32',
      env: environment,
      stdio: 'inherit',
    })
    child.once('error', rejectExit)
    child.once('close', (code, signal) => resolveExit({ code, signal }))
    onSignalReady(child)
  })
}

function childEnvironment({ apiKey, home, proxyBaseUrl }) {
  const environment = {}
  for (const name of CHILD_ENV_ALLOWLIST) {
    if (process.env[name] !== undefined) environment[name] = process.env[name]
  }
  return {
    ...environment,
    CI: '1',
    COREPACK_ENABLE_DOWNLOAD_PROMPT: '0',
    DEEPSEEK_API_KEY: apiKey,
    DEEPSEEK_BASE_URL: proxyBaseUrl.toString().replace(/\/$/, ''),
    DSH_HOME: home,
    DSH_TELEMETRY_DISABLED: '1',
    HOME: home,
    XDG_CACHE_HOME: join(home, '.cache'),
    XDG_CONFIG_HOME: join(home, '.config'),
    XDG_DATA_HOME: join(home, '.local', 'share'),
    XDG_STATE_HOME: join(home, '.local', 'state'),
  }
}

function signalExitCode(signal) {
  const number = osConstants.signals[signal]
  return typeof number === 'number' ? 128 + number : 1
}

function processTable() {
  if (process.platform === 'linux' && existsSync('/proc')) {
    const rows = []
    for (const name of readdirSync('/proc')) {
      if (!/^\d+$/.test(name)) continue
      try {
        const stat = readFileSync(`/proc/${name}/stat`, 'utf8')
        const commandEnd = stat.lastIndexOf(')')
        const fields = stat.slice(commandEnd + 2).trim().split(/\s+/)
        const pid = Number(name)
        const parent = Number(fields[1])
        const session = Number(fields[3])
        if (Number.isSafeInteger(pid) && Number.isSafeInteger(parent) && Number.isSafeInteger(session)) {
          rows.push({ parent, pid, session })
        }
      } catch {
        // Processes can exit while /proc is being scanned.
      }
    }
    return rows
  }
  try {
    const output = execFileSync('/bin/ps', ['-axo', 'pid=,ppid=,sess='], { encoding: 'utf8' })
    return output.trim().split('\n').flatMap(row => {
      const [pidText, parentText, sessionText] = row.trim().split(/\s+/)
      const pid = Number(pidText)
      const parent = Number(parentText)
      const session = Number(sessionText)
      return Number.isSafeInteger(pid) && Number.isSafeInteger(parent) && Number.isSafeInteger(session)
        ? [{ parent, pid, session }]
        : []
    })
  } catch {
    return []
  }
}

function descendantProcessIds(rootProcessIds, sessionId) {
  const children = new Map()
  const sameSession = []
  for (const { parent, pid, session } of processTable()) {
    const values = children.get(parent) ?? []
    values.push(pid)
    children.set(parent, values)
    if (session === sessionId && pid !== sessionId) sameSession.push(pid)
  }
  const descendants = [...sameSession]
  const visited = new Set(rootProcessIds)
  for (const pid of sameSession) visited.add(pid)
  const pending = [...rootProcessIds, ...sameSession]
  while (pending.length > 0) {
    const parent = pending.pop()
    for (const pid of children.get(parent) ?? []) {
      if (visited.has(pid)) continue
      visited.add(pid)
      descendants.push(pid)
      pending.push(pid)
    }
  }
  return descendants
}

function processIdIsAlive(pid) {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ESRCH') return false
    if (error instanceof Error && 'code' in error && error.code === 'EPERM') return true
    throw error
  }
}

function assertProcessTreeSupport() {
  if (process.platform === 'win32') throw new Error('the DSH relay requires POSIX process-tree signaling')
  if (process.platform === 'linux' && existsSync('/proc/self/stat')) return
  if (!existsSync('/bin/ps')) throw new Error('the DSH relay requires /bin/ps from procps or the host operating system')
  try {
    execFileSync('/bin/ps', ['-axo', 'pid=,ppid=,sess='], { stdio: 'ignore' })
  } catch {
    throw new Error('/bin/ps must support the pid, ppid, and sess output fields')
  }
}

function signalChildTree(child, signal, trackedProcessIds) {
  if (child === null || child.pid === undefined) return
  trackedProcessIds.add(child.pid)
  for (const pid of descendantProcessIds(trackedProcessIds, child.pid)) trackedProcessIds.add(pid)
  if (process.platform === 'win32') {
    if (child.exitCode === null && child.signalCode === null) child.kill(signal)
  } else {
    for (const pid of [...trackedProcessIds].sort((left, right) => right - left)) {
      try {
        process.kill(pid, signal)
      } catch (error) {
        if (!(error instanceof Error && 'code' in error && ['EPERM', 'ESRCH'].includes(error.code))) throw error
      }
    }
  }
}

function childTreeIsAlive(child, trackedProcessIds) {
  if (child === null || child.pid === undefined) return false
  if (process.platform === 'win32') return child.exitCode === null && child.signalCode === null
  return [...trackedProcessIds].some(processIdIsAlive)
}

async function waitForChildTreeExit(child, trackedProcessIds, timeoutMs) {
  const deadline = Date.now() + timeoutMs
  while (childTreeIsAlive(child, trackedProcessIds)) {
    if (Date.now() >= deadline) return false
    await new Promise(resolveWait => setTimeout(resolveWait, 25))
  }
  return true
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
  assertProcessTreeSupport()
  const baseUrl = normalizeBaseUrl(options.baseUrl)
  const record = evidenceWriter(options.capture, options.overwriteCapture)
  const relay = await startRelay({
    baseUrl,
    canonicalizeDynamoHeaders: options.canonicalizeDynamoHeaders,
    record,
  })
  let preparedHome
  try {
    preparedHome = prepareDshHome(options, relay.proxyBaseUrl)
  } catch (error) {
    await relay.close()
    throw error
  }
  const selectedApiKey = process.env[options.apiKeyEnv]
  if (options.apiKeyEnvExplicit && selectedApiKey === undefined) {
    await relay.close()
    if (preparedHome.temporary) rmSync(preparedHome.home, { recursive: true, force: true })
    throw new Error(`selected credential environment variable is unset: ${options.apiKeyEnv}`)
  }
  const apiKey = selectedApiKey ?? 'dummy'
  const environment = childEnvironment({ apiKey, home: preparedHome.home, proxyBaseUrl: relay.proxyBaseUrl })
  const command = dshCommand(options)
  record({
    api_key_env: options.apiKeyEnv,
    canonicalize_dynamo_headers: options.canonicalizeDynamoHeaders,
    kind: 'run_start',
    dsh: options.dshBin === undefined ? `${options.dshPackage} via pnpm@${options.pnpmVersion}` : options.dshBin,
    model: options.model,
    upstream: baseUrl.toString().replace(/\/$/, ''),
    session_final: options.sessionFinal,
  })
  let receivedSignal = null
  let childProcess = null
  let forceKillTimer = null
  const trackedProcessIds = new Set()
  const handlers = new Map()
  for (const signal of ['SIGINT', 'SIGTERM']) {
    const handler = () => {
      if (receivedSignal !== null) return
      receivedSignal = signal
      relay.abortRequests()
      try {
        signalChildTree(childProcess, signal, trackedProcessIds)
        record({ kind: 'process_tree_signal', signal, tracked_processes: trackedProcessIds.size })
      } catch (error) {
        record({ kind: 'process_tree_signal_error', signal, error: error instanceof Error ? error.message : String(error) })
      }
      forceKillTimer = setTimeout(() => {
        record({ kind: 'process_tree_force_kill', tracked_processes: trackedProcessIds.size })
        try {
          signalChildTree(childProcess, 'SIGKILL', trackedProcessIds)
        } catch (error) {
          record({ kind: 'process_tree_signal_error', signal: 'SIGKILL', error: error instanceof Error ? error.message : String(error) })
        }
      }, 5_000)
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
      if (receivedSignal !== null && childProcess !== null) {
        const stopped = await waitForChildTreeExit(childProcess, trackedProcessIds, 5_500)
        if (!stopped) record({ kind: 'process_tree_exit_error', error: 'tracked process tree remained after SIGKILL grace period' })
      }
      await relay.close()
    }
    if (options.sessionFinal) {
      const sessionIds = [...relay.sessions.keys()].sort()
      if (sessionIds.length === 0) {
        finalFailed = true
        const error = 'session final requested but zero DSH sessions reached Dynamo'
        record({ kind: 'session_final_error', error })
        console.error(`dsh Dynamo lifecycle error: ${error}`)
      }
      for (const sessionId of sessionIds) {
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
  if (receivedSignal !== null) return signalExitCode(receivedSignal)
  if (finalFailed) return 1
  if (childExit.code !== null) return childExit.code
  if (childExit.signal !== null) return signalExitCode(childExit.signal)
  return 1
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
