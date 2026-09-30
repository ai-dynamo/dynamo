# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{/*
Expand the name of the chart.
*/}}
{{- define "power-agent.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "power-agent.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "power-agent.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "power-agent.labels" -}}
helm.sh/chart: {{ include "power-agent.chart" . }}
{{ include "power-agent.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: power-agent
{{- end }}

{{/*
Selector labels
*/}}
{{- define "power-agent.selectorLabels" -}}
app.kubernetes.io/name: {{ include "power-agent.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "power-agent.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "power-agent.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Validate the production image reference. Exactly one of `image.tag` or
`image.digest` must be set — but only when the DaemonSet is the thing being
installed (daemonset.enabled, the default). Dev-pod mode uses dev.image.*
instead (validated by validateDevImageTag), so gating on daemonset.enabled
avoids forcing a dummy --set image.tag/digest on dev-only installs.

Rules per PR #9682 review (initial + two follow-ups):

  * Both empty       → fail. Chart removed the `:latest` fallback
                       deliberately; the operator must pin.
  * Both set         → fail. Ambiguous which one wins; refuse
                       rather than silently preferring one.
  * Either value has leading/trailing whitespace → fail. OCI tags
                       and digests do not permit whitespace, and
                       silently trimming would mask an operator's
                       `--set-string 'image.tag= v1.1.0 '` typo
                       until the kubelet's pull fails. PR #9682
                       follow-up: `--set-string image.tag=" v1.1.0 "`
                       was previously rendering `repo: v1.1.0 ` (the
                       latest-comparison branch trimmed but the
                       render path used the raw value).
  * `tag == "latest"`→ fail. Case-insensitive (so
                       `--set image.tag=LATEST` fails too). The
                       comparison is against the literal canonical
                       form, NOT a substring — release tags that
                       happen to contain "latest" (e.g.
                       `v1.0.0-latest-rc`) are still accepted.
  * `tag` starts with `sha256:` → fail. Catches the original PR9682
                       bug of putting a digest on the tag field
                       (would render the invalid reference
                       `repo:sha256:...`).
  * `digest` not exactly `sha256:<64 hex>` → fail. SHA-256 is
                       always 64 hex chars (32 bytes × 2 nybbles).
                       Anything shorter or longer is a typo /
                       truncation and must be rejected — pre-fix
                       this allowed `{32,}` which accepted
                       half-digests like
                       `sha256:abc123def4567890abc123def4567890`
                       (32 chars). Per PR9682 follow-up.

A passing chart install is guaranteed to render either
`{repo}:{tag}` (no `latest`, no whitespace) or
`{repo}@sha256:<64 hex>` — both are valid OCI image references.
*/}}
{{- define "power-agent.validateImageTag" -}}
{{- if .Values.daemonset.enabled -}}
{{- $tag := .Values.image.tag | toString -}}
{{- $digest := .Values.image.digest | toString -}}
{{- $appTag := printf "v%s" .Chart.AppVersion -}}
{{- if and (not $tag) (not $digest) -}}
{{- fail (printf "image.tag or image.digest is required (pin to a release tag like %s or a digest like sha256:abc...; :latest is not supported)" $appTag) -}}
{{- end -}}
{{- if and $tag $digest -}}
{{- fail (printf "image.tag and image.digest are mutually exclusive (got tag=%q digest=%q). Pick one — digest gives strict content-addressed reproducibility, tag is human-readable." $tag $digest) -}}
{{- end -}}
{{- if $tag -}}
{{/* PR9790 follow-up: `ne $tag (trim $tag)` only caught leading/
    trailing whitespace. Internal whitespace (e.g. `image.tag="v1 .2.0"`
    from a copy/paste error or a YAML-quoted typo) passed through and
    rendered an invalid OCI image reference like `repo:v1 .2.0`,
    deferring failure from template time to kubelet pull time.
    `regexMatch "\\s" $tag` catches any whitespace anywhere. */}}
{{- if regexMatch "\\s" $tag -}}
{{- fail (printf "image.tag=%q contains whitespace; OCI tags do not permit whitespace and rendering it verbatim would produce an invalid image reference (e.g. `repo: %s ` or `repo:v1 .2.0`). Fix the --set / values.yaml input. Hint: --set-string preserves whitespace exactly as quoted." $tag $appTag) -}}
{{- end -}}
{{- $tagLower := $tag | lower -}}
{{- if eq $tagLower "latest" -}}
{{- fail (printf "image.tag=%q is not supported. Pin a release tag (e.g. %s) or set image.digest=sha256:<hex>. The :latest tag was deliberately rejected on PR #9682 review to keep deployments reproducible." $tag $appTag) -}}
{{- end -}}
{{- if hasPrefix "sha256:" $tagLower -}}
{{- fail (printf "image.tag=%q looks like a digest (starts with sha256:) — digests must go on image.digest, NOT image.tag. Rendering %q:%q would produce an invalid OCI image reference (`repo:sha256:...` parses as repo + tag \"sha256\" with a stray suffix). Use `--set image.tag=\"\" --set image.digest=%s` instead. PR #9682 added image.digest as a separate field for exactly this reason." $tag .Values.image.repository $tag $tag) -}}
{{- end -}}
{{- end -}}
{{- if $digest -}}
{{/* Same symmetry as $tag above: trim only catches edge whitespace.
    The strict sha256-format regex below would already reject internal
    whitespace by construction, but firing the dedicated whitespace
    error first gives operators a clearer diagnosis than a generic
    "not a valid SHA-256 digest" message. */}}
{{- if regexMatch "\\s" $digest -}}
{{- fail (printf "image.digest=%q contains whitespace; OCI digests do not permit whitespace. Fix the --set / values.yaml input." $digest) -}}
{{- end -}}
{{- if not (regexMatch "^sha256:[0-9a-fA-F]{64}$" $digest) -}}
{{- fail (printf "image.digest=%q is not a valid SHA-256 digest. Must match sha256:<64 hex chars> exactly (SHA-256 is 32 bytes = 64 nybbles). Anything shorter is a truncated digest; anything longer is a typo. Example: image.digest=sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855. PR #9682 follow-up tightened this from {32,} to {64}." $digest) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Validate that dev.image.tag is set when dev-pod mode is enabled. Mirror of
validateImageTag for the dev iteration image; keeps dev installs from silently
falling back to a mutable tag.
*/}}
{{- define "power-agent.validateDevImageTag" -}}
{{- if .Values.dev.enabled -}}
{{- $tag := .Values.dev.image.tag | toString -}}
{{- if not $tag -}}
{{- fail "dev.image.tag is required when dev.enabled (pin the dev iteration image to a release tag; :latest is not supported)" -}}
{{- end -}}
{{/* Mirror the production validateImageTag guards: the error text above
    promises ":latest is not supported", so actually enforce it (and reject
    whitespace) rather than accepting any non-empty string. */}}
{{- if regexMatch "\\s" $tag -}}
{{- fail (printf "dev.image.tag=%q contains whitespace; OCI tags do not permit whitespace and rendering it verbatim would produce an invalid image reference. Fix the --set / values.yaml input." $tag) -}}
{{- end -}}
{{- if eq (lower $tag) "latest" -}}
{{- fail (printf "dev.image.tag=%q is not supported. Pin the dev iteration image to a concrete release tag; the :latest tag is rejected to keep dev installs reproducible (mirrors the production image.tag guard)." $tag) -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Render the canonical image reference for a given repository / tag /
digest set. Used by daemonset.yaml + dev-pod.yaml so the two stay in
lockstep on digest handling.

  * `image.digest` set → `{repository}@{digest}` (OCI digest form)
  * else                → `{repository}:{tag}` (OCI tag form)

Assumes the relevant tag validator has already run on the same value
tree at the top of the including file: daemonset.yaml includes
`validateImageTag`, dev-pod.yaml includes its mirror
`validateDevImageTag`. So we don't re-validate here.

Usage: `image: {{ include "power-agent.imageRef" (dict "repository"
.Values.image.repository "tag" .Values.image.tag "digest"
.Values.image.digest) | quote }}`

Validator guarantees both tag and digest are whitespace-free at
this point, so we render them verbatim — silent trimming here
would mask validator regressions.
*/}}
{{- define "power-agent.imageRef" -}}
{{- if .digest -}}
{{- printf "%s@%s" .repository .digest -}}
{{- else -}}
{{- printf "%s:%s" .repository .tag -}}
{{- end -}}
{{- end -}}

{{/*
Validate that production DaemonSet mode and in-cluster dev-pod mode are
not both enabled, and that dev mode has a pinned nodeName. Surfaces at
`helm install` / `helm template` time, not as two competing Pods at runtime.
*/}}
{{- define "power-agent.validateMutex" -}}
{{- if and .Values.daemonset.enabled .Values.dev.enabled -}}
{{- fail "daemonset.enabled and dev.enabled are mutually exclusive. Set exactly one." -}}
{{- end -}}
{{- if and .Values.dev.enabled (not .Values.dev.nodeName) -}}
{{- fail "dev.enabled requires dev.nodeName (the GPU node to pin the dev pod to)." -}}
{{- end -}}
{{- end -}}

{{/*
Validate the pod termination grace period. SIGTERM cleanup restores managed GPU
caps from run()'s finally after any in-flight pod LIST returns, so very small
values defeat the safety invariant this knob exists to protect.
*/}}
{{- define "power-agent.validateTerminationGracePeriod" -}}
{{- $grace := int .Values.terminationGracePeriodSeconds -}}
{{- if lt $grace 60 -}}
{{- fail (printf "terminationGracePeriodSeconds=%d is too low for safe Power Agent shutdown; must be >= 60 seconds so SIGTERM cleanup has time to restore managed GPU caps after an in-flight pod LIST returns. Default is 60." $grace) -}}
{{- end -}}
{{- end -}}

{{/*
Validate the actuator selection. Catches typos at `helm install` /
`helm template` time so the operator doesn't discover their mistake
when the pod CrashLoopBackOffs with argparse's terse "invalid choice"
error.

The Power Agent's --actuator CLI flag has the same choices=["nvml",
"dcgm"] guard, but surfacing the error at template time gives a
clearer message and keeps a misconfigured install from ever creating
Pod objects (no leftover ImagePullBackOff, no leftover RBAC).
*/}}
{{- define "power-agent.validateActuator" -}}
{{- $a := .Values.agent.actuator | default "" -}}
{{- if not (or (eq $a "nvml") (eq $a "dcgm")) -}}
{{- fail (printf "agent.actuator must be 'nvml' or 'dcgm'; got %q" $a) -}}
{{- end -}}
{{- end -}}

{{/*
Effective RBAC scope.

Dev mode pins to one node and one namespace, so cluster-wide pod-listing
RBAC would be excessive. The agent's --namespace CLI flag
(power_agent.py `_list_pods_on_node`) already constrains its pod queries to a
single namespace when set; the dev-pod template passes --namespace=$(POD_NAMESPACE)
via the downward API. This helper makes the RBAC default match the agent's
actual reach in dev mode.

An operator can still opt back into cluster-wide RBAC in dev mode by
setting --set dev.namespaceRestrictedOverride=true. Without that flag,
the dev-mode default wins regardless of rbac.namespaceRestricted.

Returns a Go-string "true" / "false" because Helm template conditionals
compare against the boolean values produced by YAML, but downstream
templates use it in `eq ... "true"` form for clarity.
*/}}
{{- define "power-agent.effectiveNamespaceRestricted" -}}
{{- if and .Values.dev.enabled (not .Values.dev.namespaceRestrictedOverride) -}}
true
{{- else -}}
{{- .Values.rbac.namespaceRestricted | toString -}}
{{- end -}}
{{- end -}}
