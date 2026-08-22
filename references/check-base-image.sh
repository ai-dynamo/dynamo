#!/bin/bash
# TB2 base-image policy (TB_spec_quality.md → "Base images"):
#   A Dockerfile FROM must use one of the 10 pre-approved base images UNLESS none of
#   them support the task's dependencies.
#     * If the image is one of the 10 approved families (golang/python/debian/rust/
#       node/ubuntu/eclipse-temurin/ruby/maven/gcc), it MUST be the exact pre-approved
#       pinned image (matched by @sha256 digest) → otherwise FAIL (gating).
#     * If it is some other base, that is allowed (assumed: none of the 10 fit) but is
#       flagged as a non-blocking Warning (the run still passes).
# Applies to both the agent image (environment/Dockerfile) and the verifier image
# (tests/Dockerfile). A bare `FROM <stage>` (multi-stage reference) is skipped.
#
# Pre-approved images (each digest covers a family of variants, per the spec):
#   golang          public.ecr.aws/docker/library/golang:1.24-bookworm@sha256:1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac
#   python          public.ecr.aws/docker/library/python:3.13-slim-bookworm@sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb
#   debian          public.ecr.aws/docker/library/debian:bookworm-slim@sha256:4724b8cc51e33e398f0e2e15e18d5ec2851ff0c2280647e1310bc1642182655d
#   rust            public.ecr.aws/docker/library/rust:1.85-slim@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36
#   node            public.ecr.aws/docker/library/node:22-bookworm-slim@sha256:f3a68cf41a855d227d1b0ab832bed9749469ef38cf4f58182fb8c893bc462383
#   ubuntu          public.ecr.aws/docker/library/ubuntu:24.04@sha256:0d39fcc8335d6d74d5502f6df2d30119ff4790ebbb60b364818d5112d9e3e932
#   eclipse-temurin public.ecr.aws/docker/library/eclipse-temurin:21-jdk-jammy@sha256:25d1276565738d3c805e632a4542c3a7598866ef967f4def6544c15de3a74b14
#   ruby            public.ecr.aws/docker/library/ruby:3.3-slim-bookworm@sha256:e76733e94b3a5893e4a141024ef3a583dc10781dc24becebf74f9c9f9a33e3df
#   maven           public.ecr.aws/docker/library/maven:3.9.9-eclipse-temurin-21@sha256:3a4ab3276a087bf276f79cae96b1af04f53731bec53fb2e651aca79e4b10211e
#   gcc             public.ecr.aws/docker/library/gcc:13-bookworm@sha256:930f2ebe239275fa67226654cb79273ea34eee672ae61c8a39f689c37fb7ac5c
set -u

APPROVED_FAMILIES=" golang python debian rust node ubuntu eclipse-temurin ruby maven gcc "

# digest for an approved family (empty if the family is not approved)
approved_digest_for() {
  case "$1" in
    golang)          echo "sha256:1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac" ;;
    python)          echo "sha256:01f42367a0a94ad4bc17111776fd66e3500c1d87c15bbd6055b7371d39c124fb" ;;
    debian)          echo "sha256:4724b8cc51e33e398f0e2e15e18d5ec2851ff0c2280647e1310bc1642182655d" ;;
    rust)            echo "sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36" ;;
    node)            echo "sha256:f3a68cf41a855d227d1b0ab832bed9749469ef38cf4f58182fb8c893bc462383" ;;
    ubuntu)          echo "sha256:0d39fcc8335d6d74d5502f6df2d30119ff4790ebbb60b364818d5112d9e3e932" ;;
    eclipse-temurin) echo "sha256:25d1276565738d3c805e632a4542c3a7598866ef967f4def6544c15de3a74b14" ;;
    ruby)            echo "sha256:e76733e94b3a5893e4a141024ef3a583dc10781dc24becebf74f9c9f9a33e3df" ;;
    maven)           echo "sha256:3a4ab3276a087bf276f79cae96b1af04f53731bec53fb2e651aca79e4b10211e" ;;
    gcc)             echo "sha256:930f2ebe239275fa67226654cb79273ea34eee672ae61c8a39f689c37fb7ac5c" ;;
    *)               echo "" ;;
  esac
}
# full pinned reference (for the FAIL guidance message)
approved_ref_for() {
  echo "public.ecr.aws/docker/library/$1@$(approved_digest_for "$1")"
}

FAILED=0

check_dockerfile() {  # <file>
  local df="$1" line img name_tag repo digest adigest
  [ -f "$df" ] || return 0
  while IFS= read -r line; do
    # image token after FROM (strip optional --platform=...), drop "AS stage"
    img="$(printf '%s\n' "$line" | sed -E 's/^[[:space:]]*[Ff][Rr][Oo][Mm][[:space:]]+(--platform=[^[:space:]]+[[:space:]]+)?([^[:space:]]+).*/\2/')"
    # skip bare stage references (no registry/tag/digest -> a prior "AS name")
    case "$img" in
      */*|*:*|*@*) : ;;
      *) continue ;;
    esac
    # repo family = basename with registry path and tag stripped; digest = part after @
    name_tag="${img%@*}"
    repo="${name_tag##*/}"; repo="${repo%%:*}"
    if [ "${img#*@}" != "$img" ]; then digest="${img##*@}"; else digest=""; fi

    adigest="$(approved_digest_for "$repo")"
    if [ -n "$adigest" ] && [ "$digest" = "$adigest" ]; then
      : # exact pre-approved image — OK
    elif [ -n "$adigest" ]; then
      # one of the 10 families, but not the pre-approved pinned image -> must abide
      echo "FAIL $df: '$repo' must use the pre-approved pinned image $(approved_ref_for "$repo") (found '$img')"
      FAILED=1
    else
      # not one of the 10 approved families -> allowed, but flag (non-blocking)
      echo "Warning $df: base image '$img' is not a pre-approved base; only use a non-approved base if none of the 10 approved bases (${APPROVED_FAMILIES# }) support the task's dependencies — otherwise switch to an approved image and pin it by digest."
    fi
  done < <(grep -iE '^[[:space:]]*FROM[[:space:]]' "$df")
}

for task_dir in "$@"; do
  check_dockerfile "$task_dir/environment/Dockerfile"
  check_dockerfile "$task_dir/tests/Dockerfile"
done

exit $FAILED
