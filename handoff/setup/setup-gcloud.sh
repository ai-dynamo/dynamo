#!/usr/bin/env bash

set -euo pipefail

PROJECT="${DYNAMO_GCP_PROJECT:-blaise-478114}"

if ! command -v gcloud >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    brew install --cask google-cloud-sdk
    if [ -d /opt/homebrew/share/google-cloud-sdk/bin ]; then
      export PATH="/opt/homebrew/share/google-cloud-sdk/bin:$PATH"
    fi
  else
    echo "ERROR: install Google Cloud SDK first: https://cloud.google.com/sdk/docs/install" >&2
    exit 1
  fi
fi

if [ -z "$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)" ]; then
  gcloud auth login
fi

if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
  gcloud auth application-default login
fi

if gcloud projects describe "$PROJECT" >/dev/null 2>&1; then
  gcloud config set project "$PROJECT" >/dev/null
else
  echo "WARNING: project $PROJECT is not accessible with the active account"
fi

gcloud config set compute/region asia-south1 >/dev/null || true
gcloud config set compute/zone asia-south1-b >/dev/null || true

gcloud config list
gcloud compute instances list --filter='name=instance-20260415-161450'
