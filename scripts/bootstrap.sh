#!/usr/bin/env bash
set -euo pipefail

mkdir -p runtime/sbbs runtime/transcripts runtime/tmp doors/bre doors/sre doors/tw2002

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "Initialized local runtime directories."
echo "Next: docker compose up -d"
