#!/usr/bin/env bash
set -u

ok=0

check() {
  local label="$1"
  shift
  if "$@" >/tmp/spree-doctor.out 2>/tmp/spree-doctor.err; then
    echo "ok: $label"
  else
    ok=1
    echo "fail: $label"
    sed 's/^/  /' /tmp/spree-doctor.err
  fi
}

check "python" python --version
check "docker cli" docker --version
check "docker compose config" docker compose config
check "docker daemon access" docker ps

if [[ -S /var/run/docker.sock ]]; then
  echo "info: docker socket"
  ls -l /var/run/docker.sock
fi

exit "$ok"

