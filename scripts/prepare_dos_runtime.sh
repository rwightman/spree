#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${1:-$repo_root/runtime/sbbs}"
sbbs_ini="$runtime_root/ctrl/sbbs.ini"

if [[ ! -f "$sbbs_ini" ]]; then
  echo "Fresh runtime detected; the DOS image will initialize DOSEMU2 defaults."
  exit 0
fi

set_ini_value() {
  local key="$1"
  local value="$2"

  if grep -Eq "^[[:space:]]*$key[[:space:]]*=" "$sbbs_ini"; then
    sed -i -E "s|^[[:space:]]*$key[[:space:]]*=.*|$key = $value|" "$sbbs_ini"
  else
    printf '%s = %s\n' "$key" "$value" >> "$sbbs_ini"
  fi
}

set_ini_value UseDOSemu true
set_ini_value DOSemuPath /usr/bin/dosemu.bin
set_ini_value DOSemuConfPath dosemu.conf
cp "$repo_root/docker/synchronet/dosemu.conf" "$runtime_root/ctrl/dosemu.conf"

echo "Enabled DOSEMU2 in $sbbs_ini"
