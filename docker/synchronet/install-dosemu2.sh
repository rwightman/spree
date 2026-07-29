#!/usr/bin/env bash
set -euo pipefail

: "${DOSEMU2_VERSION:?DOSEMU2_VERSION must be set}"

dosemu_key_fingerprint="6D9CD73B401A130336ED0A56EBE1B5DED2AD45D6"
dosemu_key_file="$(mktemp)"
trap 'rm -f "$dosemu_key_file"' EXIT

curl -fsSL \
  "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${dosemu_key_fingerprint}" \
  -o "$dosemu_key_file"

actual_fingerprint="$(
  gpg --batch --show-keys --with-colons "$dosemu_key_file" \
    | awk -F: '$1 == "fpr" { print $10; exit }'
)"
if [[ "$actual_fingerprint" != "$dosemu_key_fingerprint" ]]; then
  echo "Unexpected DOSEMU2 PPA signing key: $actual_fingerprint" >&2
  exit 1
fi

gpg --batch --yes --dearmor \
  --output /usr/share/keyrings/dosemu2-ppa.gpg \
  "$dosemu_key_file"
printf '%s\n' \
  'deb [signed-by=/usr/share/keyrings/dosemu2-ppa.gpg] https://ppa.launchpadcontent.net/dosemu2/ppa/ubuntu noble main' \
  > /etc/apt/sources.list.d/dosemu2-ppa.list

apt-get update
apt-get install -y --no-install-recommends "dosemu2=${DOSEMU2_VERSION}"
rm -rf /var/lib/apt/lists/*

