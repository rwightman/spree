#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-$repo_root/doors/sre}"
archive_url="http://www-cs-students.stanford.edu/~amitp/games/sre/sre-dosbox-ready.zip"
archive_sha256="8ebf902f8dced86fc1ea1c2b6c41eef26ffb78def0f498c2359142bf610fc9c5"
sre_sha256="b4381bb2326580ca2d0ee0bf28997e95dc8d6be9893f2f3d345b396ac627acb9"
reggen_sha256="d5bb58dc55949159c3af74b242859f0a3bdc56718d7b78e90a344137e5de0b1e"

for command_name in curl sha256sum unzip; do
  if ! command -v "$command_name" >/dev/null; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

verify_file() {
  local expected="$1"
  local path="$2"

  printf '%s  %s\n' "$expected" "$path" | sha256sum --check --status
}

if [[ -f "$target/SRE.EXE" && -f "$target/SREREG-8.EXE" ]]; then
  if verify_file "$sre_sha256" "$target/SRE.EXE" \
      && verify_file "$reggen_sha256" "$target/SREREG-8.EXE"; then
    echo "SRE 0.994b is already present and verified in $target"
    exit 0
  fi
  echo "Existing SRE executables in $target do not match the pinned distribution." >&2
  exit 1
fi

mkdir -p "$target"
existing="$(find "$target" -mindepth 1 ! -name .gitkeep -print -quit)"
if [[ -n "$existing" ]]; then
  echo "Refusing to mix the SRE distribution with existing files in $target" >&2
  exit 1
fi

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT
archive="$temp_dir/sre-dosbox-ready.zip"
extract_root="$temp_dir/extracted"

echo "Downloading the SRE archive published by its original author..."
curl --fail --location --proto '=http,https' --output "$archive" "$archive_url"
if ! verify_file "$archive_sha256" "$archive"; then
  echo "Downloaded SRE archive failed its SHA-256 check." >&2
  exit 1
fi

mkdir -p "$extract_root"
unzip -q "$archive" -d "$extract_root"
source_dir="$extract_root/solar-realms-elite"
if [[ ! -d "$source_dir" ]] \
    || ! verify_file "$sre_sha256" "$source_dir/SRE.EXE" \
    || ! verify_file "$reggen_sha256" "$source_dir/SREREG-8.EXE"; then
  echo "The verified archive does not contain the expected SRE 0.994b files." >&2
  exit 1
fi

cp -a "$source_dir"/. "$target"/
echo "Fetched and verified SRE 0.994b into $target"
