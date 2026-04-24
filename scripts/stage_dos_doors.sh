#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sbbs_xtrn="$root/runtime/sbbs/xtrn"

stage_one() {
  local code="$1"
  local source="$root/doors/$code"
  local target="$sbbs_xtrn/$code"

  if ! find "$source" -mindepth 1 ! -name .gitkeep -print -quit | grep -q .; then
    echo "Skipping $code: put the door distribution files in $source first."
    return 0
  fi

  mkdir -p "$target"
  cp -a "$source"/. "$target"/
  cp "$root/config/install-xtrn/$code/install-xtrn.ini" "$target/install-xtrn.ini"
  echo "Staged $code into $target"
}

mkdir -p "$sbbs_xtrn"
stage_one bre
stage_one tw2002

echo
echo "If the BBS container is running, install matching configs with:"
echo "  docker compose exec bbs jsexec install-xtrn.js ../xtrn/bre -auto"
echo "  docker compose exec bbs jsexec install-xtrn.js ../xtrn/tw2002 -auto"
echo
echo "You still need to run each DOS door's own setup program inside DOSEMU/SCFG"
echo "to set node paths, registration keys, and game resets."

