#!/usr/bin/env bash
set -euo pipefail

: "${SYNCHRONET_COMMIT:?SYNCHRONET_COMMIT must be set}"

make \
  -f /sbbs/repo/install/install-sbbs.mk \
  RELEASE=1 \
  NO_X=1 \
  NOCAP=1 \
  SBBSDIR=/sbbs \
  SBBSUSER=root \
  SBBSGROUP=root \
  install

cp /sbbs/exec/node /sbbs/exec/sbbsnode
{
  /sbbs/exec/sbbs version
  printf 'Git Hash %s\n' "$SYNCHRONET_COMMIT"
} > /sbbs/exec/version.txt

sed -i -E 's/^[[:space:]]*Interface[[:space:]]*=.*/    Interface = 0.0.0.0/' /sbbs/ctrl/sbbs.ini
sed -i -E 's/^[[:space:]]*UseDOSemu[[:space:]]*=.*/UseDOSemu = true/' /sbbs/ctrl/sbbs.ini
sed -i -E 's|^[[:space:]]*DOSemuPath[[:space:]]*=.*|DOSemuPath = /usr/bin/dosemu.bin|' /sbbs/ctrl/sbbs.ini
sed -i 's|\.\./webv4|../web|g' /sbbs/ctrl/modopts.ini

mkdir -p /sbbs/dist
mv /sbbs/ctrl /sbbs/dist/ctrl
mv /sbbs/docs /sbbs/dist/docs
mv /sbbs/text /sbbs/dist/text
mv /sbbs/xtrn /sbbs/dist/xtrn
mv /sbbs/web /sbbs/dist/web-runemaster
mv /sbbs/webv4 /sbbs/dist/web-ecweb4
mv /sbbs/node1 /sbbs/dist/node1

rm -rf /sbbs/node2 /sbbs/node3 /sbbs/node4 /sbbs/3rdp /sbbs/repo /sbbs/src
