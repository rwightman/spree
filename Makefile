.PHONY: init build-dos prepare-dos up up-dos down logs scfg shell install-js-tw2 reset-js-tw2 grant-js-tw2-turns fetch-sre install-sre patch-sre-reset-time stage-dos-doors smoke test status doctor

TURNS ?= 30
SRE_BACKUP_DIRECTORY ?= runtime/backups/sre-reset-$(shell date -u +%Y%m%dT%H%M%SZ)

init:
	./scripts/bootstrap.sh

build-dos:
	docker compose -f docker-compose.yml -f docker-compose.dos.yml build bbs

prepare-dos: init
	./scripts/prepare_dos_runtime.sh

up: init
	docker compose up -d

up-dos: prepare-dos
	docker compose -f docker-compose.yml -f docker-compose.dos.yml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200 bbs

status:
	docker compose ps

doctor:
	./scripts/doctor.sh

scfg: init
	docker compose run --rm bbs scfg

shell:
	docker compose exec bbs bash

install-js-tw2:
	docker compose exec bbs jsexec install-xtrn.js ../xtrn/tw2 -auto

reset-js-tw2:
	docker compose cp scripts/sbbs_reset_tw2.js bbs:/tmp/sbbs_reset_tw2.js
	docker compose exec -T bbs /sbbs/exec/jsexec /tmp/sbbs_reset_tw2.js

grant-js-tw2-turns:
	test -n "$(PLAYER)" || (echo "usage: make grant-js-tw2-turns PLAYER=RLoginSmoke [TURNS=30]" >&2; exit 2)
	docker compose cp scripts/sbbs_tw2_grant_turns.js bbs:/tmp/sbbs_tw2_grant_turns.js
	docker compose exec -T bbs /sbbs/exec/jsexec /tmp/sbbs_tw2_grant_turns.js "$(PLAYER)" "$(TURNS)"

stage-dos-doors:
	./scripts/stage_dos_doors.sh

fetch-sre:
	./scripts/fetch_sre.sh

install-sre: fetch-sre stage-dos-doors
	docker compose -f docker-compose.yml -f docker-compose.dos.yml exec bbs jsexec install-xtrn.js ../xtrn/sre -auto

patch-sre-reset-time:
	UV_CACHE_DIR=runtime/tmp/uv-cache uv run --no-sync python scripts/patch_sre_reset_time.py --backup-directory "$(SRE_BACKUP_DIRECTORY)"

smoke:
	uv run bbs-gym smoke --host "$${BBS_HOST:-127.0.0.1}" --port "$${TELNET_PORT:-2323}"

test:
	uv run pytest tests
