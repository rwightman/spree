.PHONY: init build-dos up up-dos down logs scfg shell install-js-tw2 stage-dos-doors smoke test status doctor

init:
	./scripts/bootstrap.sh

build-dos:
	docker compose -f docker-compose.yml -f docker-compose.dos.yml build bbs

up: init
	docker compose up -d

up-dos: init
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

stage-dos-doors:
	./scripts/stage_dos_doors.sh

smoke:
	python -m bbs_gym.cli smoke --host "$${BBS_HOST:-127.0.0.1}" --port "$${TELNET_PORT:-2323}"

test:
	python -m pytest tests
