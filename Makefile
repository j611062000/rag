PYTHON ?= python3
PIP ?= pip3
DOCKER_COMPOSE ?= docker-compose
DOCKER_FILE := docker/docker-compose.yml

.PHONY: start stop docker-build docker-up docker-down api install test ingest

start:
	./start.sh

stop:
	./stop.sh

docker-build:
	$(DOCKER_COMPOSE) -f $(DOCKER_FILE) build

docker-up:
	$(DOCKER_COMPOSE) -f $(DOCKER_FILE) up --build

docker-down:
	$(DOCKER_COMPOSE) -f $(DOCKER_FILE) down

api:
	$(PYTHON) -m app.main

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest

ingest:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make ingest FILE=/path/to/doc.pdf"; \
		exit 1; \
	fi
	$(PYTHON) scripts/ingest_pdfs.py $(FILE)
