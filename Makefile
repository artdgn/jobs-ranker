REPO_NAME=jobs-ranker
VENV_NAME=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=$(VENV_NAME)/bin/python3
DOCKER_TAG=artdgn/$(REPO_NAME)
DOCKER_DATA_ARG=-v $(realpath ./data):/$(REPO_NAME)/data

venv:
	python3.6 -m venv $(VENV_NAME)

requirements: venv
	$(VENV_ACTIVATE); \
	pip install -U pip pip-tools; \
	pip-compile requirements.in

install: requirements
	$(VENV_ACTIVATE); \
	pip install -r requirements.txt

python:
	@echo $(PYTHON)

server: venv
	$(VENV_ACTIVATE); \
	python server.py

build-docker: requirements.txt
	docker build -t $(DOCKER_TAG) .

push-docker: build-docker
	docker push $(DOCKER_TAG)

docker-bash: build-docker
	docker run --rm -it $(DOCKER_DATA_ARG) $(DOCKER_TAG) bash

docker-server: build-docker
	docker run --rm -it $(DOCKER_DATA_ARG) -p 5000:5000 $(DOCKER_TAG) python server.py

