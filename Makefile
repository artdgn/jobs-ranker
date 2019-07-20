REPO_NAME=jobs-ranker
VENV_NAME=venv
VENV_ACTIVATE=. $(VENV_NAME)/bin/activate
PYTHON=$(VENV_NAME)/bin/python3
DOCKER_TAG=artdgn/$(REPO_NAME)
DOCKER_DATA_ARG=-v $(realpath ./data):/data
DOCKER_TIME_ARG=-e TZ=$(shell cat /etc/timezone)

venv:
	python3.6 -m venv $(VENV_NAME)

requirements.txt: venv
	$(VENV_ACTIVATE); \
	pip install -U pip pip-tools; \
	pip-compile requirements.in

install: venv requirements.txt
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
	docker run --rm -it \
	$(DOCKER_DATA_ARG) \
	$(DOCKER_TIME_ARG) \
	--name $(REPO_NAME) \
	$(DOCKER_TAG) bash

docker-server: build-docker
	docker run -dit \
	$(DOCKER_DATA_ARG) \
	$(DOCKER_TIME_ARG) \
	--name $(REPO_NAME) \
	--restart unless-stopped \
	-p 5000:5000 \
	$(DOCKER_TAG) python server.py

docker-server-update:
	docker rm -f $(REPO_NAME) || sleep 1
	make docker-server

docker-server-logs:
	docker logs $(REPO_NAME) -f

chown-dirs:
	@echo chown $(shell id -u):$(shell id -u) -R .
