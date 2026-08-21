.PHONY: help install dev test lint run docker

PY ?= python3

help:
	@echo "install   install package (requires-python >= 3.10)"
	@echo "dev       install with dev extras"
	@echo "test      run pytest"
	@echo "run       run decision-gen locally"
	@echo "docker    build image"

install:
	$(PY) -m pip install .

dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

run:
	$(PY) -m decision_gen

docker:
	docker build -t decision-gen:latest .
