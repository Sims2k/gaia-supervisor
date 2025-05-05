.PHONY: all format lint test tests test_watch integration_tests docker_tests help extended_tests

# Default target executed when no arguments are given to make.
all: help

# Define a variable for the test file path.
TEST_FILE ?= tests/unit_tests/

test:
	python -m pytest $(TEST_FILE)

test_watch:
	python -m ptw --snapshot-update --now . -- -vv tests/unit_tests

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg

extended_tests:
	python -m pytest --only-extended $(TEST_FILE)


######################
# LINTING AND FORMATTING
######################

# Define a variable for Python and notebook files.
PYTHON_FILES=src/
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check .
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format format_diff:
	ruff format $(PYTHON_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILE=<test_file>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'

# See https://tech.davis-hansson.com/p/make/
SHELL := bash
.DELETE_ON_ERROR:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-print-directory

# Set up python environment
.PHONY: venv
venv:  ## Set up a virtual environment
	python -m venv venv

.PHONY: deps
deps:  ## Install dependencies
	pip install -e "."

.PHONY: deps-dev
deps-dev: deps  ## Install dev dependencies
	pip install -e ".[dev]"

.PHONY: deps-youtube
deps-youtube:  ## Install YouTube-related dependencies
	pip install youtube-transcript-api pytube

.PHONY: deps-all
deps-all: deps-dev deps-youtube  ## Install all dependencies including YouTube tools

.PHONY: clean
clean:  ## Clean up
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +

.PHONY: format
format:  ## Format code
	ruff format .

.PHONY: lint
lint:  ## Lint code
	ruff check . --fix

.PHONY: check
check:  ## Run code quality checks
	ruff check .
	ruff format --check .

.PHONY: test
test:  ## Run tests
	python -m pytest

.PHONY: test-cov
test-cov:  ## Run tests with coverage
	python -m pytest --cov=langgraph.prebuilt --cov-report=term-missing

.PHONY: deploy-docs
deploy-docs:  ## Deploy documentation
	cd docs && vercel --prod

.PHONY: watch-docs
watch-docs:  ## Watch documentation
	cd docs && npx vercel

.PHONY: build-docs
build-docs:  ## Build documentation
	cd docs && npm run docs:build

.PHONY: app
app:  ## Run the app
	python -m src.react_agent.app

.PHONY: poetry-app
poetry-app:  ## Run the app with poetry
	poetry run python .\src\react_agent\app.py

.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

