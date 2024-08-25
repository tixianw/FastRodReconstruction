#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

# Define color codes
BOLD := \033[1m
GREEN := \033[0;32m
RESET := \033[0m

#* Poetry: the dependency management and packaging tool for Python
.PHONY: install-poetry
install-poetry:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) -
	@if ! echo $$PATH | grep -q "$(HOME)/.local/bin"; then \
		echo "================================="; \
		echo "Adding $(HOME)/.local/bin to PATH"; \
		echo 'export PATH="$(HOME)/.local/bin:$$PATH"' >> $(HOME)/.bashrc; \
		echo "Please run $(GREEN)'source $(HOME)/.bashrc'$(RESET) to activate poetry"; \
		echo "================================="; \
	fi

.PHONY: uninstall-poetry
uninstall-poetry:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) - --uninstall

#* Installation
.PHONY: check-env install

# Store the initial environment path
INITIAL_ENV := $(shell poetry env info --path 2>/dev/null)

check-env:
	@echo "Initial environment: $(INITIAL_ENV)"

install: check-env
	poetry export --without-hashes > requirements.txt
	poetry install -n
	@NEW_ENV=$$(poetry env info --path 2>/dev/null); \
	if [ "$$NEW_ENV" != "$(INITIAL_ENV)" ]; then \
		echo "================================="; \
		echo "New virtual environment detected at: $$NEW_ENV"; \
		echo "To activate environment and run Python with the package, please run $(GREEN)'make activate-env'$(RESET)"; \
		echo "To deactivate environment, please run $(GREEN)'exit'$(RESET)"; \
		echo "================================="; \
	fi

.PHONY: activate-env
activate-env:
	poetry shell

#* Installation of pre-commit: tool of Git hook scripts
.PHONY: install-pre-commit
install-pre-commit:
	poetry run pre-commit install

#* Unittests
.PHONY: pytest
pytest:
	poetry run pytest -c pyproject.toml --cov=src/ros2_vicon

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Check type-hinting
.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml src/

#* Linting
.PHONY: test_ci
test_ci:
	poetry run pytest -c pyproject.toml --cov=src/cobra --cov-report=xml

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./

.PHONY: lint
lint: test_ci check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D "isort[colors]@latest" mypy@latest pre-commit@latest pydocstyle@latest pylint@latest pytest@latest pyupgrade@latest coverage@latest pytest-html@latest pytest-cov@latest black@latest

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
