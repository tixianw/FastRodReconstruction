#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* Poetry: the dependency management and packaging tool for Python
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry export --without-hashes > requirements.txt
	poetry install -n

#* Installation of pre-commit: tool of Git hook scripts
.PHONY: pre-commit-install
pre-commit-install:
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
