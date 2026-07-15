ifneq ($(wildcard /goinfre/$(USER)),)
UV_ENV := UV_CACHE_DIR=/goinfre/$(USER)/uv-cache \
	HF_HOME=/goinfre/$(USER)/hf-cache \
	UV_PROJECT_ENVIRONMENT=/goinfre/$(USER)/call_me_maybe-venv \
	UV_LINK_MODE=copy
else
UV_ENV :=
endif

.PHONY: install run debug clean lint lint-strict test

install:
	$(UV_ENV) uv sync

run:
	$(UV_ENV) uv run python -m src

debug:
	$(UV_ENV) uv run python -m pdb -m src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .mypy_cache
	rm -rf data/output

lint:
	$(UV_ENV) uv run flake8 . --extend-exclude=.venv,llm_sdk
	$(UV_ENV) uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV_ENV) uv run flake8 . --extend-exclude=.venv,llm_sdk
	$(UV_ENV) uv run mypy . --strict

test:
	$(UV_ENV) uv run python -m pytest tests/ -v