.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .mypy_cache
	rm -rf data/output

lint:
	uv run flake8 . --extend-exclude=.venv,llm_sdk
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--ignore-incomplete-defs --no-implicit-optional

lint-strict:
	uv run flake8 . --extend-exclude=.venv,llm_sdk
	uv run mypy . --strict