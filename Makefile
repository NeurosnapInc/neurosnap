.PHONY: docs

docs:
	./venv/bin/sphinx-apidoc -o ./docs/source ./src/neurosnap
	cd docs && make html
