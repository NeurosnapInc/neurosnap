.PHONY: docs

docs:
	sphinx-apidoc -o ./docs/source ./src/neurosnap
	cd docs && make html
