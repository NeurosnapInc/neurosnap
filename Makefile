.PHONY: docs docs-api docs-html

docs-api:
	sphinx-apidoc --implicit-namespaces -f -e -M -o ./docs/source ./src/neurosnap
	python scripts/patch_apidoc_titles.py

docs-html:
	cd docs && make html

docs: docs-api docs-html
