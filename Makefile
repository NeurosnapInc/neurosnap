.PHONY: docs docs-api docs-html

ifneq ("$(wildcard .local.secrets.env)","")
include .local.secrets.env
endif

docs-api:
	sphinx-apidoc --implicit-namespaces -f -e -M -o ./docs/source ./src/neurosnap
	python scripts/patch_apidoc_titles.py

docs-html:
	$(MAKE) -C docs html DOCS_SYNC_TARGET="$(DOCS_SYNC_TARGET)"
	@if [ -n "$(DOCS_SYNC_TARGET)" ]; then \
		python scripts/publish_docs.py docs/build/html "$(DOCS_SYNC_TARGET)"; \
	else \
		echo "DOCS_SYNC_TARGET not set; skipping external docs sync."; \
	fi

docs: docs-api docs-html
