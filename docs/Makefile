# Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
BUILDDIR      ?= _build

.PHONY: help clean html

help:
	@echo "Usage: make [TARGET]"
	@echo "Targets:"
	@echo "  html       - Build HTML documentation"
	@echo "  clean      - Remove build directory"

clean:
	rm -rf $(BUILDDIR)/*

html:
	$(SPHINXBUILD) -b html . $(BUILDDIR)/html
	@echo "Build finished. HTML available in $(BUILDDIR)/html/index.html"
