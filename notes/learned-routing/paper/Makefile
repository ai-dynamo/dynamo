# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the learned-routing draft into build/main.pdf (pdflatex + BibTeX via latexmk).
#   make          build the PDF
#   make tables   regenerate generated/*.tex from the final report's table fragments
#                 (CR=<campaign root> make tables; see scripts/make_tables.py)
#   make figures  redraw the report's figures as generated/fig_*.tex (pgfplots) from the
#                 report's data (CR=<campaign root> make figures; see scripts/make_figures.py)
#   make check    build, then list undefined references/citations and overfull boxes
#   make watch    rebuild on change
#   make clean    remove build/ (nothing else)

PAPER_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
LATEXMK   ?= latexmk
BUILD     := $(PAPER_DIR)/build

.PHONY: all pdf tables figures check watch clean

all: pdf

pdf:
	cd $(PAPER_DIR) && $(LATEXMK) main.tex

PYTHON ?= python3
tables:
	@test -n "$(CR)" || { echo "set CR to the campaign root"; exit 1; }
	cd $(PAPER_DIR) && $(PYTHON) scripts/make_tables.py --tables "$(CR)/report/data/tables" --out generated

figures:
	@test -n "$(CR)" || { echo "set CR to the campaign root"; exit 1; }
	cd $(PAPER_DIR) && $(PYTHON) scripts/make_figures.py --cr "$(CR)" --out generated

check: pdf
	@cd $(PAPER_DIR) && { \
	  grep -nE "Warning: (Reference|Citation|Label).*(undefined|multiply)" build/main.log || true; \
	  grep -nE "^(Overfull|Underfull) \\\\hbox" build/main.log | head -20 || true; \
	  grep -nE "Warning--" build/main.blg || true; \
	  printf 'pending markers: '; grep -o '\\pending{' main.tex sections/*.tex | wc -l; \
	  printf 'citeneeded markers: '; grep -o '\\citeneeded{' sections/*.tex | wc -l; }

watch:
	cd $(PAPER_DIR) && $(LATEXMK) -pvc -view=none main.tex

clean:
	rm -rf $(BUILD)
