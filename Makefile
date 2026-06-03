#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = audio_model_stealing
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
UV = uv
MAIN_VENV = .venv
MODERN_RUN = $(UV) run
GLUE_DATA_DIR = external/repro/glue
WIKITEXT103_DIR = external/repro/wikitext103
MODERN_CLASSIFIER_CONFIG = modern_bert_extraction/configs/classifier.yaml

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(UV) sync


## Set up the main Python environment
.PHONY: create_environment
create_environment:
	$(UV) venv $(MAIN_VENV) --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\$(MAIN_VENV)\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./$(MAIN_VENV)/bin/activate"


## Download all public GLUE task data into the repo-local reproduction directory
.PHONY: download_glue
download_glue:
	$(MODERN_RUN) python scripts/download_glue_data.py --data-dir $(GLUE_DATA_DIR) --tasks all


## Download WikiText-103 raw splits from Hugging Face into the repo-local reproduction directory
.PHONY: download_wikitext_hf
download_wikitext_hf:
	$(MODERN_RUN) python scripts/download_hf_assets.py wikitext --output-dir $(WIKITEXT103_DIR)


## Download the datasets needed by the modern classifier pipeline
.PHONY: download_modern_data
download_modern_data: download_glue download_wikitext_hf


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(MODERN_RUN) python model_steal/dataset.py


## Run the modern PyTorch classifier pipeline for SST-2 RANDOM
.PHONY: modern_reproduce_sst2_random
modern_reproduce_sst2_random:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline --config $(MODERN_CLASSIFIER_CONFIG) --task SST-2 --scheme random


## Run the modern PyTorch classifier pipeline for SST-2 WIKI
.PHONY: modern_reproduce_sst2_wiki
modern_reproduce_sst2_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline --config $(MODERN_CLASSIFIER_CONFIG) --task SST-2 --scheme wiki


## Run the modern PyTorch classifier pipeline for MNLI RANDOM
.PHONY: modern_reproduce_mnli_random
modern_reproduce_mnli_random:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline --config $(MODERN_CLASSIFIER_CONFIG) --task MNLI --scheme random


## Run the modern PyTorch classifier pipeline for MNLI WIKI
.PHONY: modern_reproduce_mnli_wiki
modern_reproduce_mnli_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline --config $(MODERN_CLASSIFIER_CONFIG) --task MNLI --scheme wiki


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT


## Show this help message
.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
