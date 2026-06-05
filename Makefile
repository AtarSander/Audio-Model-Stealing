#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = audio_model_stealing
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
UV = uv
MAIN_VENV = .venv
MODERN_RUN = $(UV) run
GLUE_DATA_DIR = data/raw/glue
WIKITEXT103_DIR = data/raw/wikitext103
SQUAD_DATA_DIR = data/raw/squad
BOOLQ_DATA_DIR = data/raw/boolq
SST2_RANDOM_VICTIM_DIR = checkpoints/repro/modern_classifier/sst2_random/victim_model
MNLI_RANDOM_VICTIM_DIR = checkpoints/repro/modern_classifier/mnli_random/victim_model
SQUAD11_RANDOM_VICTIM_DIR = checkpoints/repro/modern_qa/squad11_random/victim_model
BOOLQ_RANDOM_VICTIM_DIR = checkpoints/repro/modern_qa/boolq_random/victim_model

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


## Download the datasets needed by the modern reproduction pipelines
.PHONY: download_modern_data
download_modern_data: download_glue download_wikitext_hf download_qa_data


## Download SQuAD 1.1 and BoolQ data into the repo-local reproduction directory
.PHONY: download_qa_data
download_qa_data:
	$(MODERN_RUN) python scripts/download_qa_data.py --squad-dir $(SQUAD_DATA_DIR) --boolq-dir $(BOOLQ_DATA_DIR) --tasks all


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
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=SST-2 run.scheme=random


## Run the modern PyTorch classifier pipeline for SST-2 WIKI
.PHONY: modern_reproduce_sst2_wiki
modern_reproduce_sst2_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=SST-2 run.scheme=wiki


## Run the modern PyTorch classifier pipeline for SST-2 WIKI using the existing RANDOM victim
.PHONY: modern_reproduce_sst2_wiki_reuse_victim
modern_reproduce_sst2_wiki_reuse_victim:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=SST-2 run.scheme=wiki paths.victim_model_dir=$(SST2_RANDOM_VICTIM_DIR)


## Run the modern PyTorch classifier pipeline for MNLI RANDOM
.PHONY: modern_reproduce_mnli_random
modern_reproduce_mnli_random:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=MNLI run.scheme=random


## Run the modern PyTorch classifier pipeline for MNLI WIKI
.PHONY: modern_reproduce_mnli_wiki
modern_reproduce_mnli_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=MNLI run.scheme=wiki


## Run the modern PyTorch classifier pipeline for MNLI WIKI using the existing RANDOM victim
.PHONY: modern_reproduce_mnli_wiki_reuse_victim
modern_reproduce_mnli_wiki_reuse_victim:
	$(MODERN_RUN) -m modern_bert_extraction.run_classifier_pipeline run.task=MNLI run.scheme=wiki paths.victim_model_dir=$(MNLI_RANDOM_VICTIM_DIR)


## Run the modern PyTorch QA pipeline for SQuAD 1.1 RANDOM
.PHONY: modern_reproduce_squad11_random
modern_reproduce_squad11_random:
	$(MODERN_RUN) -m modern_bert_extraction.run_squad_pipeline run.scheme=random


## Run the modern PyTorch QA pipeline for SQuAD 1.1 WIKI
.PHONY: modern_reproduce_squad11_wiki
modern_reproduce_squad11_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_squad_pipeline run.scheme=wiki


## Run the modern PyTorch QA pipeline for SQuAD 1.1 WIKI using the existing RANDOM victim
.PHONY: modern_reproduce_squad11_wiki_reuse_victim
modern_reproduce_squad11_wiki_reuse_victim:
	$(MODERN_RUN) -m modern_bert_extraction.run_squad_pipeline run.scheme=wiki paths.victim_model_dir=$(SQUAD11_RANDOM_VICTIM_DIR)


## Run the modern PyTorch BoolQ pipeline for RANDOM
.PHONY: modern_reproduce_boolq_random
modern_reproduce_boolq_random:
	$(MODERN_RUN) -m modern_bert_extraction.run_boolq_pipeline run.scheme=random


## Run the modern PyTorch BoolQ pipeline for WIKI
.PHONY: modern_reproduce_boolq_wiki
modern_reproduce_boolq_wiki:
	$(MODERN_RUN) -m modern_bert_extraction.run_boolq_pipeline run.scheme=wiki


## Run the modern PyTorch BoolQ pipeline for WIKI using the existing RANDOM victim
.PHONY: modern_reproduce_boolq_wiki_reuse_victim
modern_reproduce_boolq_wiki_reuse_victim:
	$(MODERN_RUN) -m modern_bert_extraction.run_boolq_pipeline run.scheme=wiki paths.victim_model_dir=$(BOOLQ_RANDOM_VICTIM_DIR)


## Run all modern WIKI pipelines with freshly trained per-run victims
.PHONY: modern_reproduce_wiki
modern_reproduce_wiki: modern_reproduce_sst2_wiki modern_reproduce_mnli_wiki modern_reproduce_squad11_wiki modern_reproduce_boolq_wiki


## Run all modern WIKI pipelines using existing RANDOM victims
.PHONY: modern_reproduce_wiki_reuse_victims
modern_reproduce_wiki_reuse_victims: modern_reproduce_sst2_wiki_reuse_victim modern_reproduce_mnli_wiki_reuse_victim modern_reproduce_squad11_wiki_reuse_victim modern_reproduce_boolq_wiki_reuse_victim


## Run the modern HuBERT audio encoder extraction pipeline
.PHONY: modern_audio_hubert
modern_audio_hubert:
	$(MODERN_RUN) -m modern_audio_extraction.run_audio_encoder_pipeline


## Run the HuBERT extraction matrix over query budgets, surrogate architectures, and query sources
.PHONY: modern_audio_hubert_matrix
modern_audio_hubert_matrix:
	$(MODERN_RUN) -m modern_audio_extraction.run_audio_experiment_matrix


## Collect text and audio extraction metrics into one comparison table
.PHONY: modern_collect_results
modern_collect_results:
	$(MODERN_RUN) -m modern_audio_extraction.collect_results


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:48}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT


## Show this help message
.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
