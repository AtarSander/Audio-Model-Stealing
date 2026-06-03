# Audio Model Stealing

This repository now contains a modern PyTorch/Hugging Face reimplementation of
the classifier extraction pipeline from *Thieves on Sesame Street! Model
Extraction of BERT-based APIs*.

The modern implementation lives in [modern_bert_extraction](/home/atarsander/University/SPZC/audio_model_stealing/modern_bert_extraction).
The legacy TensorFlow 1.15 rescue path, Docker scaffolding, and TF1 checkpoint
conversion helpers have been removed.

## Environment

Create the main environment:

```bash
make create_environment
make requirements
source .venv/bin/activate
```

The project uses `uv` for dependency management. The modern run targets are
wired through `uv run`, so activating the environment is optional once
dependencies are installed.

## Data

Download GLUE:

```bash
make download_glue
```

Download WikiText-103 raw splits:

```bash
make download_wikitext_hf
```

Or both:

```bash
make download_modern_data
```

This writes:

```text
external/repro/glue/
external/repro/wikitext103/wiki.train.raw
external/repro/wikitext103/wiki.valid.raw
external/repro/wikitext103/wiki.test.raw
external/repro/wikitext103/wikitext103-sentences.txt
```

## Modern Reproduction

The classifier pipeline keeps the original paper logic:

1. fine-tune a victim BERT model on `SST-2` or `MNLI`
2. generate `RANDOM` or `WIKI` thief queries
3. query the victim for soft labels
4. train an extracted model on the distilled dataset
5. evaluate dev accuracy and victim/extracted agreement

Default config:

- [modern_bert_extraction/configs/classifier.yaml](/home/atarsander/University/SPZC/audio_model_stealing/modern_bert_extraction/configs/classifier.yaml)

Recommended first run:

```bash
make modern_reproduce_sst2_random
```

Other runs:

```bash
make modern_reproduce_sst2_wiki
make modern_reproduce_mnli_random
make modern_reproduce_mnli_wiki
```

Run a single stage manually:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  --config modern_bert_extraction/configs/classifier.yaml \
  --task SST-2 \
  --scheme random \
  --step generate_queries \
  --dataset-size 32
```

Outputs are written under:

```text
external/repro/modern_classifier/
```

## Notes

- The implementation is framework-modernized, but the query-generation and
  distillation logic was checked directly against the original classifier code.
- The default model is `google-bert/bert-large-uncased`.
- The pipeline requires a working CUDA PyTorch stack if `require_gpu: true`
  remains enabled in the config.
