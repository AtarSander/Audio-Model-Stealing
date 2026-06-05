# Audio Model Stealing

This repository now contains a modern PyTorch/Hugging Face reimplementation of
the classifier and QA extraction pipelines from *Thieves on Sesame Street!
Model Extraction of BERT-based APIs*.

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

## Configuration

The modern pipelines use Hydra configs instead of argparse flags. Each runner
loads a small package-local experiment config:

```text
modern_bert_extraction/configs/classifier.yaml
modern_bert_extraction/configs/squad11.yaml
modern_bert_extraction/configs/boolq.yaml
modern_audio_extraction/configs/hubert_stolenencoder.yaml
modern_audio_extraction/configs/collect_results.yaml
```

Those top-level files compose reusable config groups instead of storing every
setting inline. The BERT configs are split into:

```text
model/              pretrained encoder defaults
paths/              data, output, and checkpoint roots
training/           task-specific training hyperparameters
query_generation/   classifier and QA query-generation defaults
runtime/            device and CUDA behavior
```

The audio config is split into:

```text
target/             target encoder
student/            surrogate encoder
query_source/       audio query source
training/           StolenEncoder training loss and optimizer settings
downstream/         optional downstream probe setup
runtime/            device, precision, and batch-size runtime settings
matrix/             matrix experiment budgets, students, and query sources
```

Override values directly from the command line with Hydra dotlist syntax:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  run.task=SST-2 \
  run.scheme=random \
  run.step=generate_queries \
  query_generation.dataset_size=32
```

Swap whole config groups when there is a reusable alternative:

```bash
uv run -m modern_audio_extraction.run_audio_encoder_pipeline \
  student=wav2vec2_base \
  query_source=librispeech_asr
```

Preview the resolved config without running an experiment:

```bash
uv run -m modern_audio_extraction.run_audio_encoder_pipeline --cfg job
```

## Data

Download GLUE:

```bash
make download_glue
```

Download WikiText-103 raw splits:

```bash
make download_wikitext_hf
```

Download SQuAD 1.1 and BoolQ:

```bash
make download_qa_data
```

Or all reproduction data:

```bash
make download_modern_data
```

This writes:

```text
data/raw/glue/
data/raw/wikitext103/wiki.train.raw
data/raw/wikitext103/wiki.valid.raw
data/raw/wikitext103/wiki.test.raw
data/raw/squad/train-v1.1.json
data/raw/squad/dev-v1.1.json
data/raw/boolq/train.jsonl
data/raw/boolq/dev.jsonl
data/processed/wikitext103/wikitext103-sentences.txt
data/processed/wikitext103/wikitext103-paragraphs.txt
```

The reproduction layout is split by artifact type:

```text
data/raw/            downloaded datasets
data/processed/      generated reusable dataset caches
output/repro/        generated queries, metrics, summaries, and reports
checkpoints/repro/   victim, extracted, and stolen model checkpoints
```

## Modern Classifier Reproduction

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

After the RANDOM victim models have been trained, use the reuse targets for
WIKI paper comparisons. This avoids retraining equivalent victims and evaluates
the WIKI extracted models against the same task victims:

```bash
make modern_reproduce_wiki_reuse_victims
```

Run a single stage manually:

```bash
uv run -m modern_bert_extraction.run_classifier_pipeline \
  run.task=SST-2 \
  run.scheme=random \
  run.step=generate_queries \
  query_generation.dataset_size=32
```

Outputs are written under:

```text
output/repro/modern_classifier/
checkpoints/repro/modern_classifier/
```

## Modern QA Reproduction

SQuAD 1.1 and BoolQ use the same extraction lifecycle, but with the paper's QA
query-generation logic:

1. fine-tune a victim BERT model on SQuAD 1.1 or BoolQ
2. generate `RANDOM` or `WIKI` thief queries from WikiText paragraphs
3. query the victim model
4. build a distilled training set from victim answers/probabilities
5. train an extracted model
6. evaluate task accuracy and victim/extracted agreement

Configs:

- [modern_bert_extraction/configs/squad11.yaml](/home/atarsander/University/SPZC/audio_model_stealing/modern_bert_extraction/configs/squad11.yaml)
- [modern_bert_extraction/configs/boolq.yaml](/home/atarsander/University/SPZC/audio_model_stealing/modern_bert_extraction/configs/boolq.yaml)

Run SQuAD 1.1:

```bash
make modern_reproduce_squad11_random
make modern_reproduce_squad11_wiki
make modern_reproduce_squad11_wiki_reuse_victim
```

Run BoolQ:

```bash
make modern_reproduce_boolq_random
make modern_reproduce_boolq_wiki
make modern_reproduce_boolq_wiki_reuse_victim
```

Outputs are written under:

```text
output/repro/modern_qa/
checkpoints/repro/modern_qa/
```

## Modern Audio Encoder Extraction

The second-stage audio pipeline lives in
[modern_audio_extraction](/home/atarsander/University/SPZC/audio_model_stealing/modern_audio_extraction).
It adapts the StolenEncoder setup to HuBERT-style audio encoders:

1. query a target encoder such as `facebook/hubert-base-ls960`
2. cache target hidden states for a finite query budget
3. train a surrogate encoder with the StolenEncoder feature-matching and augmentation loss
4. evaluate feature similarity and optional downstream probe transfer

Default config:

- [modern_audio_extraction/configs/hubert_stolenencoder.yaml](/home/atarsander/University/SPZC/audio_model_stealing/modern_audio_extraction/configs/hubert_stolenencoder.yaml)

Run the default HuBERT experiment:

```bash
make modern_audio_hubert
```

Run the matrix over query budgets, surrogate architectures, and query sources:

```bash
make modern_audio_hubert_matrix
```

Collect text and audio results into one comparison table:

```bash
make modern_collect_results
```

Outputs are written under:

```text
output/repro/modern_audio/
output/repro/comparison/
checkpoints/repro/modern_audio/
```

## Notes

- The implementation is framework-modernized, but the query-generation and
  distillation logic was checked directly against the original paper code paths.
- The default model is `google-bert/bert-large-uncased`.
- The pipeline requires a working CUDA PyTorch stack if `require_gpu: true`
  remains enabled in the config.
